"""Shared cancellation boundaries; no hardware or model imports."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from types import SimpleNamespace


class TaskCancelled(RuntimeError):
    pass


_cancel_check = ContextVar("robot_cancel_check", default=None)


def check_cancelled():
    check = _cancel_check.get()
    if check is not None and check():
        raise TaskCancelled("任务已取消，丢弃本次结果")


@contextmanager
def cancellation_scope(check):
    token = _cancel_check.set(check)
    try:
        yield
    finally:
        _cancel_check.reset(token)


def guarded_call(function, *args, **kwargs):
    check_cancelled()
    try:
        result = function(*args, **kwargs)
    finally:
        # Also discard failures from an old request instead of retrying it.
        check_cancelled()
    return result


class CancellationClient:
    """Keep the SDK interface, checking at the actual request boundary.

    Already-sent HTTP requests may finish, but their result is never consumed.
    Task requests disable SDK automatic retries and have a bounded timeout.
    """
    def __init__(self, client):
        self._client = client
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *args, **kwargs):
        check_cancelled()
        client = self._client
        if _cancel_check.get() is not None:
            client = client.with_options(max_retries=0, timeout=45)
        return guarded_call(client.chat.completions.create, *args, **kwargs)

    def with_options(self, **kwargs):
        return type(self)(self._client.with_options(**kwargs))

    def __getattr__(self, name):
        return getattr(self._client, name)


class TaskDriver:
    """Task-scoped view of a persistent driver; cancellation never closes it."""
    def __init__(self, driver, check, lock):
        self._driver, self._check, self._lock = driver, check, lock

    def __getattr__(self, name):
        value = getattr(self._driver, name)
        if not callable(value) or name in ("cancel_move", "close_camera"):
            return value

        @wraps(value)
        def call(*args, **kwargs):
            with cancellation_scope(self._check):
                # Linearize outgoing motion commands with accepting cancellation.
                # Do not hold the task lock during camera/HTTP/model waits.
                if name in ("move_location", "move_marker", "cruise", "joy_control",
                            "insert_marker_here", "insert_marker_by_pose"):
                    with self._lock:
                        return guarded_call(value, *args, **kwargs)
                return guarded_call(value, *args, **kwargs)
        return call


class SharedMappingCamera:
    """Non-owning RGB-D view for auto_explore; only the web service closes USB."""
    def __init__(self, driver, stopped):
        self.driver, self.stopped = driver, stopped

    def read(self, **kwargs):
        with cancellation_scope(self.stopped):
            return guarded_call(self.driver.grab_rgbd_frame)

    def close(self):
        pass
