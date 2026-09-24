#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continuous manual JAKA RGB-D and pose collector.

The robot is driven with the chassis joy-control API while the camera preview
stays live. A frame is captured only when the operator explicitly requests it.
Saving is delegated to the original collector so its on-disk format remains
identical.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import cv2


DEFAULT_LUMI_ROOT = Path(r"E:\Code\LUMI\LUMI_DEMO-v2")
DEFAULT_LINEAR_SPEED = 0.15
DEFAULT_ANGULAR_SPEED = 0.30
JOY_HEARTBEAT_INTERVAL = 0.25


def load_legacy_collector(lumi_root):
    """Load the original collector, including its exact camera/save logic."""
    root = Path(lumi_root).expanduser().resolve()
    script = root / "jaka_step.py"
    if not script.is_file():
        raise FileNotFoundError(f"Original collector not found: {script}")

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    spec = importlib.util.spec_from_file_location("lumi_legacy_jaka_step", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load original collector: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuous keyboard control with manual RGB-D/pose capture"
    )
    parser.add_argument("--camera-idx", type=int, default=None, help="Orbbec camera index")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--lumi-root", type=str, default=str(DEFAULT_LUMI_ROOT),
        help="Directory containing the original jaka_step.py, agv.py and get_pos.py",
    )
    parser.add_argument(
        "--linear-speed", type=float, default=DEFAULT_LINEAR_SPEED,
        help="Forward/backward speed sent to joy_control",
    )
    parser.add_argument(
        "--angular-speed", type=float, default=DEFAULT_ANGULAR_SPEED,
        help="Left/right angular speed sent to joy_control",
    )
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=800, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--preview-scale", type=float, default=0.5, help="Preview scale")
    return parser.parse_args()


def ask_camera_index(default=0):
    raw = input(f"Camera index (default {default}; usually 0=head camera): ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[WARN] Invalid camera index {raw!r}; using {default}")
        return default


class ContinuousCollector:
    def __init__(self, legacy, args, camera_idx):
        self.legacy = legacy
        output = args.out
        if output is None:
            output = Path(__file__).resolve().parent / "received_frames" / "received_frames"

        self.env = legacy.JakaStepEnv(
            camera_idx=camera_idx,
            data_dir=output,
            camera_width=args.width,
            camera_height=args.height,
            camera_fps=args.fps,
        )
        self.linear_speed = abs(float(args.linear_speed))
        self.angular_speed = abs(float(args.angular_speed))
        self.preview_scale = float(args.preview_scale)
        if self.linear_speed <= 0 or self.angular_speed <= 0:
            raise ValueError("linear-speed and angular-speed must be positive")
        if self.preview_scale <= 0:
            raise ValueError("preview-scale must be positive")

        self.linear = 0.0
        self.angular = 0.0
        self.last_rgb = None
        self.last_depth = None
        self.last_command_at = 0.0

    @property
    def moving(self):
        return self.linear != 0.0 or self.angular != 0.0

    def send_velocity(self, linear, angular, force=False):
        linear = float(linear)
        angular = float(angular)
        now = time.monotonic()
        unchanged = linear == self.linear and angular == self.angular
        if unchanged and not force and now - self.last_command_at < JOY_HEARTBEAT_INTERVAL:
            return True

        endpoint = (
            "/api/joy_control"
            f"?linear_velocity={linear:.6f}&angular_velocity={angular:.6f}"
        )
        try:
            _, data = self.legacy.call_api(
                endpoint, self.env.host, self.env.http_port, self.env.tcp_port
            )
        except Exception as exc:
            print(f"[ERROR] joy_control request failed: {exc}")
            return False

        if not self.legacy.ok(data):
            print(f"[ERROR] joy_control rejected: {data}")
            return False
        self.linear = linear
        self.angular = angular
        self.last_command_at = now
        return True

    def maintain_motion(self):
        """Refresh the command before the chassis' roughly 0.5 s joy timeout."""
        if self.moving:
            self.send_velocity(self.linear, self.angular)

    def stop(self, quiet=False):
        was_moving = self.moving
        success = self.send_velocity(0.0, 0.0, force=True)
        if success and was_moving and not quiet:
            print("[STOP] Robot stopped")
        return success

    def set_direction(self, key):
        commands = {
            ord("w"): (self.linear_speed, 0.0, "forward"),
            ord("s"): (-self.linear_speed, 0.0, "backward"),
            ord("a"): (0.0, self.angular_speed, "turn left"),
            ord("d"): (0.0, -self.angular_speed, "turn right"),
        }
        linear, angular, label = commands[key]
        if self.send_velocity(linear, angular):
            print(f"[DRIVE] {label}: linear={linear:.3f}, angular={angular:.3f}")

    def refresh_frame(self):
        rgb, depth = self.env.camera.get_rgb_depth(timeout_ms=50, max_tries=3)
        if rgb is not None and depth is not None:
            self.last_rgb = rgb
            self.last_depth = depth
        return self.last_rgb, self.last_depth

    def preview(self, window):
        rgb, depth = self.refresh_frame()
        if rgb is None or depth is None:
            return
        motion = (
            f"MOVING L={self.linear:+.2f} A={self.angular:+.2f} | SPACE stop"
            if self.moving else "STOPPED | W/S/A/D drive | P capture"
        )
        status = (
            f"{motion} | cam={self.env.current_cam_idx} "
            f"| next={self.env.step_idx:06d}"
        )
        if rgb.shape[:2] != self.legacy.EXPECTED_HW or depth.shape[:2] != self.legacy.EXPECTED_HW:
            status += f" | BAD SHAPE rgb={rgb.shape[:2]} depth={depth.shape[:2]}"
        image = self.legacy.make_preview(
            rgb, depth, label=status, scale=self.preview_scale
        )
        cv2.imshow(window, image)

    def capture_and_save(self, window):
        if self.moving:
            print("[CAPTURE] Stopping robot before capture")
        if not self.stop(quiet=True):
            print("[ERROR] Capture canceled because the robot could not be stopped")
            return

        # Let chassis vibration settle and discard the first few camera frames.
        settle_until = time.monotonic() + 0.7
        while time.monotonic() < settle_until:
            self.preview(window)
            cv2.waitKey(1)

        rgb = depth = None
        for _ in range(8):
            rgb, depth = self.env.camera.get_rgb_depth(timeout_ms=100, max_tries=3)
            if rgb is not None and depth is not None:
                break
            time.sleep(0.05)
        if rgb is None or depth is None:
            print("[ERROR] Could not capture RGB-D; nothing was saved")
            return

        try:
            pose = self.env.get_pose_safe()
            self.env.save_data("manual", rgb, depth, pose)
        except Exception as exc:
            print(f"[ERROR] Capture/save failed: {exc}")
            return

        self.last_rgb, self.last_depth = rgb, depth
        label = f"SAVED frame={self.env.step_idx - 1:06d}"
        cv2.imshow(
            window,
            self.legacy.make_preview(rgb, depth, label=label, scale=self.preview_scale),
        )
        cv2.waitKey(250)

    def close(self):
        try:
            self.stop(quiet=True)
        finally:
            if self.env.camera is not None:
                self.env.camera.close()


def main():
    args = parse_args()
    legacy = load_legacy_collector(args.lumi_root)
    camera_idx = args.camera_idx if args.camera_idx is not None else ask_camera_index()
    collector = None
    window = "JAKA continuous manual RGB-D pose collector"

    try:
        collector = ContinuousCollector(legacy, args, camera_idx)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("\nControls (click the preview window first):")
        print("  W / S : start moving forward / backward")
        print("  A / D : start turning left / right")
        print("  SPACE or X : stop immediately")
        print("  P or ENTER : stop, wait for stability, then save RGB-D + pose")
        print("  C : switch camera (only when its calibration matches)")
        print("  Q or ESC : stop and quit\n")

        while True:
            collector.preview(window)
            collector.maintain_motion()
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("w"), ord("a"), ord("s"), ord("d")):
                collector.set_direction(key)
            elif key in (ord(" "), ord("x")):
                collector.stop()
            elif key in (ord("p"), 13):
                collector.capture_and_save(window)
            elif key == ord("c"):
                collector.stop(quiet=True)
                collector.env.switch_camera()

    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted")
    finally:
        if collector is not None:
            collector.close()
        cv2.destroyAllWindows()
        print("[DONE] Robot stopped; program ended")


if __name__ == "__main__":
    main()
