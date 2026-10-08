"""Persistent per-conversation records. No intent classification or robot actions."""
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid

ACTIVE = {'planned', 'running', 'canceling', 'needs_clarification'}
TERMINAL = {'succeeded', 'failed', 'error', 'aborted', 'canceled', 'not_found', 'inconclusive', 'interrupted', 'superseded'}


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{8,128}', value):
        raise ValueError('会话或消息标识无效')
    return value


def media_urls(value):
    if isinstance(value, dict):
        capture_id = value.get('capture_id') or value.get('captureId')
        if isinstance(capture_id, str) and re.fullmatch(r'[0-9a-f]{32}', capture_id):
            yield f'/captures/{capture_id}.jpg'
        for item in value.values():
            yield from media_urls(item)
    elif isinstance(value, list):
        for item in value:
            yield from media_urls(item)
    elif isinstance(value, str) and re.fullmatch(r'/(?:captures|references)/[0-9a-f]{32}\.(?:jpg|jpeg|png|webp)', value):
        yield value
    elif isinstance(value, str) and re.fullmatch(r'/videos/[0-9a-f]{32}\.(?:mp4|avi)', value):
        yield value


def compact_record(value, budget=1800):
    """Bound context, not stored evidence. Explicitly label excerpts as incomplete."""
    raw = json.dumps(value, ensure_ascii=False)
    if len(raw) <= budget:
        return copy.deepcopy(value)
    return {'excerpt': raw[:budget], 'truncated': True,
            'image_urls': [u for u in dict.fromkeys(media_urls(value)) if not u.startswith('/videos/')][:8]}


class ConversationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.db() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, created REAL, updated REAL,
                    memory TEXT NOT NULL DEFAULT '{}', deleted INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, created REAL, body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS message_conversation ON messages(conversation_id, created);
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, updated REAL, body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS task_conversation ON tasks(conversation_id, updated);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL, kind TEXT NOT NULL,
                    created REAL, body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS event_conversation ON events(conversation_id, seq);
                CREATE TABLE IF NOT EXISTS media (
                    url TEXT, conversation_id TEXT, PRIMARY KEY(url, conversation_id));
            ''')

    def claim_runtime(self):
        """Only one robot web process may recover/run this database at a time."""
        handle = open(self.path.with_suffix('.lock'), 'a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                if handle.seek(0, 2) == 0:
                    handle.write(b'0')
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError('已有网页进程正在使用会话数据库，请勿重复启动服务') from exc
        self.runtime_lock = handle  # Held for the process lifetime, released by the OS on exit.

    @contextmanager
    def db(self):
        with self.lock:
            db = sqlite3.connect(str(self.path), timeout=10)
            db.row_factory = sqlite3.Row
            try:
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

    def _exists(self, db, cid):
        valid_id(cid)
        row = db.execute('SELECT * FROM conversations WHERE id=? AND deleted=0', (cid,)).fetchone()
        if row is None:
            raise ValueError('会话不存在或已删除，请重新打开会话')
        return row

    def create(self, cid, title='新对话'):
        valid_id(cid)
        with self.db() as db:
            row = db.execute('SELECT deleted FROM conversations WHERE id=?', (cid,)).fetchone()
            if row and row['deleted']:
                raise ValueError('会话已删除，不能由旧缓存恢复')
            now = time.time()
            db.execute('INSERT OR IGNORE INTO conversations(id,title,created,updated) VALUES(?,?,?,?)',
                       (cid, str(title)[:80] or '新对话', now, now))
        return self.conversation(cid)

    def list(self):
        with self.db() as db:
            return [dict(r) for r in db.execute('SELECT id,title,created*1000 AS createdAt,updated*1000 AS updatedAt FROM conversations WHERE deleted=0 ORDER BY updated DESC')]

    def conversation(self, cid):
        with self.db() as db:
            row = self._exists(db, cid)
            messages = [json.loads(r['body']) for r in db.execute('SELECT body FROM messages WHERE conversation_id=? ORDER BY created,rowid', (cid,))]
            tasks = {r['id']: json.loads(r['body']) for r in db.execute('SELECT id,body FROM tasks WHERE conversation_id=?', (cid,))}
            for message in messages:
                if message.get('taskId') in tasks:
                    message['task'] = tasks[message['taskId']]
            linked = {m.get('taskId') for m in messages}
            # A disconnect/crash can occur between plan creation and the final reply.
            for tid, task in tasks.items():
                if tid not in linked:
                    messages.append({'id': 'task-' + tid, 'role': 'assistant', 'state': 'done',
                                     'text': '本会话的任务记录（请以任务状态为准）。',
                                     'taskId': tid, 'task': task})
            return {'id': cid, 'title': row['title'], 'createdAt': row['created']*1000,
                    'updatedAt': row['updated']*1000, 'messages': messages}

    def validate_turn_ids(self, cid, user_id, assistant_id):
        valid_id(user_id)
        valid_id(assistant_id)
        if user_id == assistant_id:
            raise ValueError('用户消息和回复必须使用不同标识')
        with self.db() as db:
            self._exists(db, cid)
            if db.execute('SELECT 1 FROM messages WHERE id IN (?,?)', (user_id, assistant_id)).fetchone():
                raise ValueError('该消息已接收，请刷新会话查看结果；不会重复调用工具')

    def latest_task(self, cid):
        with self.db() as db:
            row = db.execute('SELECT tasks.body FROM tasks JOIN conversations ON conversations.id=tasks.conversation_id '
                             'WHERE tasks.conversation_id=? AND conversations.deleted=0 ORDER BY tasks.updated DESC,tasks.rowid DESC LIMIT 1', (cid,)).fetchone()
            return json.loads(row['body']) if row else None

    def start_turn(self, cid, user, assistant_id):
        self.validate_turn_ids(cid, user['id'], assistant_id)
        with self.db() as db:
            self._exists(db, cid)
            now = time.time()
            assistant = {'id': assistant_id, 'role': 'assistant', 'text': '本轮正在处理。',
                         'state': 'processing', 'createdAt': now*1000 + 1}
            for message in (user, assistant):
                db.execute('INSERT INTO messages VALUES(?,?,?,?)',
                           (message['id'], cid, message['createdAt']/1000, json.dumps(message, ensure_ascii=False)))
                self._pin(db, cid, message)
            db.execute("UPDATE conversations SET updated=?, title=CASE WHEN title='新对话' THEN ? ELSE title END WHERE id=?",
                       (now, user['text'][:24], cid))

    def _pin(self, db, cid, value):
        for url in media_urls(value):
            db.execute('INSERT OR IGNORE INTO media VALUES(?,?)', (url, cid))

    def message(self, cid, message):
        mid = valid_id(message['id'])
        with self.db() as db:
            self._exists(db, cid)
            existing = db.execute('SELECT conversation_id FROM messages WHERE id=?', (mid,)).fetchone()
            if existing and existing['conversation_id'] != cid:
                raise ValueError('消息不属于当前会话')
            db.execute('INSERT INTO messages VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body',
                       (mid, cid, message.get('createdAt', time.time()*1000)/1000, json.dumps(message, ensure_ascii=False)))
            self._pin(db, cid, message)
            db.execute('UPDATE conversations SET updated=? WHERE id=?', (time.time(), cid))
            if message.get('role') == 'user':
                db.execute("UPDATE conversations SET title=? WHERE id=? AND title='新对话'", (message.get('text', '')[:24] or '新对话', cid))

    def save_memory(self, cid, memory):
        with self.db() as db:
            self._exists(db, cid)
            db.execute('UPDATE conversations SET memory=? WHERE id=?', (json.dumps(memory, ensure_ascii=False), cid))
            self._pin(db, cid, memory)

    def event(self, cid, kind, body):
        with self.db() as db:
            self._exists(db, cid)
            db.execute('INSERT INTO events(conversation_id,kind,created,body) VALUES(?,?,?,?)',
                       (cid, kind, time.time(), json.dumps(body, ensure_ascii=False)))
            self._pin(db, cid, body)

    def task(self, task):
        cid = task.get('conversation_id')
        if not cid:
            return
        with self.db() as db:
            self._exists(db, cid)
            old = db.execute('SELECT body,conversation_id FROM tasks WHERE id=?', (task['id'],)).fetchone()
            if old and old['conversation_id'] != cid:
                raise ValueError('任务不能转移到其他会话')
            body = json.dumps(task, ensure_ascii=False)
            if old and old['body'] == body:
                return
            db.execute('INSERT INTO tasks VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET updated=excluded.updated,body=excluded.body',
                       (task['id'], cid, time.time(), body))
            self._pin(db, cid, task)
            prior = json.loads(old['body']) if old else {}
            if prior.get('status') != task.get('status') or prior.get('current_step') != task.get('current_step'):
                db.execute('INSERT INTO events(conversation_id,kind,created,body) VALUES(?,?,?,?)',
                           (cid, 'task_state', time.time(), body))
            db.execute('UPDATE conversations SET updated=? WHERE id=?', (time.time(), cid))

    def load_memory(self, cid):
        with self.db() as db:
            row = self._exists(db, cid)
            memory = json.loads(row['memory'])
            tasks = [json.loads(r['body']) for r in db.execute('SELECT body FROM tasks WHERE conversation_id=? ORDER BY updated DESC LIMIT 8', (cid,))]
            rows = [json.loads(r['body']) for r in db.execute('SELECT body FROM messages WHERE conversation_id=? ORDER BY created DESC,rowid DESC LIMIT 32', (cid,))][::-1]
        memory.setdefault('evidence', [])
        # Imported browser records are historical statements, not execution evidence.
        memory['history'] = [{'role': m['role'], 'text': m.get('text', '')[:2000], 'capture_id': m.get('captureId')}
                             for m in rows[-16:] if m.get('text') and m.get('role') in ('user', 'assistant')]
        remaining = 6000
        recent = []
        for message in reversed(memory['history']):
            if remaining <= 0:
                break
            recent.append({**message, 'text': message['text'][:remaining]})
            remaining -= len(recent[-1]['text'])
        memory['history'] = recent[::-1]
        memory['history_summary'] = [{'role': m['role'], 'text': m.get('text', '')[:120],
                                      'recorded_at': m.get('createdAt'), 'source': 'historical_dialogue_not_live'}
                                     for m in (rows[:-len(recent)] if recent else rows)[-8:] if m.get('text')]
        memory['task_experiences'] = []
        latest_scene = memory.get('last_scene') or {}
        for task in tasks:
            observations = task.get('observations', [])[-4:]
            experience = {k: copy.deepcopy(task[k]) for k in
                ('id','instruction','status','started_at','finished_at','map_name') if k in task}
            experience['instruction'] = str(experience.get('instruction') or '')[:200]
            experience['result_text'] = str(task.get('result_text') or '')[:400]
            experience['error'] = str(task.get('error') or '')[:200]
            experience['steps'] = compact_record(task.get('steps', []), 500)
            experience['observations'] = compact_record(observations, 700)
            memory['task_experiences'].append(experience)
            for observation in observations:
                url = observation.get('image_url', '')
                match = re.fullmatch(r'/captures/([0-9a-f]{32})\.jpg', url)
                observed_at = observation.get('observed_at') or task.get('finished_at') or task.get('started_at') or 0
                if match and observed_at > latest_scene.get('observed_at', 0):
                    latest_scene = {'capture_id': match[1], 'observed_at': observed_at,
                                    'task_id': task['id'], 'target_ann_id': observation.get('ann_id'),
                                    'source': 'task_observation_not_live'}
            pending = memory.get('pending_request') or {}
            planned = memory.get('planned_task') or {}
            if planned.get('id') == task['id']:
                memory['planned_task'] = {**planned, 'status': task['status'], 'recorded_at': task.get('finished_at') or time.time()}
            if pending.get('task_id') == task['id']:
                memory['pending_request'] = None if task['status'] in TERMINAL else {**pending, 'status': task['status']}
        # The newest three tasks have details; older tasks retain a small index.
        memory['task_experiences'][3:] = [{k: t[k] for k in ('id','instruction','status','finished_at') if k in t}
                                         for t in memory['task_experiences'][3:]]
        if latest_scene:
            memory['last_scene'] = latest_scene
        return memory

    def recall(self, cid, query='', offset=0, limit=6):
        """Scoped retrieval chosen by the model, never a global search."""
        with self.db() as db:
            self._exists(db, cid)
            rows = db.execute('SELECT seq,kind,created,body FROM events WHERE conversation_id=? AND instr(body,?)>0 ORDER BY seq DESC LIMIT ? OFFSET ?',
                              (cid, query, limit+1, offset)).fetchall()
            return {'source_type': 'conversation_history_not_live', 'items': [
                {'seq': r['seq'], 'kind': r['kind'], 'recorded_at': r['created'], 'record': compact_record(json.loads(r['body']))}
                for r in rows[:limit]], 'next_offset': offset+limit if len(rows)>limit else None}

    def owns_media(self, cid, url):
        with self.db() as db:
            return db.execute('SELECT 1 FROM media WHERE conversation_id=? AND url=?', (cid, url)).fetchone() is not None

    def pinned_media(self):
        with self.db() as db:
            return {r['url'] for r in db.execute('SELECT url FROM media JOIN conversations ON conversations.id=media.conversation_id WHERE deleted=0')}

    def delete(self, cid):
        with self.db() as db:
            self._exists(db, cid)
            tasks = [json.loads(r['body']) for r in db.execute('SELECT body FROM tasks WHERE conversation_id=?', (cid,))]
            if any(t.get('status') in ACTIVE for t in tasks):
                raise ValueError('请先停止或取消本会话的待执行/执行中任务，再删除会话')
            for table in ('messages', 'tasks', 'events', 'media'):
                db.execute(f'DELETE FROM {table} WHERE conversation_id=?', (cid,))
            db.execute("UPDATE conversations SET deleted=1,memory='{}' WHERE id=?", (cid,))

    def import_legacy(self, conversation):
        cid = valid_id(conversation.get('id'))
        if not isinstance(conversation.get('messages', []), list):
            raise ValueError('会话消息必须是列表')
        with self.db() as db:
            if db.execute('SELECT 1 FROM conversations WHERE id=?', (cid,)).fetchone():
                return  # Existing server record or deletion tombstone always wins.
            now = time.time()
            db.execute('INSERT INTO conversations(id,title,created,updated) VALUES(?,?,?,?)',
                       (cid, str(conversation.get('title') or '新对话')[:80], now, now))
            for old in conversation.get('messages', []):
                if not isinstance(old, dict) or old.get('role') not in ('user','assistant'):
                    continue
                # Preserve display history, but never import client task claims into tasks/memory.
                message = {k: copy.deepcopy(old[k]) for k in ('role','text','createdAt','task','image','captureId',
                    'imageSource','toolTrace','referenceStoredImage','referenceImage') if k in old}
                message.update(id=uuid.uuid4().hex, state='done', legacy=True)
                message['text'] = str(message.get('text') or '')
                if not isinstance(message.get('createdAt'), (int, float)) or not math.isfinite(message['createdAt']):
                    message['createdAt'] = time.time()*1000
                if isinstance(message.get('task'), dict):
                    message['task']['status'] = 'interrupted'
                    message['task']['error'] = '导入的历史展示记录，不作为执行凭据；需要动作请重新规划。'
                for key in ('image','referenceStoredImage','referenceImage'):
                    if key in message and not list(media_urls(message[key])):
                        message.pop(key)
                db.execute('INSERT INTO messages VALUES(?,?,?,?)',
                           (message['id'], cid, message['createdAt']/1000, json.dumps(message, ensure_ascii=False)))
                self._pin(db, cid, message)
                db.execute('INSERT INTO events(conversation_id,kind,created,body) VALUES(?,?,?,?)',
                           (cid, 'legacy_dialogue_unverified', now, json.dumps(message, ensure_ascii=False)))

    def recover(self):
        """No automatic resumption or claim of physical completion after restart."""
        with self.db() as db:
            tasks = [json.loads(r['body']) for r in db.execute('SELECT body FROM tasks')]
            for row in db.execute('SELECT id,body FROM messages').fetchall():
                message = json.loads(row['body'])
                if message.get('state') == 'processing':
                    message.update(state='done', incomplete=True, text='网页服务曾重启，本轮处理未完成；请核实任务状态后继续。')
                    db.execute('UPDATE messages SET body=? WHERE id=?', (json.dumps(message, ensure_ascii=False), row['id']))
        for task in tasks:
            if task.get('status') in ACTIVE:
                task.update(status='interrupted', error='网页服务曾重启，任务未自动恢复；请核实现场状态并重新规划。', finished_at=time.time())
                self.task(task)
