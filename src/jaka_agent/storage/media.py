"""Conversation/turn media archive. Stable web URLs, explicit writer ownership.

This module never captures frames, moves the robot, deletes media, or changes
capture intervals. SQLite is the URL index; JSON sidecars make folders readable.
"""
import json
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

from jaka_agent.storage.memory import valid_id

URL_RE = re.compile(r'/(captures|references|videos)/([0-9a-f]{32})\.(jpg|jpeg|png|webp|mp4|avi)')
KINDS = {'uploads', 'snapshots', 'observations', 'videos'}


class MediaArchive:
    def __init__(self, store, root, capture_dir, video_dir):
        self.store = store
        self.root = Path(root).resolve()
        self.capture_dir = Path(capture_dir)
        self.video_dir = Path(video_dir)
        self.lock = threading.RLock()
        with store.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS media_turns (
                    turn_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                    created REAL NOT NULL, question TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS media_assets (
                    url TEXT, conversation_id TEXT, turn_id TEXT, kind TEXT,
                    relative_path TEXT, created REAL, metadata TEXT NOT NULL,
                    PRIMARY KEY(url,conversation_id,turn_id));
                CREATE INDEX IF NOT EXISTS media_asset_url ON media_assets(url);
                CREATE INDEX IF NOT EXISTS media_asset_turn ON media_assets(conversation_id,turn_id);
            ''')

    def _path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError('媒体路径超出归档目录')
        return path

    def _json(self, path, value):
        # Unique temporary names and replace prevent partial manifests after a crash.
        path = self._path(path.relative_to(self.root))
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
            temp.replace(path)
        finally:
            if temp.exists(): temp.unlink()

    def prepare_turn(self, cid, turn_id, question=''):
        valid_id(cid); valid_id(turn_id)
        with self.lock:
            with self.store.db() as db:
                conversation = self.store._exists(db, cid)
                row = db.execute('SELECT * FROM media_turns WHERE turn_id=?', (turn_id,)).fetchone()
                if row and row['conversation_id'] != cid:
                    raise ValueError('本轮对话不属于当前会话')
                created = row['created'] if row else time.time()
                text = str((row['question'] if row else '') or question or '')[:4000]
                db.execute('INSERT INTO media_turns VALUES(?,?,?,?) ON CONFLICT(turn_id) DO UPDATE SET question=excluded.question',
                           (turn_id,cid,created,text))
                session = {'conversation_id':cid,'title':conversation['title'],'created_at':conversation['created']}
            folder = self._path(Path(cid) / turn_id)
            for kind in KINDS: self._path(Path(cid)/turn_id/kind).mkdir(parents=True,exist_ok=True)
            self._json(self._path(Path(cid)/'session.json'),session)
            self._json(folder/'turn.json',{'conversation_id':cid,'turn_id':turn_id,'question':text,'created_at':created,
                'folders':{'uploads':'用户上传/本任务复用的参考图','snapshots':'定时或连续检测抓拍',
                           'observations':'原地/到点/巡逻观察照片','videos':'任务录像'}})
            return folder

    @staticmethod
    def _check_url(url, kind=None):
        match = URL_RE.fullmatch(url or '')
        if not match: raise ValueError('媒体地址无效')
        prefix, ident, ext = match.groups()
        if (prefix == 'videos') != (ext in ('mp4','avi')):
            raise ValueError('媒体地址与类型不一致')
        if kind is not None:
            if kind not in KINDS or (prefix == 'videos') != (kind == 'videos') or (prefix == 'references') != (kind == 'uploads'):
                raise ValueError('媒体分类无效')
        return prefix, ident, ext

    def allocate(self, url, cid, turn_id, kind, task_id=None, metadata=None):
        prefix, ident, ext = self._check_url(url, kind)
        with self.lock:
            self.prepare_turn(cid, turn_id)
            with self.store.db() as db:
                old = db.execute('SELECT * FROM media_assets WHERE url=?', (url,)).fetchall()
                if old:
                    own = next((r for r in old if r['conversation_id']==cid and r['turn_id']==turn_id and r['kind']==kind),None)
                    if own: return self._path(own['relative_path'])
                    raise ValueError('媒体标识已属于其他对话，不能覆盖')
                now = time.time()
                filename = time.strftime('%Y%m%d_%H%M%S',time.localtime(now)) + '_' + ident + '.' + ext
                relative = (Path(cid)/turn_id/kind/filename).as_posix()
                record = {**(metadata or {}),'url':url,'conversation_id':cid,'turn_id':turn_id,
                          'task_id':task_id,'kind':kind,'created_at':now,'status':'allocated'}
                db.execute('INSERT INTO media_assets VALUES(?,?,?,?,?,?,?)',
                           (url,cid,turn_id,kind,relative,now,json.dumps(record,ensure_ascii=False)))
                self.store._pin(db,cid,{'url':url})
            path = self._path(relative)
            self._json(path.with_suffix(path.suffix+'.json'),record)
            return path

    def resolve(self, url):
        prefix, ident, ext = self._check_url(url)
        with self.store.db() as db:
            rows = db.execute('SELECT relative_path FROM media_assets WHERE url=? ORDER BY created', (url,)).fetchall()
        for row in rows:
            path = self._path(row['relative_path'])
            if path.is_file(): return path
        # Old URLs remain readable. No recursive scan and no move of live files.
        return (self.video_dir if prefix=='videos' else self.capture_dir) / (ident+'.'+ext)

    def record(self, url, **details):
        self._check_url(url)
        with self.lock:
            with self.store.db() as db:
                rows = db.execute('SELECT * FROM media_assets WHERE url=? ORDER BY created', (url,)).fetchall()
                for row in rows:
                    path = self._path(row['relative_path'])
                    record = {**json.loads(row['metadata']),**details,'updated_at':time.time()}
                    record.update(exists=path.is_file(),size_bytes=path.stat().st_size if path.is_file() else 0)
                    if 'status' not in details: record['status']='saved' if record['exists'] else 'missing'
                    db.execute('UPDATE media_assets SET metadata=? WHERE url=? AND conversation_id=? AND turn_id=?',
                               (json.dumps(record,ensure_ascii=False),url,row['conversation_id'],row['turn_id']))
                    self._json(path.with_suffix(path.suffix+'.json'),record)

    def include_existing(self, url, cid, turn_id, kind, task_id=None, source='reused_reference'):
        """Copy only known media; retain the original and the original URL."""
        self._check_url(url,kind)
        if not self.store.owns_media(cid,url): raise ValueError('媒体不属于当前会话')
        with self.lock:
            self.prepare_turn(cid,turn_id)
            with self.store.db() as db:
                old=db.execute('SELECT relative_path FROM media_assets WHERE url=? AND conversation_id=? AND turn_id=?',
                               (url,cid,turn_id)).fetchone()
            if old and self._path(old['relative_path']).is_file(): return self._path(old['relative_path'])
            original=self.resolve(url)
            if not original.is_file(): return None
            _, ident, ext=self._check_url(url)
            relative=(Path(cid)/turn_id/kind/(ident+'.'+ext)).as_posix()
            dest=self._path(relative)
            if dest.exists():
                # Never overwrite an unrelated file, even after an interrupted copy.
                import hashlib
                def digest(p):
                    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').digest()
                if digest(dest)!=digest(original): raise ValueError('归档文件已存在且内容不同')
            else:
                temp=dest.with_name(dest.name+'.'+uuid.uuid4().hex+'.tmp')
                try:
                    shutil.copy2(original,temp)
                    temp.replace(dest)
                finally:
                    if temp.exists(): temp.unlink()
            now=time.time()
            record={'url':url,'conversation_id':cid,'turn_id':turn_id,'task_id':task_id,'kind':kind,
                    'source':source,'created_at':original.stat().st_mtime,'status':'saved',
                    'exists':True,'size_bytes':dest.stat().st_size}
            with self.store.db() as db:
                db.execute('INSERT OR REPLACE INTO media_assets VALUES(?,?,?,?,?,?,?)',
                           (url,cid,turn_id,kind,relative,now,json.dumps(record,ensure_ascii=False)))
            self._json(dest.with_suffix(dest.suffix+'.json'),record)
            return dest

    def task_record(self, task):
        cid, tid=task.get('conversation_id'), task.get('turn_id')
        if not cid or not tid: return
        with self.lock:
            folder=self.prepare_turn(cid,tid,task.get('instruction',''))
            reference=(task.get('reference') or {}).get('image_url')
            if reference: self.include_existing(reference,cid,tid,'uploads',task['id'])
            # File-explorer index; full execution records remain in the conversation DB.
            self._json(folder/('task_'+valid_id(task['id'])+'.json'),
                       {k:task.get(k) for k in ('id','conversation_id','turn_id','instruction','status','skill',
                                              'created_at','started_at','finished_at','error','video')})

    def list(self, cid, turn_id=None, limit=200, offset=0):
        if not 1 <= limit <= 500 or offset < 0:
            raise ValueError('媒体分页参数无效')
        with self.store.db() as db:
            self.store._exists(db,cid)
            rows=db.execute('SELECT * FROM media_assets WHERE conversation_id=? AND (? IS NULL OR turn_id=?) ORDER BY created,rowid LIMIT ? OFFSET ?',
                            (cid,turn_id,turn_id,limit,offset)).fetchall()
        return [{**json.loads(r['metadata']),'relative_path':r['relative_path'],
                 'exists':self._path(r['relative_path']).is_file()} for r in rows]
