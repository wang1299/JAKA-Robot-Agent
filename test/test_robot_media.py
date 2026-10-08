"""Media ownership, stable URLs and task writers; no robot or model connection."""
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_media import MediaArchive, KINDS
from robot_memory import ConversationStore, media_urls
from robot_runtime import TaskCancelled, cancellation_scope
import robot_web
from test_robot_skills import web_state
from test_robot_agent import model_fixture, action

A, B = 'session-A', 'session-B'
T, U = 'user-turn-1', 'user-turn-2'
PHOTO = '/captures/' + 'a' * 32 + '.jpg'
UPLOAD = '/references/' + 'b' * 32 + '.png'
VIDEO = '/videos/' + 'c' * 32 + '.mp4'
PNG = b'\x89PNG\r\n\x1a\nfixture-only'


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = ConversationStore(self.root / 'conversations.sqlite3')
        for cid in (A, B): self.store.create(cid, '测试会话')
        self.captures, self.videos = self.root / 'old-photos', self.root / 'old-videos'
        self.captures.mkdir(); self.videos.mkdir()
        self.archive = MediaArchive(self.store, self.root / 'media', self.captures, self.videos)
        self.web = web_state()
        self.web.conversation_store = self.store
        self.web.media_archive = self.archive
        self.web.agent_sessions = {}
        # Restore real storage methods hidden by the hardware-free planning fixture.
        del self.web.reference_path
        self.manager = self.web.tasks

    def task(self, **values):
        self.manager.task = {'id':'task-one-1','conversation_id':A,'turn_id':T,
            'status':'running','instruction':'寻找杯子','steps':[], **values}
        return self.manager.task

    def put(self, url=PHOTO, kind='observations', cid=A, tid=T):
        path = self.archive.allocate(url,cid,tid,kind)
        path.write_bytes(b'media fixture')
        self.archive.record(url)
        return path

    def test_four_categories_and_reopen_keep_urls(self):
        for url, kind in ((PHOTO,'observations'),(UPLOAD,'uploads'),(VIDEO,'videos'),
                          ('/captures/'+'d'*32+'.jpg','snapshots')):
            path=self.put(url,kind)
            self.assertEqual(path.parent,self.root/'media'/A/T/kind)
        reopened=MediaArchive(ConversationStore(self.store.path),self.root/'media',self.captures,self.videos)
        self.assertEqual(reopened.resolve(PHOTO).read_bytes(),b'media fixture')
        self.assertEqual(len(reopened.list(A,T)),4)
        self.assertEqual(reopened.list(B),[])
        self.assertTrue(all((self.root/'media'/A/T/k).is_dir() for k in KINDS))

    def test_question_and_ids_are_human_readable_in_manifest(self):
        self.archive.prepare_turn(A,T)
        self.archive.prepare_turn(A,T,'帮我找这个杯子')
        self.archive.prepare_turn(A,T,'后来的任务摘要不能覆盖原话')
        manifest=json.loads((self.root/'media'/A/T/'turn.json').read_text('utf-8'))
        self.assertEqual(manifest['question'],'帮我找这个杯子')
        self.assertEqual(manifest['conversation_id'],A)

    def test_idempotent_allocate_and_reject_foreign_overwrite(self):
        path=self.put()
        self.assertEqual(self.archive.allocate(PHOTO,A,T,'observations'),path)
        for cid,tid in ((B,U),(A,U)):
            with self.assertRaises(ValueError): self.archive.allocate(PHOTO,cid,tid,'observations')
        with self.assertRaises(ValueError): self.archive.prepare_turn(B,T)
        self.assertEqual(path.read_bytes(),b'media fixture')

    def test_path_and_type_validation(self):
        for cid,tid in (('../outside',T),(A,'../../outside')):
            with self.assertRaises(ValueError): self.archive.prepare_turn(cid,tid)
        with self.assertRaises(ValueError): self.archive.allocate(VIDEO,A,T,'uploads')
        with self.assertRaises(ValueError): self.archive.resolve('/captures/../../secret.jpg')
        with self.assertRaises(ValueError): self.archive._path('../outside')

    def test_upload_uses_opaque_filename_and_same_turn(self):
        result=self.web.save_reference_upload('../danger.png',PNG,{'conversation_id':A,'turn_id':T})
        path=self.archive.resolve(result['image_url'])
        self.assertEqual(path.parent,self.root/'media'/A/T/'uploads')
        self.assertNotIn('danger',path.name)
        self.assertEqual(path.read_bytes(),PNG)
        self.assertTrue(self.store.owns_media(A,result['image_url']))
        self.assertFalse(self.store.owns_media(B,result['image_url']))

    def test_legacy_fallback_and_reference_reuse_are_copy_only(self):
        original=self.captures/('b'*32+'.png')
        original.write_bytes(PNG)
        self.store.event(A,'reference_upload',{'url':UPLOAD})
        self.assertEqual(self.archive.resolve(UPLOAD),original)
        first=self.archive.include_existing(UPLOAD,A,T,'uploads')
        second=self.archive.include_existing(UPLOAD,A,U,'uploads')
        self.assertNotEqual(first,second)
        self.assertEqual(first.read_bytes(),original.read_bytes())
        self.assertEqual(second.read_bytes(),original.read_bytes())
        with self.assertRaises(ValueError): self.archive.include_existing(UPLOAD,B,'user-turn-B','uploads')
        self.assertEqual(original.read_bytes(),PNG)

    def test_legacy_video_fallback_and_pin(self):
        path=self.videos/('c'*32+'.mp4'); path.write_bytes(b'old video')
        self.store.event(A,'task_video',{'url':VIDEO})
        self.assertEqual(self.archive.resolve(VIDEO),path)
        self.assertEqual(list(media_urls({'video':{'url':VIDEO}})),[VIDEO])
        copied=self.archive.include_existing(VIDEO,A,T,'videos')
        self.assertEqual(copied.read_bytes(),b'old video')
        self.assertTrue(path.exists())

    def test_copy_collision_never_overwrites(self):
        self.put(UPLOAD,'uploads')
        self.archive.prepare_turn(A,U)
        collision=self.root/'media'/A/U/'uploads'/('b'*32+'.png')
        collision.write_bytes(b'keep me')
        with self.assertRaises(ValueError): self.archive.include_existing(UPLOAD,A,U,'uploads')
        self.assertEqual(collision.read_bytes(),b'keep me')

    def test_task_reference_copied_and_status_manifest_updated(self):
        self.put(UPLOAD,'uploads',tid=U)
        self.task(reference={'image_url':UPLOAD})
        self.manager._persist_task()
        self.manager._update(status='canceled')
        manifest=json.loads((self.root/'media'/A/T/'task_task-one-1.json').read_text('utf-8'))
        self.assertEqual(manifest['status'],'canceled')
        self.assertEqual(self.archive.list(A,T)[0]['kind'],'uploads')
        self.assertEqual(self.archive.list(B),[])
        with self.assertRaises(ValueError): self.manager.bind_conversation('task-one-1',A,U)
        with self.assertRaises(ValueError): self.manager.bind_conversation('task-one-1',B,T)

    def test_chat_and_upload_share_user_message_folder(self):
        reference=self.web.save_reference_upload('cup.png',PNG,{'conversation_id':A,'turn_id':T})
        self.web.agent_complete=model_fixture(['收到。'])
        self.web.agent_chat({'question':'帮我看看','conversation_id':A,'user_message_id':T,
                             'assistant_message_id':'assistant-one','reference':reference})
        turns=[p.name for p in (self.root/'media'/A).iterdir() if p.is_dir()]
        self.assertEqual(turns,[T])
        self.web.capture.assert_not_called()
        self.manager.execute.assert_not_called()

    def test_agent_plan_keeps_origin_turn(self):
        self.web.agent_complete=model_fixture([action('plan_inspect_location',instruction='去桌旁看看',
            target_ann_id=7,question='桌旁有什么')])
        result=self.web.agent_chat({'question':'去桌旁看看','conversation_id':A,
            'user_message_id':T,'assistant_message_id':'assistant-one'})
        self.assertEqual(result['task']['turn_id'],T)
        # A later conversation display cannot change the writer's owner.
        self.web.agent_sessions[B]={}
        path=self.manager._capture_path('d'*32,'snapshots',stage='行进中抓拍')
        self.assertEqual(path.parent,self.root/'media'/A/T/'snapshots')
        self.assertEqual(self.archive.list(B),[])
        self.manager.execute.assert_not_called()

    def test_nonmatch_transit_frames_archived_even_not_published(self):
        self.task()
        driver=Mock()
        driver.capture.side_effect=lambda path: Path(path).write_bytes(b'frame')
        driver.get_pose.return_value=(1,2,0)
        self.web.compare_reference=Mock(return_value={'found':False,'confidence':'low'})
        self.manager._find_object_match_value=Mock(return_value={'verdict':'no_match'})
        result=self.manager._capture_reference_snapshot(driver,{},7,{},'行进中抓拍',publish_mode='none')
        self.assertFalse(result['matched'])
        row=self.archive.list(A,T)[0]
        self.assertEqual(row['kind'],'snapshots')
        self.assertTrue(row['exists'])
        self.assertFalse(row['matched'])
        self.assertEqual(row['task_id'],'task-one-1')
        self.assertEqual(self.manager.task.get('search_attempts',[]),[])

    def test_welcome_nonmatch_and_failed_analysis_retain_frame(self):
        self.task(pickup_ann_id=7)
        driver=Mock()
        driver.capture.side_effect=lambda path: Path(path).write_bytes(b'frame')
        self.web.compare_person_reference=Mock(return_value={'found':False,'confidence':'low'})
        objects={7:{'floor_xy':[1,2],'category_zh':'接客点'}}
        matched,_,_=self.manager._capture_welcome_snapshot(driver,objects)
        self.assertFalse(matched)
        self.web.compare_person_reference.side_effect=RuntimeError('fixture vision failure')
        with self.assertRaisesRegex(RuntimeError,'人物识别失败'):
            self.manager._capture_welcome_snapshot(driver,objects)
        self.assertEqual(len(self.archive.list(A,T)),2)
        self.assertTrue(all(r['exists'] for r in self.archive.list(A,T)))
        self.assertEqual(self.manager.task.get('observations',[]),[])

    def test_cancel_before_capture_allocates_nothing(self):
        self.task(cancel_requested=True)
        with cancellation_scope(lambda: True):
            with self.assertRaises(TaskCancelled): self.manager._capture_path('d'*32,'snapshots')
        self.assertEqual(self.archive.list(A),[])

    def test_media_pagination_is_bounded_and_stable(self):
        self.put(); self.put(VIDEO,'videos')
        first=self.archive.list(A,limit=1)
        second=self.archive.list(A,limit=1,offset=1)
        self.assertEqual(len(first),1)
        self.assertEqual(len(second),1)
        self.assertNotEqual(first[0]['url'],second[0]['url'])
        for options in ({'limit':0},{'limit':501},{'offset':-1}):
            with self.assertRaises(ValueError): self.archive.list(A,**options)

    def test_historical_analysis_reads_archived_photo_without_new_capture(self):
        path=self.put()
        self.store.save_memory(A, {'history':[], 'last_scene':{'capture_id':'a'*32},
                                  'evidence':[]})
        self.web.agent_vision=Mock(return_value='箱盖关闭，看不到内部。')
        self.web.agent_complete=model_fixture([
            action('inspect_previous_scene',question='里面有什么'),'箱盖关闭，看不到内部。'])
        result=self.web.agent_chat({'question':'里面有什么','conversation_id':A,
            'user_message_id':U,'assistant_message_id':'assistant-two'})
        self.assertNotIn('task',result)
        self.assertEqual(self.web.agent_vision.call_args.args[0],path)
        self.web.capture.assert_not_called()
        self.assertEqual(self.archive.list(A,U),[])

    def test_video_finishes_in_its_original_turn(self):
        recorder=robot_web.TaskVideoRecorder('task-one-1',Mock(),archive=self.archive,
            media_scope={'conversation_id':A,'turn_id':T})
        self.assertEqual(recorder.path.parent,self.root/'media'/A/T/'videos')
        recorder.path.write_bytes(b'video fixture')
        recorder.request_stop()
        with patch.dict(sys.modules,{'cv2':Mock()}): recorder._run()
        recorder.driver.grab_color_frame.assert_not_called()
        self.assertEqual(self.archive.list(A)[0]['status'],'finished')
        self.assertTrue(self.archive.list(A)[0]['finished_at'])
        self.assertEqual(self.archive.list(B),[])

    def test_video_writer_released_before_final_index(self):
        driver=Mock()
        driver.grab_color_frame.return_value=SimpleNamespace(shape=(8,12,3))
        recorder=robot_web.TaskVideoRecorder('task-one-1',driver,archive=self.archive,
            media_scope={'conversation_id':A,'turn_id':T})
        writer=Mock(); writer.isOpened.return_value=True
        def write(_):
            recorder.path.write_bytes(b'recorded frame fixture')
            recorder.request_stop()
        writer.write.side_effect=write
        cv2=SimpleNamespace(VideoWriter_fourcc=Mock(return_value=0),VideoWriter=Mock(return_value=writer))
        with patch.dict(sys.modules,{'cv2':cv2}): recorder._run()
        writer.release.assert_called_once()
        self.assertEqual(self.archive.list(A)[0]['frame_count'],1)
        self.assertEqual(self.archive.resolve(recorder.url).read_bytes(),b'recorded frame fixture')

    def test_http_upload_download_and_listing(self):
        self.put(); self.put(VIDEO,'videos')
        server=robot_web.RobotWebServer(('127.0.0.1',0),robot_web.RobotWebHandler,self.web)
        worker=threading.Thread(target=server.serve_forever,daemon=True); worker.start()
        try:
            base='http://127.0.0.1:'+str(server.server_address[1])
            req=Request(base+'/api/reference/upload?name=cup.png&conversation_id='+A+'&turn_id='+T,
                        data=PNG,headers={'Content-Type':'image/png'})
            with urlopen(req,timeout=3) as response: result=json.load(response)
            for url,body in ((result['image_url'],PNG),(PHOTO,b'media fixture'),(VIDEO,b'media fixture')):
                with urlopen(base+url,timeout=3) as response: self.assertEqual(response.read(),body)
            with urlopen(base+'/api/conversation/media?id='+A+'&turn_id='+T,timeout=3) as response:
                self.assertEqual(len(json.load(response)['media']),3)
            with urlopen(base+'/api/conversation/media?id='+B,timeout=3) as response:
                self.assertEqual(json.load(response)['media'],[])
        finally:
            server.shutdown(); server.server_close(); worker.join(2)


if __name__ == '__main__': unittest.main()
