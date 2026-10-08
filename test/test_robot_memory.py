"""Persistent conversation contracts. All robot/model operations use fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_memory import ConversationStore
import robot_web
from test_robot_skills import web_state
from test_robot_agent import action, model_fixture

A, B = 'session-A', 'session-B'
PHOTO = 'a' * 32


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'memory.sqlite3'
        self.store = ConversationStore(self.path)
        self.store.create(A)
        self.store.create(B)
        self.web = web_state()
        self.web.conversation_store = self.store
        self.web.agent_sessions = {}
        self.web.capture_path = Mock(return_value=Path(__file__))
        self.web.agent_vision = Mock(return_value='箱盖关闭，看不到内部。')

    def chat(self, outputs, question='你好', cid=A, **kwargs):
        self.web.agent_complete = model_fixture(outputs)
        return self.web.agent_chat({'question': question, 'conversation_id': cid, **kwargs})

    def plan(self):
        return self.chat([action('plan_inspect_location', instruction='去桌旁看看', target_ann_id=7,
                                 question='桌旁是什么')], question='去桌旁看看')['task']

    def complete_task(self):
        task = self.plan()
        self.web.tasks._update(status='succeeded', finished_at=time.time(), observations=[{
            'image_url': '/captures/' + PHOTO + '.jpg', 'observed_at': time.time(),
            'ann_id': 7, 'text': '箱盖关闭'}])
        return task

    def tearDown(self):
        self.web.capture.assert_not_called()
        self.web.tasks.execute.assert_not_called()
        self.web.speak.assert_not_called()

    def test_complete_dialogue_persists_after_reopen(self):
        self.chat(['我记住了。'], question='今天讨论航空箱')
        reopened = ConversationStore(self.path)
        self.assertEqual(len(reopened.conversation(A)['messages']), 2)
        self.assertIn('航空箱', json.dumps(reopened.load_memory(A), ensure_ascii=False))
        self.assertNotIn('航空箱', json.dumps(reopened.load_memory(B), ensure_ascii=False))

    def test_server_history_wins_over_client_history(self):
        self.chat(['你好'], cid=B, history=[{'role':'user','text':'来自其他会话的秘密'}])
        sent = json.dumps(self.web.agent_complete.call_args.args[0], ensure_ascii=False)
        self.assertNotIn('其他会话的秘密', sent)

    def test_async_completion_reconciles_plan_and_photo_in_origin(self):
        task = self.complete_task()
        memory = self.store.load_memory(A)
        self.assertIsNone(memory['pending_request'])
        self.assertEqual(memory['planned_task']['status'], 'succeeded')
        self.assertEqual(memory['last_scene']['capture_id'], PHOTO)
        self.assertEqual(memory['last_scene']['target_ann_id'], 7)
        self.assertEqual(self.store.conversation(A)['messages'][-1]['task']['status'], 'succeeded')
        self.assertEqual(self.store.load_memory(B)['task_experiences'], [])
        self.assertIsNone(self.web.conversation_task(B))
        self.assertEqual(self.web.conversation_task(A)['id'], task['id'])

    def test_followup_can_reuse_task_photo_after_restart_without_navigation(self):
        self.complete_task()
        self.web.agent_sessions.clear()
        self.web.conversation_store = ConversationStore(self.path)
        result = self.chat([action('inspect_previous_scene', question='里面有什么'), '箱盖关闭，无法判断内部。'], question='它里面有什么')
        self.assertNotIn('task', result)
        self.web.agent_vision.assert_called_once()
        prompt = self.web.agent_complete.call_args_list[0].args[0][0]['content']
        self.assertIn('succeeded', prompt)
        self.assertIn('task_observation_not_live', prompt)

    def test_historical_photo_scope_is_enforced(self):
        self.complete_task()
        result = self.chat([action('inspect_memory_image', capture_id=PHOTO, question='看看'), '本会话没有这张照片。'], cid=B)
        self.assertFalse(result['tool_trace'][0]['ok'])
        self.web.agent_vision.assert_not_called()

    def test_missing_photo_does_not_take_new_picture(self):
        self.complete_task()
        self.web.capture_path.return_value = Path(self.tmp.name) / 'missing.jpg'
        result = self.chat([action('inspect_memory_image',capture_id=PHOTO,question='看看'), '照片不可用。'])
        self.assertFalse(result['tool_trace'][0]['ok'])

    def test_shared_busy_but_private_task_contents(self):
        self.plan()
        self.web.tasks._update(status='running', instruction='会话A的私有需求')
        self.chat([action('get_robot_status'), '机器人正在忙。'], cid=B)
        sent = json.dumps(self.web.agent_complete.call_args.args[0], ensure_ascii=False)
        self.assertNotIn('会话A的私有需求', sent)
        self.assertIn('"robot_busy": true', sent.replace('\\"', '"'))

    def test_other_conversation_cannot_replace_pending_plan(self):
        original = self.plan()
        callback = Mock()
        with self.assertRaises(ValueError):
            self.web.plan_for_conversation(B, callback)
        callback.assert_not_called()
        self.assertEqual(self.web.tasks.snapshot()['id'], original['id'])
        with self.assertRaises(ValueError): self.web.require_task_owner(original['id'], B)
        self.web.require_task_owner(original['id'], A)

    def test_task_owner_cannot_change_in_store(self):
        task = self.plan()
        with self.assertRaises(ValueError):
            self.store.task({**task, 'conversation_id': B})

    def test_same_conversation_replan_supersedes_old(self):
        old = self.plan()
        new = self.plan()
        tasks = {t['id']:t for t in self.store.load_memory(A)['task_experiences']}
        self.assertEqual(tasks[old['id']]['status'], 'superseded')
        self.assertEqual(tasks[new['id']]['status'], 'planned')

    def test_restart_marks_pending_tasks_and_inflight_turn_interrupted(self):
        self.plan()
        self.store.start_turn(B, {'id':'pending-user','role':'user','text':'还没回复','createdAt':time.time()*1000}, 'pending-assistant')
        reopened = ConversationStore(self.path)
        reopened.recover()
        self.assertEqual(reopened.load_memory(A)['planned_task']['status'], 'interrupted')
        self.assertTrue(reopened.conversation(B)['messages'][-1]['incomplete'])
        self.assertIsNone(reopened.load_memory(A)['pending_request'])

    def test_duplicate_request_never_calls_model_again(self):
        kwargs = {'user_message_id':'unique-user','assistant_message_id':'unique-assistant'}
        self.chat(['你好'], **kwargs)
        self.web.agent_complete.reset_mock()
        with self.assertRaises(ValueError): self.web.agent_chat({'question':'你好','conversation_id':A, **kwargs})
        self.web.agent_complete.assert_not_called()
        self.assertEqual(len(self.store.conversation(A)['messages']), 2)

    def test_runtime_exclusive_lock_prevents_double_recovery(self):
        self.store.claim_runtime()
        try:
            other=ConversationStore(self.path)
            with self.assertRaises(RuntimeError): other.claim_runtime()
        finally:
            self.store.runtime_lock.close()

    def test_reference_cannot_disguise_another_conversations_image(self):
        self.store.event(A,'reference_upload',{'image_url':'/references/'+'a'*32+'.jpg'})
        with self.assertRaises(ValueError):
            self.chat(['不应调用'],reference={'image_url':'/references/'+'a'*32+'.jpg',
                'reference_id':'b'*32,'suffix':'.jpg'})
        self.web.agent_complete.assert_not_called()

    def test_old_task_status_available_after_runtime_replaced(self):
        task=self.complete_task()
        self.web.tasks.task=None
        self.assertEqual(self.web.conversation_task(A)['id'],task['id'])
        self.assertIsNone(self.web.conversation_task(B))

    def test_failed_request_retains_both_messages(self):
        with self.assertRaises(RuntimeError): self.chat([RuntimeError('offline')])
        messages = self.store.conversation(A)['messages']
        self.assertEqual(len(messages), 2)
        self.assertTrue(messages[-1]['incomplete'])

    def test_long_conversation_bounded_context_full_records_retrievable(self):
        for n in range(90):
            msg = {'id':f'message-{n:03d}', 'role':'user', 'text':f'第{n}轮记忆' + '内容'*1200, 'createdAt':n+1}
            self.store.message(A,msg)
            self.store.event(A,'user_message',msg)
        memory = self.store.load_memory(A)
        self.assertLessEqual(len(memory['history']),16)
        self.assertLessEqual(sum(len(m['text']) for m in memory['history']),6000)
        self.assertEqual(len(self.store.conversation(A)['messages']),90)
        recall = self.store.recall(A,'第0轮记忆')
        self.assertEqual(len(recall['items']),1)
        self.assertTrue(recall['items'][0]['record']['truncated'])
        self.assertEqual(self.store.recall(B,'第0轮记忆')['items'],[])

    def test_legacy_import_idempotent_untrusted_and_delete_tombstone(self):
        legacy={'id':'legacy-123','title':'旧会话','messages':[{'role':'assistant','text':'过去说的话',
            'task':{'id':'fake-task','status':'succeeded'},'referenceImage':'blob:expired'}]}
        self.store.import_legacy(legacy)
        self.store.import_legacy(legacy)
        messages=self.store.conversation(legacy['id'])['messages']
        self.assertEqual(len(messages),1)
        self.assertEqual(messages[0]['task']['status'],'interrupted')
        self.assertNotIn('referenceImage',messages[0])
        self.assertEqual(self.store.load_memory(legacy['id'])['task_experiences'],[])
        self.store.delete(legacy['id'])
        self.store.import_legacy(legacy)
        with self.assertRaises(ValueError): self.store.conversation(legacy['id'])

    def test_delete_pending_is_blocked_completed_allowed(self):
        self.plan()
        with self.assertRaises(ValueError): self.store.delete(A)
        self.web.tasks._update(status='canceled')
        self.store.delete(A)
        self.assertEqual([c['id'] for c in self.store.list()], [B])

    def test_media_pinned_until_conversation_deleted(self):
        self.complete_task()
        url='/captures/'+PHOTO+'.jpg'
        self.assertIn(url,self.store.pinned_media())
        self.store.delete(A)
        self.assertNotIn(url,self.store.pinned_media())

    def test_orphan_plan_survives_interrupted_reply(self):
        task = self.web.plan_for_conversation(A, lambda:self.web.tasks.plan_skill('navigate', '去门口',
            target_ann_ids=[19],return_to_start=False))
        self.assertEqual(self.store.conversation(A)['messages'][-1]['taskId'],task['id'])

    def test_concurrent_task_and_chat_writes_do_not_cross_sessions(self):
        task=self.plan()
        def write():
            for i in range(10): self.store.task({**task,'current_step':i,'status':'running'})
        worker=threading.Thread(target=write)
        worker.start()
        self.chat(['这是B会话。'],cid=B)
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.store.load_memory(B)['task_experiences'],[])


class MemoryHttpTests(unittest.TestCase):
    chat = MemoryTests.chat
    plan = MemoryTests.plan
    tearDown = MemoryTests.tearDown
    def setUp(self):
        MemoryTests.setUp(self)
        self.server=robot_web.RobotWebServer(('127.0.0.1',0),robot_web.RobotWebHandler,self.web)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)

    def req(self,path,payload=None):
        req=Request(f'http://127.0.0.1:{self.server.server_port}'+path,
                    data=json.dumps(payload).encode() if payload is not None else None,
                    headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=5) as response: return json.load(response)

    def test_http_roundtrip_and_owner_enforcement(self):
        self.assertEqual(len(self.req('/api/conversations')['conversations']),2)
        self.web.agent_complete=model_fixture(['你好'])
        self.req('/api/agent/chat',{'conversation_id':A,'question':'你好'})
        self.assertEqual(len(self.req('/api/conversation?id='+A)['conversation']['messages']),2)
        task=self.plan()
        self.assertIsNone(self.req('/api/task/status?conversation_id='+B)['task'])
        self.assertTrue(self.req('/api/task/status?conversation_id='+B)['pending'])
        for operation in ('execute','cancel'):
            with self.assertRaises(HTTPError) as caught:
                self.req('/api/task/'+operation,{'task_id':task['id'],'conversation_id':B})
            self.assertEqual(caught.exception.code,400)

    def test_stream_disconnect_still_saves_answer_to_origin(self):
        ready=threading.Event()
        release=threading.Event()
        def complete(messages,tools):
            if [t['function']['name'] for t in tools] == ['set_scene_requirement']:
                return action('set_scene_requirement',scope='none',reason='聊天')
            ready.set()
            release.wait(3)
            return action('finish_response',intent='answer',text='连接断开后仍保存到A')
        self.web.agent_complete=complete
        req=Request(f'http://127.0.0.1:{self.server.server_port}/api/agent/chat',
                    data=json.dumps({'conversation_id':A,'question':'你好'}).encode(),
                    headers={'Content-Type':'application/json','Accept':'application/x-ndjson'})
        response=urlopen(req,timeout=5)
        self.assertTrue(ready.wait(2))
        response.close()
        release.set()
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            messages=self.store.conversation(A)['messages']
            if messages and messages[-1].get('text')=='连接断开后仍保存到A':break
            time.sleep(.02)
        self.assertEqual(messages[-1]['text'],'连接断开后仍保存到A')
        self.assertEqual(self.store.conversation(B)['messages'],[])


if __name__ == '__main__': unittest.main()
