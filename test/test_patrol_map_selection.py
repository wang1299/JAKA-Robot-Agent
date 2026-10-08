"""Map-picked patrols use exact IDs/order and never execute while planning."""
import json
import sys
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robot_web
from robot_agent_map import MapEvidence
from test_robot_skills import web_state


class PatrolSelectionTests(unittest.TestCase):
    def setUp(self):
        self.web = web_state()
        self.web.graph_snapshot.return_value['objects'].append(
            {'ann_id': 88, 'category_zh': '储物柜', 'floor_xy': [5, 6]})
        self.snapshot = MapEvidence(self.web.graph_snapshot()).version
        self.server = robot_web.RobotWebServer(('127.0.0.1', 0), robot_web.RobotWebHandler, self.web)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.web.tasks.execute.assert_not_called()
        self.web.capture.assert_not_called()
        self.web.speak.assert_not_called()

    def post(self, **updates):
        payload = dict(instruction='按地图选点巡逻', target_ann_ids=[88, 7, 19], rounds=-1, map_snapshot=self.snapshot)
        payload.update(updates)
        request = Request(f'http://127.0.0.1:{self.server.server_port}/api/task/patrol/plan',
                          data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=2) as response:
            return json.load(response)['task']

    def test_three_selected_points_preserve_order_without_model(self):
        with patch('qwen_planner.plan_task', side_effect=AssertionError('must not reinterpret IDs')):
            task = self.post()
        self.assertEqual(task['steps'][0]['target_ann_ids'], [88, 7, 19])
        self.assertEqual(task['status'], 'planned')
        self.assertEqual(task['steps'][0]['count'], -1)
        self.assertEqual(task['skill']['map_snapshot'], self.snapshot)

    def test_two_points_finite_rounds(self):
        task = self.post(target_ann_ids=[19, 88], rounds=2)
        self.assertEqual(task['steps'][0]['target_ann_ids'], [19, 88])
        self.assertEqual(task['steps'][0]['count'], 2)

    def test_invalid_or_stale_selection_cannot_replace_task(self):
        original = self.post()
        for payload in ({'target_ann_ids': [7]}, {'target_ann_ids': [7, 7]},
                        {'target_ann_ids': [7, 99]}, {'target_ann_ids': [True, 19]},
                        {'rounds': 0}, {'map_snapshot': 'old'}, {'map_snapshot': None}):
            with self.subTest(payload=payload), self.assertRaises(HTTPError):
                self.post(**payload)
            self.assertEqual(self.web.tasks.task['id'], original['id'])

    def test_same_name_map_change_invalidates_selection(self):
        self.web.graph_snapshot.return_value['objects'][0]['floor_xy'] = [100, 200]
        with self.assertRaises(HTTPError):
            self.post()
        self.assertIsNone(self.web.tasks.task)


if __name__ == '__main__':
    unittest.main()
