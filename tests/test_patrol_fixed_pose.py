"""Real patrol loop + fake driver/model: never connects to robot hardware."""
import jaka_agent.tasks.executor as ja_tasks_executor
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tests.test_patrol_speech import PatrolSpeechTests
from jaka_agent.tasks.runtime import cancellation_scope, TaskCancelled

class PatrolFixedPoseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manager,self.web=PatrolSpeechTests().manager([{'changed':False,'confidence':'high'}]*4)
        self.manager._capture_path=lambda cid,**_:Path(self.temp.name)/(cid+'.jpg')
        self.web.capture_path=lambda cid:Path(self.temp.name)/(cid+'.jpg')
        self.driver=Mock()
        self.pose=[0.,0.,0.]
        self.offset=0.
        self.driver.get_pose.side_effect=lambda:tuple(self.pose)
        def move(x,y,theta,**kwargs):self.pose[:]=[x+self.offset,y,theta]
        self.driver.move_location.side_effect=move
        self.driver.wait_until_settled.return_value='succeeded'
        self.driver.capture.side_effect=lambda path:Path(path).write_bytes(b'saved frame')
        self.manager._task_camera_driver.return_value=self.driver
        self.executor=Mock(log=[])
        def navigate(step):
            aid=step['target_ann_id'];self.pose[:]=[float(aid),float(aid+1),.1]
            return 'succeeded'
        def observe(step):
            # Initial framing changes the pose; this final pose must be saved.
            self.pose[:]=[self.pose[0]+.2,self.pose[1]+.1,.4]
            return 'succeeded'
        self.executor._do_navigate.side_effect=navigate
        self.executor._do_observe.side_effect=observe

    def run_loop(self):
        with patch.object(ja_tasks_executor,'PlanExecutor',return_value=self.executor):
            self.manager._run_patrol_hardware()

    def test_three_rounds_reuse_first_actual_pose_not_new_candidates(self):
        self.run_loop()
        poses=self.manager.task['patrol_observation_poses']
        self.assertAlmostEqual(poses['1']['x'],1.2)
        self.assertAlmostEqual(poses['2']['x'],2.2)
        self.assertEqual(self.executor._do_navigate.call_count,2)
        self.assertEqual(self.executor._do_observe.call_count,2)
        self.assertEqual(self.driver.capture.call_count,6)
        self.assertEqual(self.web.compare_patrol_images.call_count,4)
        self.assertEqual(self.manager.task['status'],'succeeded')
        for call,aid in zip(self.driver.move_location.call_args_list,[1,2,1,2]):
            self.assertEqual(call.args,tuple(poses[str(aid)][k] for k in ('x','y','theta')))
            self.assertEqual(call.kwargs['distance_tolerance'],.08)
            self.assertEqual(call.kwargs['theta_tolerance'],math.radians(3))

    def test_rolling_baseline_does_not_shift_saved_navigation_pose(self):
        self.offset=.02
        self.run_loop()
        self.assertAlmostEqual(self.manager.task['patrol_observation_poses']['1']['x'],1.2)
        self.assertAlmostEqual(self.manager.task['patrol_baselines']['1']['capture_pose']['x'],1.22)
        self.assertAlmostEqual(self.driver.move_location.call_args_list[2].args[0],1.2)

    def test_success_status_with_wrong_position_does_not_compare(self):
        self.offset=.3
        with self.assertRaisesRegex(RuntimeError,'观察位姿不一致'):self.run_loop()
        self.web.compare_patrol_images.assert_not_called()
        self.assertEqual(self.driver.capture.call_count,2)

    def test_success_status_with_wrong_heading_does_not_compare(self):
        self.driver.move_location.side_effect=lambda x,y,theta,**kw:self.pose.__setitem__(slice(None),[x,y,theta+.3])
        with self.assertRaisesRegex(RuntimeError,'朝向误差'):self.run_loop()
        self.web.compare_patrol_images.assert_not_called()

    def test_heading_wraparound_is_equivalent(self):
        self.manager._check_patrol_pose({'x':0,'y':0,'theta':-math.pi+.01},{'x':0,'y':0,'theta':math.pi-.01},1)

    def test_missing_pose_never_falls_back_to_replanning(self):
        with self.assertRaisesRegex(RuntimeError,'缺少有效首轮观察位姿'):
            self.manager._return_to_patrol_pose(self.driver,1,None)
        self.driver.move_location.assert_not_called()

    def test_movement_during_capture_does_not_publish_baseline(self):
        def capture(path):
            Path(path).write_bytes(b'frame');self.pose[0]+=.4
        self.driver.capture.side_effect=capture
        with self.assertRaisesRegex(RuntimeError,'观察位姿不一致'):self.run_loop()
        self.assertFalse(self.manager.task['patrol_baselines'])
        self.manager._record_capture.assert_not_called()
        self.web.compare_patrol_images.assert_not_called()

    def test_cancel_during_capture_does_not_publish_baseline(self):
        canceled=False
        def capture(path):
            nonlocal canceled
            Path(path).write_bytes(b'frame');canceled=True
        self.driver.capture.side_effect=capture
        with cancellation_scope(lambda:canceled),self.assertRaises(TaskCancelled):self.run_loop()
        self.assertFalse(self.manager.task['patrol_baselines'])
        self.web.compare_patrol_images.assert_not_called()

    def test_failed_return_does_not_compare_or_announce_normal(self):
        self.driver.wait_until_settled.return_value='failed'
        self.run_loop()
        self.assertEqual(self.manager.task['status'],'aborted')
        self.web.compare_patrol_images.assert_not_called()
        self.assertNotIn('无异常',[c.args[0] for c in self.web.speak.call_args_list])

    def test_nonfinite_pose_is_rejected(self):
        self.driver.get_pose.side_effect=lambda:(float('nan'),0,0)
        with self.assertRaisesRegex(RuntimeError,'位姿无效'):self.run_loop()
        self.assertFalse(self.manager.task['patrol_baselines'])

if __name__=='__main__':unittest.main()
