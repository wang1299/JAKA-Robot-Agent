"""Regression checks for approach selection using a mock chassis."""
import unittest
from unittest.mock import Mock, patch
import qwen_planner as p
from robot_runtime import TaskCancelled


class OriginalApproachTests(unittest.TestCase):
    def setUp(self):
        self.driver = Mock()
        self.driver.get_pose.return_value = (-4, 0, 0)
        self.obj = {'ann_id':65,'category':'case','floor_xy':[0,0],
                    'box3d':{'center':[0,0,0],'size':[2,2,2]}}
        self.ex = p.PlanExecutor([self.obj],self.driver,mark_start=False)

    def test_original_rank_prefers_small_turn_not_nearest_edge(self):
        self.driver.accessible_point_query.side_effect=lambda x,y:(x,y)
        result=self.ex._query_accessible_candidates(
            [(-2.4,0,'straight'),(0,1.7,'side')],(0,0),(-4,0,0),[],.35)
        self.assertEqual(result[2:4],(-2.4,0))

    def test_center_query_accepted_without_global_boundary_search(self):
        self.driver.accessible_point_query.return_value=(-1,0)
        x,y,theta,source=self.ex._find_accessible_target(self.obj)
        self.assertEqual((x,y,theta),(-1,0,0))
        self.driver.accessible_point_query.assert_called_once_with(0,0)
        self.driver.move_location.assert_not_called()

    def test_viewpoint_keeps_original_priority(self):
        self.obj['viewpoint']={'x':-2,'y':0}
        self.driver.accessible_point_query.return_value=(-2,0)
        self.assertEqual(self.ex._find_accessible_target(self.obj,True)[:3],(-2,0,0))
        self.driver.accessible_point_query.assert_called_once_with(-2,0)

    def test_unsafe_center_distance_rejected(self):
        self.driver.accessible_point_query.return_value=(.1,0)
        errors=[]
        self.assertIsNone(self.ex._query_accessible_candidates([(0,0,'center')],(0,0),(-4,0,0),errors,.35))
        self.assertTrue(errors)

    def test_query_and_pose_cancellation_not_swallowed(self):
        self.driver.accessible_point_query.side_effect=TaskCancelled('cancel')
        with self.assertRaises(TaskCancelled): self.ex._find_accessible_target(self.obj)
        self.driver.get_pose.side_effect=TaskCancelled('cancel')
        with self.assertRaises(TaskCancelled): self.ex._find_accessible_target(self.obj)

    def test_failed_navigation_never_starts_orientation(self):
        self.ex._find_accessible_target=Mock(return_value=(-1,0,0,'test'))
        self.ex._face_object_center=Mock()
        self.driver.wait_until_settled.return_value='failed'
        self.assertEqual(self.ex._do_navigate({'target_ann_id':65}),'failed')
        self.ex._face_object_center.assert_not_called()

    def test_successful_navigation_faces_object_after_arrival_callback(self):
        order=[]
        self.ex._find_accessible_target=Mock(return_value=(-1,0,0,'test'))
        self.driver.wait_until_settled.side_effect=lambda **kw: order.append('arrived') or 'succeeded'
        self.ex._face_object_center=Mock(side_effect=lambda obj:order.append('face') or 'succeeded')
        with patch.object(p,'announce'):
            self.assertEqual(self.ex._do_navigate({'target_ann_id':65,
                '_on_translation_arrived':lambda:order.append('stop_sampling')}),'succeeded')
        self.assertEqual(order,['arrived','stop_sampling','face'])


if __name__ == '__main__': unittest.main()
