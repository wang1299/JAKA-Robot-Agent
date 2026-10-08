"""Heading tolerance checks with mock drivers only; never move hardware."""
import jaka_agent.models.settings as ja_models_settings
import jaka_agent.tasks.events as ja_tasks_events
import jaka_agent.tasks.executor as ja_tasks_executor
import math
import unittest
from unittest.mock import Mock, patch



class HeadingToleranceTests(unittest.TestCase):
    def executor(self, poses):
        driver = Mock()
        driver.get_pose.side_effect = poses
        obj = {'ann_id': 65, 'category': 'case', 'floor_xy': [2, 0]}
        return ja_tasks_executor.PlanExecutor([obj], driver, mark_start=False), driver, obj

    def test_ten_degree_residual_does_not_trigger_extra_rotation(self):
        for angle in (0, 1, 9.9, -9.9):
            pose = (0, 0, math.radians(angle))
            executor, driver, obj = self.executor([pose, pose])
            with patch.object(ja_models_settings, 'NAV_GOAL_THETA_TOLERANCE_DEG', 10):
                self.assertEqual(executor._face_object_center(obj), 'succeeded')
            driver.move_location.assert_not_called()

    def test_larger_error_uses_same_tolerance_for_correction(self):
        pose = (0, 0, math.radians(20))
        near = (0, 0, math.radians(2))
        executor, driver, obj = self.executor([pose, pose, near, near])
        driver.wait_until_settled.return_value = 'succeeded'
        with patch.object(ja_models_settings, 'NAV_GOAL_THETA_TOLERANCE_DEG', 10):
            self.assertEqual(executor._face_object_center(obj), 'succeeded')
        self.assertEqual(driver.move_location.call_count, 1)
        self.assertAlmostEqual(driver.move_location.call_args.kwargs['theta_tolerance'], math.radians(10))

    def test_navigation_goal_uses_ten_degrees(self):
        executor, driver, obj = self.executor([(0, 0, 0)])
        obj['navigation_pose'] = {'x': 1, 'y': 0, 'theta': 0}
        driver.wait_until_settled.return_value = 'succeeded'
        with patch.object(ja_models_settings, 'NAV_GOAL_THETA_TOLERANCE_DEG', 10), patch.object(ja_tasks_events, 'announce'):
            self.assertEqual(executor._do_navigate({'target_ann_id': 65}), 'succeeded')
        self.assertAlmostEqual(driver.move_location.call_args.kwargs['theta_tolerance'], math.radians(10))


if __name__ == '__main__':
    unittest.main()
