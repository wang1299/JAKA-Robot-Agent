#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_web_routing import _project_metric_query


class ProjectMetricQueryTests(unittest.TestCase):
    def test_semantic_graph_scale_query(self):
        answer = _project_metric_query("查询语义图谱规模")
        self.assertIn("105,000", answer)
        self.assertIn("760,649", answer)

    def test_object_recognition_metric_query(self):
        answer = _project_metric_query("查询空间对象识别指标")
        self.assertIn("308", answer)
        self.assertIn("92.45%", answer)

    def test_spatial_object_count_query(self):
        answer = _project_metric_query("查询空间拟物体数量")
        self.assertIn("2,038", answer)
        for scene in ("办公", "居家", "教育"):
            self.assertIn(scene, answer)

    def test_unrelated_question_is_not_intercepted(self):
        self.assertIsNone(_project_metric_query("地图里有几个椅子"))


if __name__ == "__main__":
    unittest.main()
