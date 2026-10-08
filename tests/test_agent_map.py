"""Generic category/condition queries; no production object-name rules."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jaka_agent.agent.map_evidence import MapEvidence


class MapEvidenceTests(unittest.TestCase):
    def make(self, objects):
        return MapEvidence({'name': 'test', 'objects': objects})

    def test_arbitrary_categories_and_deduplication(self):
        evidence = self.make([{'ann_id': i, 'category': label} for i, label in enumerate(['设备甲', '设备乙', '设备甲'])])
        result = evidence.query('所有甲类设备', ['设备甲', '设备甲'])
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['target_ids'], ['0', '2'])
        self.assertEqual(result['groups'], [{'category': '设备甲', 'label': '设备甲', 'count': 2}])

    def test_counts_all_matches_not_only_display_page(self):
        evidence = self.make([{'ann_id': i, 'category': 'Thing'} for i in range(1100)])
        first = evidence.query('全部', ['Thing'])
        self.assertEqual(first['count'], 1100)
        self.assertEqual(len(first['objects']), 20)
        self.assertTrue(first['highlight_truncated'])
        self.assertEqual(evidence.query('全部', ['Thing'], offset=20)['count'], 1100)

    def test_unknown_category_does_not_silently_match_everything(self):
        evidence = self.make([{'ann_id': 1, 'category': 'A'}])
        with self.assertRaises(ValueError):
            evidence.query('不存在类别', ['invented'])
        self.assertEqual(evidence.query('没有匹配类别', [])['count'], 0)
        with self.assertRaises(ValueError):
            evidence.query('混合选择', ['*', 'A'])

    def test_generic_filters_and_case_insensitive_values(self):
        evidence = self.make([{'ann_id': i, 'category': 'A', 'position': location}
            for i, location in enumerate(['Room One', 'Room One', 'Room Two'])])
        result = evidence.query('指定区域', ['A'], [{'field': 'position', 'operator': 'equals', 'value': 'room one'}])
        self.assertEqual(result['count'], 2)
        self.assertEqual(evidence.query('全部区域', ['*'])['count'], 3)

    def test_missing_attributes_are_disclosed(self):
        evidence = self.make([{'ann_id': 1, 'category': 'A'}])
        result = evidence.query('某个区域', ['A'], [{'field': 'position', 'operator': 'contains', 'value': '会议室'}])
        self.assertEqual(result['count'], 0)
        self.assertEqual(result['missing_filter_fields'], ['position'])

    def test_excludes_inactive_and_ignores_identical_duplicates(self):
        item = {'ann_id': 1, 'category': 'A'}
        evidence = self.make([item, item.copy(), {'ann_id': 2, 'lifecycle': 'deleted'}, {'category': 'unknown'}])
        self.assertEqual(evidence.query('全部', ['*'])['count'], 1)
        self.assertEqual(evidence.excluded, 2)

    def test_conflicting_ids_fail_instead_of_guessing(self):
        with self.assertRaises(ValueError):
            self.make([{'ann_id': 1, 'category': 'A'}, {'ann_id': 1, 'category': 'B'}])

    def test_empty_map(self):
        evidence = self.make([])
        self.assertEqual(evidence.query('全部', ['*'])['count'], 0)
        self.assertIsNone(evidence.category_catalog()['next_offset'])

    def test_schema_sensitive_extra_data_not_exposed(self):
        result = self.make([{'ann_id': 1, 'category': 'A', 'private_extra': 'secret'}]).query('所有', ['A'])
        self.assertNotIn('secret', str(result))

    def test_catalog_paging_and_snapshot_changes(self):
        evidence = self.make([{'ann_id': i, 'category': str(i)} for i in range(100)])
        self.assertEqual(evidence.category_catalog()['next_offset'], 80)
        self.assertEqual(len(evidence.category_catalog(offset=80)['categories']), 20)
        self.assertNotEqual(evidence.version, self.make([]).version)


if __name__ == '__main__':
    unittest.main()
