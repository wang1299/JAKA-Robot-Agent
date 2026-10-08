"""Regression coverage for real package startup and migration of local data."""
import os
import base64
import io
import threading
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from jaka_agent import paths
from tools.smoke_web import smoke


class PackageRuntimeTests(unittest.TestCase):
    def test_module_starts_outside_checkout_and_serves_packaged_assets(self):
        smoke()

    def test_legacy_entry_starts_outside_checkout(self):
        smoke(legacy=True)

    def test_existing_legacy_data_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / 'legacy'
            legacy.mkdir()
            (legacy / 'conversation_data').mkdir()
            with patch.object(paths, 'LEGACY_DIR', legacy), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(paths.runtime_path('conversation_data'), legacy / 'conversation_data')

    def test_explicit_data_directory_overrides_legacy_data(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / 'legacy'
            legacy.mkdir()
            (legacy / 'conversation_data').mkdir()
            target = Path(directory) / 'new'
            with patch.object(paths, 'LEGACY_DIR', legacy), patch.object(paths, 'DATA_DIR', target), \
                    patch.dict(os.environ, {'JAKA_DATA_DIR': str(target)}):
                self.assertEqual(paths.runtime_path('conversation_data'), target / 'conversation_data')

    def test_local_model_config_precedes_packaged_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / 'configs').mkdir()
            local = work / 'configs/model_config.local.json'
            local.write_text('{}', encoding='utf8')
            with patch.object(paths, 'LEGACY_DIR', work):
                self.assertEqual(paths.model_config_path(), local)

    def test_site_maps_precede_packaged_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            site = work / 'maps'
            site.mkdir()
            name = 'zmq_scene_graph.json'
            (site / name).write_text('{}', encoding='utf8')
            with patch.object(paths, 'MAPS_DIR', site), patch.object(paths, 'LEGACY_DIR', work):
                self.assertEqual(paths.resolve_map(name), site / name)

    def test_uploaded_slam_and_calibration_survive_reload(self):
        from jaka_agent.mapping.catalog import MappingCatalogMixin
        from jaka_agent.web import settings

        class Catalog(MappingCatalogMixin):
            def __init__(self):
                self.graph = {'objects': [{'floor_xy': [0, 0]}]}
                self.slam_lock = threading.RLock()
                self.slam_metadata = self.slam_fetched_at = self.slam_fetch_error = None

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            maps = work / 'maps'
            maps.mkdir()
            image = io.BytesIO()
            Image.new('RGB', (20, 20), 'white').save(image, format='PNG')
            encoded = base64.b64encode(image.getvalue()).decode('ascii')
            with patch.object(paths, 'MAPS_DIR', maps), patch.object(paths, 'LEGACY_DIR', work), \
                    patch.object(settings, 'SLAM_CONFIG_PATH', work / 'slam_calibration.json'), \
                    patch.dict(os.environ, {}, clear=True):
                state = Catalog()
                state.upload_slam_image('site.png', encoded)
                expected = state.slam_image_path
                state.update_slam_calibration({'origin_x': 3, 'origin_y': 4, 'resolution': .05,
                                               'rotation': 0, 'opacity': .5})
                restored = Catalog()
                restored._load_slam_image()
                self.assertEqual(restored.slam_image_path, expected)
                self.assertEqual(expected.parent, maps)
                self.assertEqual(restored.slam_calibration['origin_x'], 3)


if __name__ == '__main__':
    unittest.main()
