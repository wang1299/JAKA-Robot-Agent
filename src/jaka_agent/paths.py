"""Resolve packaged assets and writable runtime data independently of source files."""
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = PACKAGE_DIR / 'resources'
TEMPLATES_DIR = PACKAGE_DIR / 'web/templates'
STATIC_DIR = PACKAGE_DIR / 'web/static'
LEGACY_DIR = Path.cwd().resolve()
DATA_DIR = Path(os.getenv('JAKA_DATA_DIR', str(LEGACY_DIR / 'data'))).expanduser().resolve()
MAPS_DIR = DATA_DIR / 'maps'


def runtime_path(name):
    """Reuse existing legacy data unless an explicit data directory was requested."""
    legacy = LEGACY_DIR / name
    return legacy if not os.getenv('JAKA_DATA_DIR') and legacy.exists() else DATA_DIR / name


def model_config_path():
    for path in (LEGACY_DIR / 'configs/model_config.local.json', LEGACY_DIR / 'model_config.json'):
        if path.is_file():
            return path
    return RESOURCES_DIR / 'model_config.json'


def map_directories():
    return (MAPS_DIR, LEGACY_DIR, RESOURCES_DIR / 'maps')


def resolve_map(name):
    path = Path(name).expanduser()
    if path.is_absolute():
        return path
    return next((folder / path for folder in map_directories() if (folder / path).is_file()), MAPS_DIR / path)


def slam_candidates():
    names = ('slam.png', 'slam_live.png', 'slam_live.jpg', 'slam.jpg', 'slam.jpeg')
    return [folder / name for folder in map_directories() for name in names]


def ensure_data_directories():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
