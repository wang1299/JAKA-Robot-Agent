"""Load model routing before the planner imports its model settings."""
def main():
    from jaka_agent.models.config import load_model_config
    load_model_config()
    from jaka_agent.paths import ensure_data_directories
    ensure_data_directories()
    from jaka_agent.cli.planner import main as run
    run()
