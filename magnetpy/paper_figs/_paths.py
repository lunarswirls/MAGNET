from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DATA_DIR = REPO_ROOT / "example_data"


def example_data_path(*parts: str) -> Path:
    return EXAMPLE_DATA_DIR.joinpath(*parts)
