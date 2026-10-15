from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def _find_project_root():
    for candidate in PACKAGE_ROOT.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "data").is_dir():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CORPUS_PATH = PROCESSED_DATA_DIR / "advisories.jsonl"
EVALUATION_FILE = PACKAGE_ROOT / "evaluation" / "evaluation_set.json"
DEFAULT_RESULTS_FILE = PROJECT_ROOT / "evaluation_results.json"
