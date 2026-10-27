import subprocess
import sys

from supplyguard.config import DEFAULT_CONFIG


def main():
    """Run the no-retrieval baseline and all RAG evaluations."""
    subprocess.run(
        [sys.executable, "-u", "-m", "supplyguard.run_baseline"],
        cwd=DEFAULT_CONFIG.paths.project_root,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-u", "-m", "supplyguard.cli", "evaluate"],
        cwd=DEFAULT_CONFIG.paths.project_root,
        check=True,
    )


if __name__ == "__main__":
    main()
