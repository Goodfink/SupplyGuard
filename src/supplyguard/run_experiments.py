import subprocess
import sys

from supplyguard.paths import PROJECT_ROOT


def main():
    """Run the no-retrieval baseline and both RAG evaluations."""
    subprocess.run(
        [sys.executable, "-u", "-m", "supplyguard.run_baseline"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-u", "-m", "supplyguard.cli", "evaluate"],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
