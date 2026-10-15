import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    """Run the no-retrieval baseline and both RAG evaluations."""
    subprocess.run(
        [sys.executable, "-u", "-m", "src.run_baseline"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-u", "-m", "src.rag_cli", "--evaluate", "all"],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
