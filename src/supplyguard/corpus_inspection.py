import json
import zipfile
from datetime import datetime

from supplyguard.config import DEFAULT_CONFIG
from supplyguard.corpus_stats import CorpusStats
from supplyguard.errors import CorpusError
from supplyguard.helpers.data_helpers import (
    has_affected_package,
    has_description,
    is_ghsa_advisory,
    is_in_date_range,
    is_malicious_advisory,
    is_valid_advisory,
    is_withdrawn_advisory,
)

START_DATE = datetime.fromisoformat("2022-01-01T00:00:00+00:00")
CUTOFF_DATE = datetime.fromisoformat("2026-07-21T23:59:59+00:00")

ZIP_FILES = {
    "npm": DEFAULT_CONFIG.paths.raw_data_dir / "npm.zip",
    "Maven": DEFAULT_CONFIG.paths.raw_data_dir / "Maven.zip",
    "Go": DEFAULT_CONFIG.paths.raw_data_dir / "go.zip",
    "PyPI": DEFAULT_CONFIG.paths.raw_data_dir / "pypi.zip",
}


def inspect_zip(ecosystem, zip_path):

    stats = CorpusStats()

    if not zip_path.is_file():
        raise CorpusError(f"Raw advisory archive not found: {zip_path}")

    try:
        archive_context = zipfile.ZipFile(zip_path, "r")
    except (OSError, zipfile.BadZipFile) as error:
        raise CorpusError(f"Could not open advisory archive '{zip_path}': {error}") from error

    with archive_context as archive:
        for filename in archive.namelist():
            if filename.endswith("/") or not filename.lower().endswith(".json"):
                continue

            stats.total_json += 1

            try:
                with archive.open(filename) as f:
                    record = json.load(f)

            except (json.JSONDecodeError, UnicodeDecodeError):
                stats.invalid_json += 1
                continue

            if is_ghsa_advisory(record):
                stats.ghsa += 1

            if is_malicious_advisory(record):
                stats.malicious += 1

            if is_withdrawn_advisory(record):
                stats.withdrawn += 1

            if is_in_date_range(record, START_DATE, CUTOFF_DATE):
                stats.published_in_date_range += 1

            if has_affected_package(record):
                stats.has_package += 1

            if has_description(record):
                stats.has_description += 1

            if is_valid_advisory(record, START_DATE, CUTOFF_DATE):
                stats.valid_candidates += 1

        print(ecosystem)
        for name, val in vars(stats).items():
            print(f"{name}: {val}")


def main():
    for ecosystem, zip_path in ZIP_FILES.items():
        inspect_zip(ecosystem, zip_path)


if __name__ == "__main__":
    main()
