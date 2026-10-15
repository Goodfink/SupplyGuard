import json
import random
import zipfile
from collections import defaultdict
from datetime import datetime

from supplyguard.config import DEFAULT_CONFIG
from supplyguard.errors import CorpusError
from supplyguard.helpers.data_helpers import get_published_date, is_valid_advisory

START_DATE = datetime.fromisoformat("2022-01-01T00:00:00+00:00")
CUTOFF_DATE = datetime.fromisoformat("2026-07-21T23:59:59+00:00")

SAMPLES_PER_YEAR = 50
YEARS = range(START_DATE.year, CUTOFF_DATE.year + 1)

ZIP_FILES = {
    "npm": DEFAULT_CONFIG.paths.raw_data_dir / "npm.zip",
    "Maven": DEFAULT_CONFIG.paths.raw_data_dir / "Maven.zip",
    "Go": DEFAULT_CONFIG.paths.raw_data_dir / "go.zip",
    "PyPI": DEFAULT_CONFIG.paths.raw_data_dir / "pypi.zip",
}


def collect_candidates(ecosystem, zip_path):
    candidates_by_year = defaultdict(list)
    seen_ids = set()

    if not zip_path.exists():
        raise CorpusError(f"Raw advisory archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        for filename in archive.namelist():
            if filename.endswith("/") or not filename.lower().endswith(".json"):
                continue

            try:
                with archive.open(filename) as f:
                    record = json.load(f)

            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            if not is_valid_advisory(record, START_DATE, CUTOFF_DATE):
                continue

            advisory_id = record.get("id")

            if advisory_id in seen_ids:
                continue

            published_date = get_published_date(record)

            if published_date is None:
                continue

            seen_ids.add(advisory_id)

            candidates_by_year[published_date.year].append(
                {
                    "record": record,
                    "source_ecosystem": ecosystem,
                    "source_filename": filename,
                    "published_year": published_date.year,
                }
            )

    return candidates_by_year


def select_candidates(ecosystem, candidates_by_year):

    selected = []

    for year in YEARS:
        candidates = candidates_by_year.get(year, [])
        candidates.sort(key=lambda item: item["record"].get("id", ""))

        if len(candidates) < SAMPLES_PER_YEAR:
            raise CorpusError(
                f"{ecosystem} only has {len(candidates)} candidates for year {year}; "
                f"at least {SAMPLES_PER_YEAR} are required"
            )

        random_generator = random.Random(f"{DEFAULT_CONFIG.runtime.random_seed}:{ecosystem}:{year}")
        year_selection = random_generator.sample(candidates, SAMPLES_PER_YEAR)
        selected.extend(year_selection)

    return selected


def extract_packages(record):
    packages = []
    seen_packages = set()

    for affected_item in record.get("affected", []):
        package = affected_item.get("package", {})

        name = package.get("name")
        ecosystem = package.get("ecosystem")
        purl = package.get("purl")

        if not name:
            continue

        package_key = (ecosystem or "", name, purl or "")

        if package_key in seen_packages:
            continue

        seen_packages.add(package_key)
        packages.append({"name": name, "ecosystem": ecosystem, "purl": purl})

    return packages


def extract_affected_ranges(record):
    affected_ranges = []

    for affected_item in record.get("affected", []):
        package = affected_item.get("package", {})
        package_name = package.get("name")

        for range_item in affected_item.get("ranges", []):
            affected_ranges.append(
                {
                    "package": package_name,
                    "type": range_item.get("type"),
                    "repository": range_item.get("repo"),
                    "events": range_item.get("events", []),
                }
            )

    return affected_ranges


def extract_fix_version(record):
    fix_versions = []
    seen_version = set()

    for affected_item in record.get("affected", []):
        package = affected_item.get("package", {})
        package_name = package.get("name")

        for range_item in affected_item.get("ranges", []):
            for event in range_item.get("events", []):
                fixed_version = event.get("fixed")

                if not fixed_version:
                    continue

                version_key = (package_name, fixed_version)

                if version_key in seen_version:
                    continue

                seen_version.add(version_key)
                fix_versions.append({"package": package_name, "version": fixed_version})

    return fix_versions


def extract_affected_versions(record):
    affected_versions = []

    for affected_item in record.get("affected", []):
        package = affected_item.get("package", {})
        package_name = package.get("name")

        versions = affected_item.get("versions", [])

        if versions:
            affected_versions.append({"package": package_name, "versions": versions})

    return affected_versions


def extract_references(record):
    references = []

    for reference in record.get("references", []):
        url = reference.get("url")

        if not url:
            continue

        references.append({"type": reference.get("type"), "url": url})

    return references


def build_retrieval_text(
    advisory_id, aliases, packages, summary, details, affected_ranges, fixed_versions
):

    sections = [f"Advisory ID: {advisory_id}"]

    if aliases:
        sections.append("Aliases: " + ", ".join(aliases))

    if packages:
        package_names = [package["name"] for package in packages]

        sections.append("Affected packages: " + ", ".join(package_names))

    if summary:
        sections.append(f"Summary: {summary}")

    if details:
        sections.append(f"Details: {details}")

    if fixed_versions:
        fixes_text = [f"{item['package']} fixed in {item['version']}" for item in fixed_versions]
        sections.append("Fixed versions: " + ", ".join(fixes_text))

    for affected_range in affected_ranges:
        event_text = []

        for event in affected_range.get("events", []):
            for event_type, version in event.items():
                event_text.append(f"{event_type} {version}")

        if event_text:
            sections.append(
                f"Affected range for {affected_range.get('package')} ({affected_range.get('type')}):"
                + ", ".join(event_text)
            )

    return "\n".join(sections)


def normalize_advisory(candidate):
    record = candidate["record"]
    advisory_id = record.get("id", "")
    aliases = record.get("aliases") or []

    summary = (record.get("summary") or "").strip()
    details = (record.get("details") or "").strip()

    packages = extract_packages(record)
    affected_ranges = extract_affected_ranges(record)
    affected_versions = extract_affected_versions(record)
    fixed_versions = extract_fix_version(record)
    references = extract_references(record)

    retrieval_text = build_retrieval_text(
        advisory_id=advisory_id,
        aliases=aliases,
        packages=packages,
        summary=summary,
        details=details,
        affected_ranges=affected_ranges,
        fixed_versions=fixed_versions,
    )

    return {
        "document_id": advisory_id,
        "chunk_id": f"{advisory_id}_chunk_0",
        "advisory_id": advisory_id,
        "aliases": aliases,
        "source_ecosystem": candidate["source_ecosystem"],
        "packages": packages,
        "summary": summary,
        "details": details,
        "affected_ranges": affected_ranges,
        "affected_versions": affected_versions,
        "fixed_versions": fixed_versions,
        "severity": record.get("severity") or [],
        "database_specific": record.get("database_specific") or {},
        "published": record.get("published"),
        "modified": record.get("modified"),
        "published_year": candidate["published_year"],
        "references": references,
        "source_filename": candidate["source_filename"],
        "retrieval_text": retrieval_text,
    }


def write_jsonl(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest(records, output_path):
    manifest = {
        "random_seed": DEFAULT_CONFIG.runtime.random_seed,
        "start_date": START_DATE.isoformat(),
        "cutoff_date": CUTOFF_DATE.isoformat(),
        "samples_per_year": SAMPLES_PER_YEAR,
        "years": list(YEARS),
        "total_documents": len(records),
        "selected_documents": [
            {
                "document_id": record["document_id"],
                "ecosystem": record["source_ecosystem"],
                "year": record["published_year"],
            }
            for record in records
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main():
    selected_candidates = []

    for ecosystem, zip_path in ZIP_FILES.items():
        candidates_by_year = collect_candidates(ecosystem, zip_path)
        ecosystem_selection = select_candidates(ecosystem, candidates_by_year)
        selected_candidates.extend(ecosystem_selection)

    processed_records = [normalize_advisory(candidate) for candidate in selected_candidates]
    processed_records.sort(
        key=lambda record: (
            record["source_ecosystem"],
            record["published_year"],
            record["document_id"],
        )
    )

    corpus_path = DEFAULT_CONFIG.paths.corpus_path
    manifest_path = DEFAULT_CONFIG.paths.processed_data_dir / "corpus_manifest.jsonl"

    write_jsonl(processed_records, corpus_path)
    write_manifest(processed_records, manifest_path)

    print(f"Saved {len(processed_records):} processed advisories")
    print(f"Manifest {manifest_path}")


if __name__ == "__main__":
    main()
