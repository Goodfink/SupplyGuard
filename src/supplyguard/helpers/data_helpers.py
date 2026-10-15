from datetime import datetime


def get_published_date(record):
    date_string = record.get("published")

    if not date_string:
        return None

    try:
        return datetime.fromisoformat(
            date_string.replace("Z", "+00:00")
        )
    except (ValueError, TypeError):
        return None


def is_ghsa_advisory(record):
    advisory_id = record.get("id", "")
    return advisory_id.startswith("GHSA-")


def is_malicious_advisory(record):
    advisory_id = record.get("id", "")
    return advisory_id.startswith("MAL-")


def is_withdrawn_advisory(record):
    return bool(record.get("withdrawn"))


def is_in_date_range(record, start_date, cutoff_date):
    published_date = get_published_date(record)

    return (
        published_date is not None
        and start_date <= published_date <= cutoff_date
    )


def has_affected_package(record):
    for affected_item in record.get("affected", []):
        package = affected_item.get("package", {})

        if package.get("name"):
            return True

    return False


def has_description(record):
    summary = record.get("summary") or ""
    details = record.get("details") or ""

    return bool(
        summary.strip()
        or details.strip()
    )


def is_valid_advisory(record, start_date, cutoff_date):
    advisory_id = record.get("id", "")

    return (
        bool(advisory_id)
        and not is_malicious_advisory(record)
        and not is_withdrawn_advisory(record)
        and is_in_date_range(record, start_date, cutoff_date)
        and has_affected_package(record)
        and has_description(record)
    )