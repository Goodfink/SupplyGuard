"""Validation boundaries for user-controlled RAG inputs."""

import unicodedata

from supplyguard.errors import ConfigurationError, InputValidationError

ALLOWED_CONTROL_CHARACTERS = {"\n", "\r", "\t"}


def validate_question(question: str, *, max_chars: int) -> str:
    """Return a normalized question or reject unsafe and ambiguous input."""
    if max_chars <= 0:
        raise ConfigurationError("max_chars must be greater than 0")
    if not isinstance(question, str):
        raise InputValidationError("Question must be text")

    normalized = question.strip()
    if not normalized:
        raise InputValidationError("Question must not be empty")
    if len(normalized) > max_chars:
        raise InputValidationError(f"Question exceeds the maximum length of {max_chars} characters")

    for character in normalized:
        if character in ALLOWED_CONTROL_CHARACTERS:
            continue
        if unicodedata.category(character) in {"Cc", "Cf"}:
            raise InputValidationError("Question contains unsupported control characters")

    return normalized
