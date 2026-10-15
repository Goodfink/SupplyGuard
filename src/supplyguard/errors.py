"""Domain-specific exceptions surfaced by SupplyGuard commands."""


class SupplyGuardError(Exception):
    """Base class for expected, user-actionable SupplyGuard failures."""

    exit_code = 1


class ConfigurationError(SupplyGuardError):
    """Raised when runtime configuration or command arguments are invalid."""

    exit_code = 2


class CorpusError(SupplyGuardError):
    """Raised when advisory data cannot be read or validated."""

    exit_code = 3


class RetrievalIndexError(SupplyGuardError):
    """Raised when a retrieval index cannot be built or queried."""

    exit_code = 4


class ModelError(SupplyGuardError):
    """Base class for model loading and inference failures."""

    exit_code = 5


class ModelLoadError(ModelError):
    """Raised when an embedding or language model cannot be loaded."""


class ModelInferenceError(ModelError):
    """Raised when an already loaded model cannot complete inference."""


class ModelGenerationError(ModelInferenceError):
    """Raised when the language model cannot generate a usable answer."""
