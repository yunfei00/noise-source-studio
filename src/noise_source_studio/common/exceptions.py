"""Application-specific exception hierarchy."""


class NoiseSourceStudioError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(NoiseSourceStudioError):
    """Raised when persisted configuration cannot be loaded or saved."""


class InferenceEngineNotConfiguredError(NoiseSourceStudioError):
    """Raised when inference is requested without a configured engine."""


class RuntimeUnavailableError(NoiseSourceStudioError):
    """Raised when the separately delivered inference runtime is unavailable."""


class ModelPackageValidationError(NoiseSourceStudioError):
    """Raised when a model package does not satisfy the supported contract."""


class RuntimeCompatibilityError(ModelPackageValidationError):
    """Raised when a model package requires an incompatible runtime."""


class DuplicateModelError(NoiseSourceStudioError):
    """Raised when the same model name and version are already registered."""


class ModelActivationError(NoiseSourceStudioError):
    """Raised when a verified model cannot be activated safely."""


class DeviceSelectionError(NoiseSourceStudioError):
    """Raised when a requested compute device cannot be used safely."""


class ActiveModelDeletionError(NoiseSourceStudioError):
    """Raised when deletion is requested for the active model."""


class PredictionBusyError(NoiseSourceStudioError):
    """Raised when another inference operation already owns the session."""


class PredictionExecutionError(NoiseSourceStudioError):
    """Raised when runtime prediction fails."""


class PredictionExportError(NoiseSourceStudioError):
    """Raised when a prediction result cannot be exported."""
