"""Application-specific exception hierarchy."""


class NoiseSourceStudioError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(NoiseSourceStudioError):
    """Raised when persisted configuration cannot be loaded or saved."""


class InferenceEngineNotConfiguredError(NoiseSourceStudioError):
    """Raised when inference is requested without a configured engine."""
