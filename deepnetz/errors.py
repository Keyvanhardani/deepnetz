"""DeepNetz error hierarchy."""


class DeepNetzError(Exception):
    """Base error for all DeepNetz exceptions."""
    pass


class ModelNotFoundError(DeepNetzError):
    """Model file or reference could not be resolved."""
    pass


class BackendNotAvailableError(DeepNetzError):
    """Requested backend is not installed or not running."""
    pass


class HardwareInsufficientError(DeepNetzError):
    """Model doesn't fit in available hardware."""
    pass


class BackendError(DeepNetzError):
    """Error from the inference backend during generation."""
    pass


class SecurityError(DeepNetzError):
    """Security violation (e.g., untrusted model format)."""
    pass
