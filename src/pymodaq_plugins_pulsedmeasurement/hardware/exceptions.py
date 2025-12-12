class PulseBlasterError(Exception):
    """Base exception for PulseBlaster-related errors."""

    pass


class BoardNotFoundError(Exception):
    """Raised when no PulseBlaster board is found."""

    pass


class InvalidChannelError(Exception):
    """Raised when an invalid channel is accessed."""

    pass


class FirmwareMismatchError(Exception):
    """Raised when the firmware is not recognized and user input is invalid."""

    pass
