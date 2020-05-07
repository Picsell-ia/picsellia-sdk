class PicselliaError(Exception):
    """Base class for exceptions."""
    def __init__(self, message):
        """
        Args:
            message (str): Informative message about the exception.
            cause (Exception): The cause of the exception (an Exception
                raised by Python or another library). Optional.
        """
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class AuthenticationError(PicselliaError):
    """Raised when your token does not match to any known token"""
    pass


class ResourceNotFoundError(PicselliaError):
    """Exception raised when a given resource is not found. """

    def __init__(self, message):

        super().__init__(message)


class InvalidQueryError(PicselliaError):
    """ Indicates a malconstructed or unsupported query. This can be the result of either client
    or server side query validation. """
    pass


class NetworkError(PicselliaError):
    """Raised when an HTTPError occurs."""
    def __init__(self, message):
        super().__init__(message)


class ApiLimitError(PicselliaError):
    """ Raised when the user performs too many requests in a short period
    of time. """
    pass


class ProcessingError(PicselliaError):
    """Raised when an algorithmic error occurs."""
    def __init__(self, message):
        super().__init__(message)
