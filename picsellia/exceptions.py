class PicselliaError(Exception):
    """Base class for exceptions."""
    def __init__(self, message, cause=None):
        """
        Args:
            message (str): Informative message about the exception.
            cause (Exception): The cause of the exception (an Exception
                raised by Python or another library). Optional.
        """
        super().__init__(message, cause)
        self.message = message
        self.cause = cause

    def __str__(self):
        return self.message


class AuthenticationError(PicselliaError):
    """Raised when your token does not match to any known token"""
    pass


class ResourceNotFoundError(PicselliaError):
    """Exception raised when a given resource is not found. """

    def __init__(self, project_id, training_id):
        """ Constructor.
        Args:
            db_object_type (type): A labelbox.schema.DbObject subtype.
            params (dict): Dict of params identifying the sought resource.
        """
        super().__init__("Resouce '%s' not found for project: %r training id : %s" % (project_id, training_id ))


class InvalidQueryError(PicselliaError):
    """ Indicates a malconstructed or unsupported query. This can be the result of either client
    or server side query validation. """
    pass


class NetworkError(PicselliaError):
    """Raised when an HTTPError occurs."""
    def __init__(self):
        super().__init__("Network Error")


class ApiLimitError(PicselliaError):
    """ Raised when the user performs too many requests in a short period
    of time. """
    pass


class ProcessingError(PicselliaError):
    """Raised when an algorithmic error occurs."""
    def __init__(self):
        super().__init__("Proccessing Error")
