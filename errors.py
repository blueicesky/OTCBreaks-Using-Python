class BadRequestException(Exception):
    """When a request is malformed, we send this exception"""
    pass


class NoOutputException(Exception):
    """When we can't find any results for some reason, we send this exception"""
    pass


class NoFieldsFoundException(Exception):
    """Raise this when no fields can be detected"""
    pass