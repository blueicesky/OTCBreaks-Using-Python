
from errors import BadRequestException


def _validate_file_structure(request):
    return 'file_name' in request.files


def _validate_content_type(file):
    return file.content_type in ('application/csv')


def validate_file(request):
    """Validate request and raise an Exception if necessary"""
    if not _validate_file_structure(request):
        raise BadRequestException('Request structure is not valid.')
    elif not _validate_content_type(request.files['file_name']):
        raise BadRequestException('Invalid content type given.')