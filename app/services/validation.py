from typing import BinaryIO
from app.core.logs import logger
from app.core.config import settings

def validate_image(img: BinaryIO) -> bool:  # Returns format directly
    """Validate image format.

    Raises:
        ValueError: If image format is invalid or unsupported
    """
    logger.debug('Reading image format')
    header = img.read(16)
    img.seek(0)

    if len(header) < 4:
        logger.error('Image file is too small')
        raise ValueError('Image file is too small')

    if len(header) >= 12 and header.startswith(b'RIFF') and header[8:12] == b'WEBP':
        logger.debug('WebP image detected')
        return True

    for format_name, sigs in settings.SIGNATURES.items():
        for sig in sigs:
            if header.startswith(sig):
                logger.debug(f'{format_name} image detected')
                return True
    logger.error('Unsupported image format')
    raise ValueError('Unsupported image format')