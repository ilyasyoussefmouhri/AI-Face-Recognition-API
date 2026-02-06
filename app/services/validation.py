from typing import BinaryIO
from app.core.security import signatures
from app.core.logs import logger


def validate_image(img: BinaryIO) -> bool:  # Returns format directly
    """Validate image format.

    Raises:
        ValueError: If image format is invalid or unsupported
    """
    header = img.read(16)
    img.seek(0)

    if len(header) < 4:
        raise ValueError('Image file is too small')

    if len(header) >= 12 and header.startswith(b'RIFF') and header[8:12] == b'WEBP':
        logger.info('WebP image detected')
        return True

    for format_name, sigs in signatures.items():
        for sig in sigs:
            if header.startswith(sig):
                logger.info(f'{format_name} image detected')
                return True

    raise ValueError('Unsupported image format')