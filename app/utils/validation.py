from typing import BinaryIO, Optional, Tuple
from app.core.security import signatures
from app.core.logs import logger

def validate_image(img: BinaryIO) -> Tuple[bool, Optional[str]]:
    header = img.read(16)
    img.seek(0)
    # Edge case for empty/too small file
    if len(header) < 4:  # Minimum for any valid image
        logger.error('Image is too small')
        raise ValueError('Image is too small')
        return False, None

    # WEBP requires checking TWO locations
    if len(header) >= 12 and header.startswith(b'RIFF') and header[8:12] == b'WEBP':
        logger.info('WebP image detected')
        return True, 'webp'


    for format_name, sigs in signatures.items():
        for sig in sigs:
            if header.startswith(sig):
                logger.info(f'{format_name} image detected')
                return True, format_name
    logger.error('Unsupported image format')
    raise ValueError('Unsupported image format')
    return False, None

