import numpy as np
import io
import cv2
from PIL import Image, ImageOps
from typing import Set
from app.core.logs import logger
from app.core.config import ImageConfig
from app.utils.exceptions import ImageProcessingError

Image.MAX_IMAGE_PIXELS = ImageConfig().max_image_pixels * 2


def decode_image(
        file,
        max_dimensions: tuple[int, int] = None,
        allowed_formats: Set[str] = None,
) -> Image.Image:
    """
    Decode an uploaded image file that has already been validated.

    Args:
        file: File-like object with async read() method
        max_dimensions: Maximum (width, height) tuple
        allowed_formats: Set of allowed PIL formats (JPEG, PNG, etc.)
        verify_format: Whether to verify PIL format matches expected

    Returns:
        PIL Image object with corrected orientation

    Raises:
        ImageProcessingError: If image cannot be decoded or exceeds limits
    """
    if max_dimensions is None:
        max_dimensions = ImageConfig().max_dimensions
    if allowed_formats is None:
        allowed_formats = ImageConfig().allowed_formats
    verify_format = ImageConfig().verify_format
    try:
        # Read the already-validated file
        bytes_data = file.read()

        if not bytes_data:
            logger.error("Empty file data received")
            raise ImageProcessingError("Empty file data received")
        logger.debug(f"Read {len(bytes_data)} bytes from file")

        # Decode image
        logger.debug("Decoding image...")
        stream = io.BytesIO(bytes_data)
        img = Image.open(stream)

        # Verify the image actually loaded
        img.load()  # Forces PIL to fully decode and validate the image

        # Verify the image format is allowed
        if verify_format and img.format not in allowed_formats:
            logger.error(
                f"Image format '{img.format}' not in allowed formats {allowed_formats}"
            )
            raise ImageProcessingError(
                f"Unsupported image format: {img.format}"
            )
        # Capture original format
        original_format = img.format

        # Apply EXIF orientation correction
        img = ImageOps.exif_transpose(img)

        # Validate dimensions
        if img.width > max_dimensions[0] or img.height > max_dimensions[1]:
            logger.error(f"Image dimensions exceed maximum {max_dimensions}")
            raise ImageProcessingError(
                f"Image dimensions ({img.width}x{img.height}) exceed "
                f"maximum {max_dimensions}"
            )



        logger.debug(
            f"Decoded image: {original_format}, {img.width}x{img.height}, "
            f"mode={img.mode}, size={len(bytes_data)} bytes"
        )

        return img

    except ImageProcessingError:
        raise
    except Image.DecompressionBombError as e:
        logger.error(f"Decompression bomb detected: {e}")
        raise ImageProcessingError("Image too large (decompression bomb)") from e
    except Exception as e:
        logger.error(f"Failed to decode image: {type(e).__name__}: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to decode image: {str(e)}") from e




def load_image(img: Image.Image) -> np.asarray:

   try:
        # Discard the Alpha channel if it exists and ensure 3-channel RGB
        img_rgb = img.convert("RGB")
        # Convert to NumPy array
        img_array = np.asarray(img_rgb, dtype="uint8")
        # Convert to BGR
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        # Validation
        if img_bgr.size == 0:
            raise ImageProcessingError("Conversion resulted in empty array")
        logger.debug(f"Converted image to array: shape={img_bgr.shape}, dtype={img_bgr.dtype}")
        return img_bgr
   except ImageProcessingError:
       raise
   except Exception as e:
       logger.error(f"Failed to convert image to array: {type(e).__name__}: {e}", exc_info=True)
       raise ImageProcessingError(f"Failed to convert image to array: {str(e)}") from e
