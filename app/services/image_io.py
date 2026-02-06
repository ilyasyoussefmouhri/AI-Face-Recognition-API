import numpy as np
import io
from PIL import Image, ImageOps
from typing import Set
from app.core.logs import logger
from app.core.config import ImageConfig, ImageProcessingError

Image.MAX_IMAGE_PIXELS = ImageConfig().max_image_pixels * 2


async def decode_image(
        file,
        max_dimensions: tuple[int, int] = None,
        allowed_formats: Set[str] = None,
        verify_format: bool = True,
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
        bytes_data = await file.read()

        if not bytes_data:
            raise ImageProcessingError("Empty file data received")

        # Decode image
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

    finally:
       file.close()




def load_image(img: Image.Image) -> np.asarray:

   try:
        # Discard the Alpha channel if it exists and ensure 3-channel RGB
        img = img.convert("RGB")
        img_array = np.asarray(img, dtype="uint8")
        # Validation
        if img_array.size == 0:
            raise ImageProcessingError("Conversion resulted in empty array")
        logger.debug(f"Converted image to array: shape={img_array.shape}, dtype={img_array.dtype}")
        return img_array
   except ImageProcessingError:
       raise
   except Exception as e:
       logger.error(f"Failed to convert image to array: {type(e).__name__}: {e}", exc_info=True)
       raise ImageProcessingError(f"Failed to convert image to array: {str(e)}") from e
