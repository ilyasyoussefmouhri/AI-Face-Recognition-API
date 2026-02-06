from app.core.config import ImageProcessingError
import cv2
import numpy as np
from numpy import zeros, uint8



def resize_array(
    img_array: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """
    Resize an input image array to the specified size.

    This function resizes by preserving the aspect ratio of the input image and adding black-padding to maintain the desired size.

    Args:
        img_array: The input image represented as a multidimensional numpy array.
                      The image data must not be empty.

        size: A tuple (height, width) representing the desired target size of the
                 resized image.

    Returns:
        A numpy array representing the resized image.

    Raises:
         ImageProcessingError: Raised when the input image array is empty.
    """

    if img_array.size == 0:
        raise ImageProcessingError("Cannot resize empty image")
    if img_array.ndim != 3 or img_array.shape[2] != 3:
        raise ImageProcessingError("Expected RGB image with 3 channels")


    # Calculate the aspect ratio of the image
    h = img_array.shape[0]
    w = img_array.shape[1]
    target_h = size[0]
    target_w = size[1]
    scale = min(target_w / w, target_h / h)
    new_h = int(h * scale)
    new_w = int(w * scale)
    #protect against zero-sized images
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # Calculate padding required to maintain aspect ratio
    left = target_w - new_w // 2
    top = target_h - new_h // 2

    #resize image and add padding
    resized_image = cv2.resize(
        img_array,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA)

    canvas = zeros((target_h, target_w, 3), dtype=uint8)
    canvas[top:top + new_h, left:left + new_w] = resized_image

    return canvas

def normalize_image(img):
    # normalize image
    pass