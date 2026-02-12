# Custom Exceptions
class NoFaceDetectedError(Exception):
    """Raised when no face is detected in the image."""
    pass


class MultipleFacesDetectedError(Exception):
    """Raised when multiple faces are detected in the image."""
    def __init__(self, num_faces: int):
        self.num_faces = num_faces
        super().__init__(f"Expected 1 face, but detected {num_faces} faces")

class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass
