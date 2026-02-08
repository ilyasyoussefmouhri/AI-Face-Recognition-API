"""
InsightFace Embedder Module
Provides face detection and embedding extraction using InsightFace.
"""
import numpy as np
from insightface.app import FaceAnalysis
from app.schemas.detection import FaceEmbedding
from app.core.config import NoFaceDetectedError, MultipleFacesDetectedError, Device
from app.core.logs import logger



class InsightFaceEmbedder:
    def __init__(self, model_name: str = 'buffalo_l', device: Device = Device.CPU):
        """
        Initialize the InsightFace embedder.

        Args:
            model_name: InsightFace model name (e.g., 'buffalo_l', 'arcface_r100_v1', etc.)
            device: Device to run inference on (Device.CPU or Device.GPU)
        """

        # Set providers based on device
        if device == Device.GPU:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0  # GPU context
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1  # CPU context
        self.model_name = model_name
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id)
        self.device = device
    def embed(self, img_array: np.ndarray) -> FaceEmbedding:
        """
        Extract face embedding from a preprocessed image.

        Args:
            img_array: Preprocessed image (H, W, 3), BGR, uint8, [0, 255]

        Returns:
            FaceEmbedding containing the embedding vector and detection score

        Raises:
            ValueError: If input image is not in the correct format
            NoFaceDetectedError: When no face is found in the image
            MultipleFacesDetectedError: When multiple faces are detected
        """
        # Input validation
        if not isinstance(img_array, np.ndarray):
            logger.error("Input must be a numpy array")
            raise ValueError("Input must be a numpy array")
        if img_array.ndim != 3 or img_array.shape[2] != 3:
            logger.error(f"Input image must have 3 channels (H, W, 3), got shape {img_array.shape}")
            raise ValueError(f"Input image must have 3 channels (H, W, 3), got shape {img_array.shape}")

        # Detect faces and extract embedding
        faces = self.app.get(img_array)
        logger.info(f"Detecting faces in image with InsightFace {self.model_name} model...")

        # Handle edge cases: no face or multiple faces = error
        if len(faces) == 0:
            logger.error("No face detected in the provided image")
            raise NoFaceDetectedError("No face detected in the provided image")

        if len(faces) > 1:
            logger.error(f"Multiple faces detected in the provided image: {len(faces)}")
            raise MultipleFacesDetectedError(num_faces=len(faces))

        # Extract the single face
        face = faces[0]
        logger.info(f"Face detected with embedding of shape {face.embedding.shape}")
        return FaceEmbedding(
            embedding=face.embedding,
            detection_score=float(face.det_score)
        )