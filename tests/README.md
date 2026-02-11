# InsightFace Embedder – Pytest Guide

This test suite currently focuses on the InsightFace embedder and its schema/exception behavior. The API layer is not covered yet; tests live in `tests/test_embedder.py` and use fixtures from `tests/conftest.py`.

## Test-Related Layout
```
app/
├── models/insightface.py    # InsightFaceEmbedder, Device, embed() logic
├── schemas/detection.py     # FaceEmbedding Pydantic model
└── core/config.py           # NoFaceDetectedError, MultipleFacesDetectedError, Device
tests/
├── conftest.py              # fixtures: mock faces, synthetic images, embedder instances
├── test_embedder.py         # unit, integration, edge-case, parametrized tests
└── run_tests.sh             # helper script (interactive)
```

## Dependencies
- Runtime: `pip install -r requirements.txt` (includes `insightface`, `numpy`, `pydantic`, etc.).
- Tests: `pip install pytest` (optional: `pytest-cov`, `pytest-benchmark`).
- `opencv-python` is needed because fixtures use `cv2` (often installed as a dependency of InsightFace).
- GPU tests: `onnxruntime` with CUDA (if you want real GPU runs).

## What the Tests Cover (Current Logic)
- `app.models.insightface.InsightFaceEmbedder.embed`:
  - Raises `NoFaceDetectedError` on zero faces.
  - Raises `MultipleFacesDetectedError` on multiple faces.
  - Returns a `FaceEmbedding` with a 512-dim `np.ndarray` and a `[0.0, 1.0]` detection score.
- `app.schemas.detection.FaceEmbedding`:
  - Enforces detection score bounds and required fields.
- `app.core.config` exceptions:
  - Error messages and `num_faces` attribute on `MultipleFacesDetectedError`.
- Integration behavior:
  - Real InsightFace model initialization (marked `slow`/`integration`).
- Optional benchmark:
  - `pytest-benchmark` test in `TestPerformance`.

## Fixtures You Can Reuse
From `tests/conftest.py`:
- `mock_embedder`, `mock_embedder_no_face`, `mock_embedder_multi_face`
- `single_face_image`, `no_face_image`, `multi_face_image`, `invalid_format_image`
- `embedder_cpu`, `embedder_gpu`
- `gpu_available`
- `test_data_dir` (auto-creates `tests/data/`)

## Running Tests
- All tests: `pytest`
- Skip slow tests: `pytest -m "not slow"`
- Only integration tests: `pytest -m "integration"`
- Specific class: `pytest tests/test_embedder.py::TestEmbedderUnitTests`
- Coverage (package is `app`, not `src`):  
  `pytest --cov=app --cov-report=term-missing --cov-report=html`

Note: `pytest.ini` and `tests/run_tests.sh` still point coverage at `src`; override with `--cov=app` as shown above.

## Model Download Note
Integration tests initialize InsightFace and will download models on first run (~200MB). If you want a fast loop, stick to unit tests with the mock embedder.
