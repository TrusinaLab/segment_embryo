import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.micro_sam_nuclei import (
    DAPI_CHANNEL,
    apply_embedding_progress_patch,
    default_embedding_path,
    embeddings_cache_complete,
    try_load_resume_labels,
)

assert DAPI_CHANNEL == 1
path = default_embedding_path()
assert path.name.endswith(".zarr")
assert "segment_dapi" in path.stem

# Resume is optional — do not require labels on disk
_ = try_load_resume_labels()

apply_embedding_progress_patch()
assert embeddings_cache_complete(path) is False

print("ok")
