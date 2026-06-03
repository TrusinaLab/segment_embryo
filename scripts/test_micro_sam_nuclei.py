import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_io import DAPI_CHANNEL, WNT_CHANNEL
from segmentation.micro_sam_nuclei import (
    apply_embedding_progress_patch,
    default_embedding_path,
    embeddings_cache_complete,
    try_load_resume_labels,
)

assert WNT_CHANNEL == 1
assert DAPI_CHANNEL == 2
path = default_embedding_path()
assert path.name.endswith(".zarr")
assert "segment_dapi" in path.stem

_ = try_load_resume_labels()

apply_embedding_progress_patch()
assert embeddings_cache_complete(path) is False

print("ok")
