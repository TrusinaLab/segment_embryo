"""
Re-segment fused nuclei with micro-SAM while viewing existing labels.

Opens Annotator 3d on plane-split DAPI with:
  - ``reference_labels_qc`` — frozen contour overlay (spot fused IDs)
  - ``committed_objects`` — same labels, editable via SAM commit / save

Embeddings are computed in Napari (Annotator 3d → **Compute Embeddings**).

Run (micro-sam-napari conda env):
    conda activate micro-sam-napari
    python run_micro_sam_resegment.py

Or:
    scripts\\run_micro_sam_resegment.bat

Requires step 1 segment TIFFs and a label volume in ``data/test_cell_labels/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from image_io import (
    align_label_volume_to_reference,
    discover_label_volume_path,
    load_label_volume,
)
from segmentation.micro_sam_nuclei import (
    DEFAULT_MODEL,
    default_embedding_path,
    embeddings_cache_complete,
    launch_resegment_annotator,
    load_dapi_from_segment,
    print_resegment_workflow,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="micro-SAM re-segment: DAPI + reference labels overlay for fused-cell QC."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Instance label volume (default: largest file in test_cell_labels/)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SAM model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=None,
        help="Zarr embedding cache (default: data/embeddings/segment_dapi_<model>.zarr)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dapi = load_dapi_from_segment()
    label_path = args.labels or discover_label_volume_path()
    labels = load_label_volume(label_path)
    labels = align_label_volume_to_reference(labels, dapi.shape)

    embedding_path = args.embedding_path or default_embedding_path(args.model)
    embeddings_ready = embeddings_cache_complete(embedding_path)
    print_resegment_workflow(
        dapi,
        embedding_path,
        label_path,
        embeddings_ready=embeddings_ready,
    )
    launch_resegment_annotator(
        dapi,
        labels,
        embedding_path=embedding_path,
        model_type=args.model,
    )


if __name__ == "__main__":
    main()
