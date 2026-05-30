"""
Step 2 — Segment 3D cell nuclei with micro-SAM.

Loads DAPI from the masked segment (step 1) and opens Annotator 3d. Save the
``committed_objects`` layer to ``data/test_cell_labels/`` when finished.

Run (micro-sam-napari conda env):
    conda activate micro-sam-napari
    python run_micro_sam_nuclei.py

Or:
    scripts\\run_micro_sam_nuclei.bat

See ``notes/micro_sam_nuclei_workflow.md`` and ``notes/processing_steps.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from segmentation.micro_sam_nuclei import (
    DEFAULT_MODEL,
    default_embedding_path,
    embeddings_cache_complete,
    launch_nuclei_annotator,
    load_dapi_from_segment,
    precompute_embeddings,
    print_nuclei_workflow,
    try_load_resume_labels,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: micro-SAM 3D nucleus segmentation on plane-split DAPI."
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
        help="Zarr path for embedding cache (default: data/embeddings/segment_dapi_<model>.zarr)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not load existing labels from data/test_cell_labels/",
    )
    parser.add_argument(
        "--skip-precompute",
        action="store_true",
        help="Do not precompute embeddings in the terminal before opening Napari",
    )
    parser.add_argument(
        "--precompute-only",
        action="store_true",
        help="Only precompute embeddings (tqdm in terminal), then exit",
    )
    parser.add_argument(
        "--force-precompute",
        action="store_true",
        help="Recompute embeddings even if the zarr cache looks complete",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dapi = load_dapi_from_segment()
    resume = None if args.no_resume else try_load_resume_labels()
    embedding_path = args.embedding_path or default_embedding_path(args.model)

    need_precompute = not args.skip_precompute and (
        args.force_precompute or not embeddings_cache_complete(embedding_path)
    )
    if need_precompute:
        precompute_embeddings(
            dapi,
            embedding_path,
            model_type=args.model,
            force=args.force_precompute,
        )

    if args.precompute_only:
        return

    embeddings_ready = embeddings_cache_complete(embedding_path)
    print_nuclei_workflow(
        dapi,
        embedding_path,
        resumed=resume is not None,
        embeddings_ready=embeddings_ready,
    )
    launch_nuclei_annotator(
        dapi,
        embedding_path=embedding_path,
        segmentation_result=resume,
        model_type=args.model,
    )


if __name__ == "__main__":
    main()
