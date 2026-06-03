"""
Step 2 — Segment 3D cell nuclei with micro-SAM.

Loads DAPI from the masked segment (step 1) and opens Annotator 3d. Save the
``committed_objects`` layer to ``data/test_cell_labels/`` when finished.

Embeddings are **not** computed on launch. In Napari: set z range (optional),
then Annotator 3d → **Compute Embeddings**.

Run from Python::

    from run_micro_sam_nuclei import run_nuclei_segmentation

    run_nuclei_segmentation()
    run_nuclei_segmentation(z_range="50-80")
    run_nuclei_segmentation(bottom_z_third=True)

Or from a terminal::

    python run_micro_sam_nuclei.py

See ``notes/micro_sam_nuclei_workflow.md`` and ``notes/processing_steps.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

from image_io import (
    align_labels_to_volume_start,
    align_labels_to_z_local_indices,
    discover_z_index_list,
    find_segment_dir,
    load_segment_channels,
    parse_z_range_spec,
    subset_channels_by_absolute_z_range,
    project_root,
    segment_z_index_summary,
)
from image_io import DAPI_CHANNEL
from segmentation.micro_sam_nuclei import (
    DEFAULT_MODEL,
    bottom_third_z_tag,
    default_embedding_path,
    embeddings_cache_complete,
    launch_nuclei_annotator,
    print_nuclei_workflow,
    allocate_embedding_cache,
    resolve_embedding_cache_path,
    try_load_resume_labels,
    z_range_tag,
)

ZRangeArg = Union[str, tuple[int, int], None]


def _initial_z_bounds(
    z_all: list[int],
    *,
    z_range: ZRangeArg = None,
    bottom_z_third: bool = False,
) -> tuple[int, int, str | None, list[int] | None]:
    """Return z_min, z_max, embedding z_tag, and local indices for the first load."""
    if bottom_z_third:
        from image_io import bottom_z_slice_count

        n = bottom_z_slice_count(len(z_all))
        return z_all[0], z_all[n - 1], bottom_third_z_tag(), list(range(n))

    if z_range is not None:
        if isinstance(z_range, tuple):
            z_min, z_max = z_range
        else:
            z_min, z_max = parse_z_range_spec(z_range)
        return z_min, z_max, z_range_tag(z_min, z_max), None

    return z_all[0], z_all[-1], None, None


def _load_channels_for_z(
    full_channels: dict[int, object],
    z_all: list[int],
    z_min: int,
    z_max: int,
    local_indices: list[int] | None,
) -> tuple[dict[int, object], list[int] | None, int | None]:
    if local_indices is not None:
        selection = subset_channels_by_absolute_z_range(
            full_channels, z_all, z_min, z_max
        )
        return selection.channels, selection.local_indices, len(z_all)

    if z_min != z_all[0] or z_max != z_all[-1]:
        selection = subset_channels_by_absolute_z_range(
            full_channels, z_all, z_min, z_max
        )
        return selection.channels, selection.local_indices, len(z_all)

    return full_channels, None, None


def run_nuclei_segmentation(
    *,
    z_range: ZRangeArg = None,
    bottom_z_third: bool = False,
    resume: bool = False,
    model: str = DEFAULT_MODEL,
    embedding_path: Path | None = None,
) -> None:
    """
    Open micro-SAM Annotator 3d on plane-split DAPI (step 2).

    Parameters
    ----------
    z_range
        Inclusive absolute z indices from segment TIFF names, e.g. ``"50-80"``
        or ``(50, 80)``. Adjust later in the **Z range (segmentation)** dock.
    bottom_z_third
        If True, initial load uses the lowest-Z third of the stack.
    resume
        Load existing labels from ``data/test_cell_labels/`` when shapes allow.
    model
        SAM model name (default ``vit_b_lm``).
    embedding_path
        Optional fixed zarr cache path; otherwise derived from model and z range.
    """
    if z_range is not None and bottom_z_third:
        raise ValueError("Pass only one of z_range or bottom_z_third.")

    full_channels = load_segment_channels()
    z_all = discover_z_index_list(find_segment_dir())
    z_min, z_max, z_tag, local_indices = _initial_z_bounds(
        z_all, z_range=z_range, bottom_z_third=bottom_z_third
    )
    channels, local_indices, full_z_count = _load_channels_for_z(
        full_channels, z_all, z_min, z_max, local_indices
    )

    if DAPI_CHANNEL not in channels:
        raise KeyError(
            f"Channel {DAPI_CHANNEL} (DAPI) not found in segment stack. "
            f"Available: {sorted(channels)}"
        )
    dapi = channels[DAPI_CHANNEL]

    resume_full = try_load_resume_labels() if resume else None
    resume = resume_full
    if resume is not None:
        if local_indices is not None:
            resume = align_labels_to_z_local_indices(
                resume,
                dapi.shape,
                local_indices,
                full_z_count=full_z_count,
            )
        else:
            resume = align_labels_to_volume_start(resume, dapi.shape)

    cache = allocate_embedding_cache(
        resolve_embedding_cache_path(embedding_path, model_type=model, z_tag=z_tag)
    )
    embeddings_ready = embeddings_cache_complete(cache)
    print_nuclei_workflow(
        dapi,
        cache,
        resumed=resume is not None,
        embeddings_ready=embeddings_ready,
    )
    launch_nuclei_annotator(
        segment_channels=channels,
        embedding_path=cache,
        segmentation_result=resume,
        model_type=model,
        full_segment_channels=full_channels,
        absolute_z_all=z_all,
        initial_z_min=z_min,
        initial_z_max=z_max,
        resume_labels_full=resume_full,
        embedding_path_override=embedding_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: micro-SAM 3D nucleus segmentation on plane-split DAPI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "From Python:  from run_micro_sam_nuclei import run_nuclei_segmentation; "
            "run_nuclei_segmentation()\n"
            "Z range can also be set in the Napari dock after launch.\n"
        ),
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
        help="Zarr path for embedding cache (default: per z-range under data/embeddings/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing labels from data/test_cell_labels/",
    )
    parser.add_argument(
        "--list-z",
        action="store_true",
        help="Print absolute z indices in the segment stack and exit",
    )
    z_group = parser.add_mutually_exclusive_group()
    z_group.add_argument(
        "--bottom-z-third",
        action="store_true",
        help="Initial load: lowest-Z one-third",
    )
    z_group.add_argument(
        "--z-range",
        metavar="ZMIN-ZMAX",
        help="Initial load: inclusive absolute z (e.g. 50-80)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_z:
        z_all = discover_z_index_list(find_segment_dir(project_root()))
        print(f"Segment stack ({segment_z_index_summary()}):")
        print(" ".join(str(z) for z in z_all))
        return

    run_nuclei_segmentation(
        z_range=args.z_range,
        bottom_z_third=args.bottom_z_third,
        resume=args.resume,
        model=args.model,
        embedding_path=args.embedding_path,
    )


if __name__ == "__main__":
    main()
