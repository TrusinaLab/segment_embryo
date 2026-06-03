"""
Split a fused instance label using divider lines on multiple z-slices.

Draw lines on ``cell_split_divider_lines``, preview the cut, apply two new IDs,
and save the updated label volume.

Run (micro-sam-napari conda env):
    conda activate micro-sam-napari
    python run_split_fused_cell.py
    python run_split_fused_cell.py --target-label 42

Or:
    scripts\\run_split_fused_cell.bat
"""

from __future__ import annotations

import argparse
from pathlib import Path

from image_io import align_label_volume_to_reference, discover_label_volume_path, load_label_volume
from image_io import load_segment_channels
from segmentation.cell_split import launch_cell_split_viewer, print_cell_split_workflow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split one fused cell ID with 3D-interpolated divider lines."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Instance label volume (default: largest file in test_cell_labels/)",
    )
    parser.add_argument(
        "--target-label",
        type=int,
        default=None,
        metavar="ID",
        help="Fused cell ID to isolate on launch",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    channels = load_segment_channels()
    ref = channels[min(channels)]
    label_path = args.labels or discover_label_volume_path()
    labels = load_label_volume(label_path)
    labels = align_label_volume_to_reference(labels, ref.shape)

    print_cell_split_workflow(label_path)
    launch_cell_split_viewer(
        channels,
        labels,
        target_label_id=args.target_label,
    )


if __name__ == "__main__":
    main()
