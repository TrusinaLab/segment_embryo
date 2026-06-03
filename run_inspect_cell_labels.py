"""
Inspect fused or merged nuclei by isolating one instance label ID.

Lightweight Napari viewer (no micro-SAM): DAPI + Wnt channels and cell labels
with dock widgets to hide all cells except one ID.

Run (micro-sam-napari conda env):
    conda activate micro-sam-napari
    python run_inspect_cell_labels.py
    python run_inspect_cell_labels.py --focus-label 42

Or:
    scripts\\run_inspect_cell_labels.bat
"""

from __future__ import annotations

import argparse
from pathlib import Path

from image_io import align_label_volume_to_reference, discover_label_volume_path, load_label_volume
from image_io import load_segment_channels
from segmentation.label_focus import launch_inspect_labels_viewer, print_inspect_workflow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect cell labels: solo view for one instance ID (fused-cell QC)."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Instance label volume (default: largest file in test_cell_labels/)",
    )
    parser.add_argument(
        "--focus-label",
        type=int,
        default=None,
        metavar="ID",
        help="On launch, hide all cells except this instance label ID",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    channels = load_segment_channels()
    ref = channels[min(channels)]
    label_path = args.labels or discover_label_volume_path()
    labels = load_label_volume(label_path)
    labels = align_label_volume_to_reference(labels, ref.shape)

    print_inspect_workflow(label_path, labels.shape)
    launch_inspect_labels_viewer(
        channels,
        labels,
        focus_label_id=args.focus_label,
    )


if __name__ == "__main__":
    main()
