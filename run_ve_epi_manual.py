"""
Manual VE / EPI labeling on segmented nuclei.

Loads the plane-split segment (DAPI + Wnt) and instance labels from
``data/test_cell_labels/``. Click cells to mark VE; when finished, assign EPI
to all remaining cells.

Run (micro-sam-napari conda env):
    conda activate micro-sam-napari
    python run_ve_epi_manual.py

Or:
    scripts\\run_ve_epi_manual.bat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import napari

from image_io import CELL_LABELS_DIR, EPI_VE_OUTPUT_DIR, discover_label_volume_path, project_root
from segmentation.ve_epi_manual import add_ve_epi_manual_widgets, setup_ve_epi_manual_viewer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually assign VE / EPI classes to segmented cells."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to instance label volume (default: largest file in test_cell_labels/)",
    )
    parser.add_argument(
        "--resume-csv",
        type=Path,
        default=None,
        help=f"Resume assignments (default: {EPI_VE_OUTPUT_DIR}/ve_epi_manual.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    label_path = args.labels
    if label_path is None:
        label_path = discover_label_volume_path()

    viewer, state, cell_layer = setup_ve_epi_manual_viewer(
        label_path=label_path,
        resume_csv=args.resume_csv,
    )
    add_ve_epi_manual_widgets(viewer, state, cell_layer)

    print(
        f"\nOutput (after Save): {EPI_VE_OUTPUT_DIR.as_posix()}/\n"
        f"  ve_epi_manual.csv — per-cell manual_class (1=VE, 2=EPI, 3=likely_fused_cells)\n"
        f"  ve_epi_manual_labels.tif — class label volume\n"
        f"Labels input: {label_path}\n"
        f"Label dir: {CELL_LABELS_DIR.as_posix()}/\n"
    )
    napari.run()


if __name__ == "__main__":
    main()
