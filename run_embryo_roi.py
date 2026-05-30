"""
Step 1: embryo ROI annotation.

Run:
    uv run python run_embryo_roi.py

Draw polygons on layer ``embryo_outline``, then click ``Build embryo ROI``.
"""

import napari

from segmentation.embryo_roi import add_embryo_roi_widgets, setup_embryo_roi_viewer


def main() -> None:
    viewer = napari.Viewer()
    setup_embryo_roi_viewer(viewer)
    add_embryo_roi_widgets(viewer)
    napari.run()


if __name__ == "__main__":
    main()
