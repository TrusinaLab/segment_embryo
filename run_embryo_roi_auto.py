"""
Step 1: automatic embryo ROI from DAPI + WNT fusion.

Run:
    conda run -n micro-sam-napari python run_embryo_roi_auto.py

Tune parameters in the dock widget, then click ``Build embryo ROI (auto)``.
"""

import napari

from segmentation.embryo_roi_auto import (
    add_embryo_roi_auto_widgets,
    setup_embryo_roi_auto_viewer,
)


def main() -> None:
    viewer = napari.Viewer()
    setup_embryo_roi_auto_viewer(viewer)
    add_embryo_roi_auto_widgets(viewer)
    napari.run()


if __name__ == "__main__":
    main()
