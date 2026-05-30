"""
Plane-split segmentation from sparse divider lines.

Run (micro-sam-napari conda env):
    conda activate micro-sam-napari
    python run_plane_split.py

Or:
    scripts\\run_plane_split.bat

Draw lines on layer ``divider_lines`` on several z-slices, then click
``Build plane split``.
"""

import napari

from segmentation.plane_split import add_plane_split_widgets, setup_plane_split_viewer


def main() -> None:
    viewer = napari.Viewer()
    setup_plane_split_viewer(viewer)
    add_plane_split_widgets(viewer)
    napari.run()


if __name__ == "__main__":
    main()
