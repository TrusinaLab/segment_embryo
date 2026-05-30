"""
Step 1 — Slice embryo from trophectoderm.

Loads the full z-stack from ``22A_E1_Wnt3/``. Draw divider lines on several
z-slices, build a plane or interpolated surface split, and save the kept region
to ``data/test segment embryo/`` for every z.

Run (micro-sam-napari conda env):
    conda activate micro-sam-napari
    python run_plane_split.py

Or:
    scripts\\run_plane_split.bat

See ``notes/processing_steps.md`` for the numbered pipeline and
``notes/plane_split_napari_workflow.md`` for Napari instructions.
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
