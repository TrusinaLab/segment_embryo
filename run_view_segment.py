"""
Optional QC — View saved segment TIFFs (not step 2 SAM).

Loads masked channels written by step 1 into ``data/test segment embryo/``.
For nucleus segmentation use ``run_micro_sam_nuclei.py`` (step 2).

Run (micro-sam-napari conda env):
    conda activate micro-sam-napari
    python run_view_segment.py

Or:
    scripts\\run_view_segment.bat

See ``notes/processing_steps.md`` for the full numbered pipeline.
"""

import napari
from magicgui import magicgui

from image_io import SEGMENT_OUTPUT_DIR, apply_channels_to_viewer, load_segment_channels


@magicgui(call_button="Reload from disk")
def reload_from_disk(viewer: napari.Viewer) -> None:
    """Re-read masked segment TIFFs from disk."""
    apply_channels_to_viewer(viewer, load_segment_channels())


def main() -> None:
    channels = load_segment_channels()

    viewer = napari.Viewer()
    apply_channels_to_viewer(viewer, channels)
    viewer.window.add_dock_widget(reload_from_disk)

    print(
        "View saved segment (QC):\n"
        f"  Loaded masked TIFFs from {SEGMENT_OUTPUT_DIR.as_posix()}/\n"
        "  (output of step 1 — run_plane_split.py)"
    )
    napari.run()


if __name__ == "__main__":
    main()
