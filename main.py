"""Load 22A_E1_Wnt3 TIFF z-stacks into napari (middle z-subset)."""

import napari
from magicgui import magicgui

from image_io import IMAGE_DIR_NAME, apply_channels_to_viewer, load_middle_z_channels


@magicgui(call_button="Reload from disk")
def reload_from_disk(viewer: napari.Viewer) -> None:
    """Re-read TIFFs from 22A_E1_Wnt3 and refresh open layers."""
    apply_channels_to_viewer(viewer, load_middle_z_channels())


def main() -> None:
    channels = load_middle_z_channels()

    viewer = napari.Viewer()
    apply_channels_to_viewer(viewer, channels)
    viewer.window.add_dock_widget(reload_from_disk)

    print(f"Loaded middle z-subset from {IMAGE_DIR_NAME}/")
    napari.run()


if __name__ == "__main__":
    main()
