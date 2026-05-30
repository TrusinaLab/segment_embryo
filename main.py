"""Load 22A_E1_Wnt3 TIFF z-stacks into napari."""

import re
from pathlib import Path

import napari
import numpy as np
import tifffile
from magicgui import magicgui

IMAGE_DIR_NAME = "22A_E1_Wnt3"
TEST_Z_SLICE_COUNT = 10
# Match z-slice and channel in filenames, e.g. "..._z50c1..." -> groups (50, 1)
_ZC_PATTERN = re.compile(r"_z(\d+)c(\d+)", re.IGNORECASE)


def find_image_dir(root: Path) -> Path:
    """Return the image directory under project root."""
    data_dir = root / IMAGE_DIR_NAME
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {data_dir}")
    return data_dir


def load_image_stack(data_dir: Path) -> dict[int, np.ndarray]:
    """Stack TIFFs into volumes keyed by channel index (Z, Y, X)."""
    slices_by_channel: dict[int, dict[int, np.ndarray]] = {}

    for path in sorted(data_dir.glob("*.tif")):
        match = _ZC_PATTERN.search(path.stem)
        if not match:
            continue
        z_idx, channel_idx = int(match.group(1)), int(match.group(2))
        img = tifffile.imread(path)
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.mean(axis=-1)
        slices_by_channel.setdefault(channel_idx, {})[z_idx] = np.asarray(img)

    volumes: dict[int, np.ndarray] = {}
    for channel_idx, z_slices in slices_by_channel.items():
        z_order = sorted(z_slices.keys())
        volumes[channel_idx] = np.stack([z_slices[z] for z in z_order], axis=0)

    if not volumes:
        raise ValueError(f"No TIFF files matched pattern in {data_dir}")

    return volumes


def subset_middle_z_slices(
    channels: dict[int, np.ndarray], n: int = TEST_Z_SLICE_COUNT
) -> dict[int, np.ndarray]:
    """Return only n contiguous Z planes from the center of each channel volume."""
    if n <= 0:
        raise ValueError("n must be positive")

    subset: dict[int, np.ndarray] = {}
    for channel_idx, volume in channels.items():
        z_size = volume.shape[0]
        if z_size <= n:
            subset[channel_idx] = volume
            print(f"channel {channel_idx}: using all {z_size} z-slices (stack smaller than {n})")
            continue

        start = (z_size - n) // 2
        end = start + n
        subset[channel_idx] = volume[start:end]
        print(f"channel {channel_idx}: z indices {start}-{end - 1} of {z_size - 1}")

    return subset


def load_test_channels(data_dir: Path) -> dict[int, np.ndarray]:
    """Load a middle subset of z-slices for faster segmentation tests."""
    return subset_middle_z_slices(load_image_stack(data_dir))


def apply_channels_to_viewer(viewer: napari.Viewer, channels: dict[int, np.ndarray]) -> None:
    """Add or update image layers without restarting the viewer."""
    for channel_idx in sorted(channels):
        name = f"channel {channel_idx}"
        volume = channels[channel_idx]
        if name in viewer.layers:
            viewer.layers[name].data = volume
        else:
            viewer.add_image(volume, name=name)
        print(f"{name}: shape {volume.shape}, dtype {volume.dtype}")


@magicgui(call_button="Reload from disk")
def reload_from_disk(viewer: napari.Viewer) -> None:
    """Re-read TIFFs from 22A_E1_Wnt3 and refresh open layers."""
    data_dir = find_image_dir(Path(__file__).resolve().parent)
    apply_channels_to_viewer(viewer, load_test_channels(data_dir))


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_dir = find_image_dir(project_root)
    channels = load_test_channels(data_dir)

    viewer = napari.Viewer()
    apply_channels_to_viewer(viewer, channels)
    viewer.window.add_dock_widget(reload_from_disk)

    napari.run()


if __name__ == "__main__":
    main()
