"""Load TIFF z-stacks and attach them to a Napari viewer."""

import re
from pathlib import Path

import napari
import numpy as np
import tifffile

IMAGE_DIR_NAME = "22A_E1_Wnt3"
# Match z-slice and channel in filenames, e.g. "..._z50c1..." -> groups (50, 1)
_ZC_PATTERN = re.compile(r"_z(\d+)c(\d+)", re.IGNORECASE)


def project_root() -> Path:
    """Directory containing this package (project root)."""
    return Path(__file__).resolve().parent


def find_image_dir(root: Path | None = None) -> Path:
    """Return the image directory under project root."""
    root = root or project_root()
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


def load_project_channels(root: Path | None = None) -> dict[int, np.ndarray]:
    """Load all channels from the default image folder."""
    return load_image_stack(find_image_dir(root))


def apply_channels_to_viewer(
    viewer: napari.Viewer, channels: dict[int, np.ndarray]
) -> None:
    """Add or update image layers without restarting the viewer."""
    for channel_idx in sorted(channels):
        name = f"channel {channel_idx}"
        volume = channels[channel_idx]
        if name in viewer.layers:
            viewer.layers[name].data = volume
        else:
            viewer.add_image(volume, name=name)
        print(f"{name}: shape {volume.shape}, dtype {volume.dtype}")
