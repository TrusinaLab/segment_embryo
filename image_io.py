"""Load TIFF z-stacks and attach them to a Napari viewer."""

import re
from pathlib import Path

import napari
import numpy as np
import tifffile

IMAGE_DIR_NAME = "22A_E1_Wnt3"
MIDDLE_Z_SLICE_COUNT = 10
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


def subset_middle_z_slices(
    channels: dict[int, np.ndarray], n: int = MIDDLE_Z_SLICE_COUNT
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


def load_middle_z_channels(
    root: Path | None = None, n: int = MIDDLE_Z_SLICE_COUNT
) -> dict[int, np.ndarray]:
    """Load n contiguous z-slices from the center of each channel."""
    return subset_middle_z_slices(load_project_channels(root), n=n)


def middle_z_index_list(z_size: int, n: int = MIDDLE_Z_SLICE_COUNT) -> list[int]:
    """Return absolute z indices used by ``subset_middle_z_slices``."""
    if z_size <= n:
        return list(range(z_size))
    start = (z_size - n) // 2
    return list(range(start, start + n))


def get_middle_z_index_list(
    root: Path | None = None, n: int = MIDDLE_Z_SLICE_COUNT
) -> list[int]:
    """Absolute z indices for the middle subset of the default image stack."""
    channels = load_project_channels(root)
    ref = channels[min(channels)]
    return middle_z_index_list(ref.shape[0], n=n)


def segment_tiff_filename(
    prefix: str, z_idx: int, channel_idx: int, shape_yx: tuple[int, int]
) -> str:
    """Match source naming, e.g. ``22A_E1_Wnt3_z100c1x0-512y0-512.tif``."""
    y_size, x_size = shape_yx
    return f"{prefix}_z{z_idx}c{channel_idx}x0-{x_size}y0-{y_size}.tif"


def save_channel_stack_as_tiffs(
    volume: np.ndarray,
    channel_idx: int,
    z_indices: list[int],
    output_dir: Path,
    prefix: str = IMAGE_DIR_NAME,
) -> list[Path]:
    """Write one TIFF per z-slice; return paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    y_size, x_size = volume.shape[1], volume.shape[2]

    for z_local, z_global in enumerate(z_indices):
        name = segment_tiff_filename(prefix, z_global, channel_idx, (y_size, x_size))
        path = output_dir / name
        tifffile.imwrite(path, volume[z_local])
        written.append(path)

    return written


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
