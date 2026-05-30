"""Load TIFF z-stacks and attach them to a Napari viewer."""

import re
from pathlib import Path

import napari
import numpy as np
import tifffile

IMAGE_DIR_NAME = "22A_E1_Wnt3"
SEGMENT_OUTPUT_DIR = Path("data") / "test segment embryo"
CELL_LABELS_DIR = Path("data") / "test_cell_labels"
EPI_VE_OUTPUT_DIR = Path("data") / "epi_ve"
EMBRYO_CUP_MASK_DIR = Path("data") / "embryo_cup_mask"
EMBRYO_CUP_SEGMENT_DIR = Path("data") / "embryo_cup_segment"
MIDDLE_Z_SLICE_COUNT = 10
_LABEL_SUFFIXES = (".tif", ".tiff", ".npy")
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


def find_segment_dir(root: Path | None = None) -> Path:
    """Return the directory with masked segment TIFFs from processing step 1."""
    root = root or project_root()
    data_dir = root / SEGMENT_OUTPUT_DIR
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Segment directory not found: {data_dir}\n"
            "Run step 1 first (run_plane_split.py) and save masked TIFFs."
        )
    return data_dir


def find_cell_labels_dir(root: Path | None = None) -> Path:
    """Return the directory with micro-SAM committed_objects label volumes."""
    root = root or project_root()
    data_dir = root / CELL_LABELS_DIR
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Cell labels directory not found: {data_dir}\n"
            "Save committed_objects from micro-SAM Annotator 3d there."
        )
    return data_dir


def discover_label_volume_path(labels_dir: Path | None = None) -> Path:
    """
    Pick a single 3D label file from ``test_cell_labels``.

    Prefers ``.npy`` / ``.tif`` at the top level; uses the largest file if several match.
    """
    labels_dir = labels_dir or find_cell_labels_dir()
    candidates: list[Path] = []
    for suffix in _LABEL_SUFFIXES:
        candidates.extend(labels_dir.glob(f"*{suffix}"))

    if not candidates:
        raise FileNotFoundError(
            f"No label volume (*.tif, *.npy) found in {labels_dir}"
        )

    return max(candidates, key=lambda p: p.stat().st_size)


def load_label_volume(path: Path | None = None, root: Path | None = None) -> np.ndarray:
    """Load a 3D instance label volume (Z, Y, X)."""
    if path is None:
        path = discover_label_volume_path(find_cell_labels_dir(root))
    if path.suffix.lower() == ".npy":
        labels = np.load(path)
    else:
        labels = tifffile.imread(path)

    labels = np.asarray(labels)
    if labels.ndim == 2:
        labels = labels[np.newaxis, ...]
    if labels.ndim != 3:
        raise ValueError(f"Expected 2D or 3D labels at {path}, got shape {labels.shape}")
    return labels.astype(np.uint32, copy=False)


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


def load_segment_channels(root: Path | None = None) -> dict[int, np.ndarray]:
    """Load masked segment channels saved by processing step 1."""
    return load_image_stack(find_segment_dir(root))


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


def save_volume_tiff(volume: np.ndarray, path: Path) -> Path:
    """Write a 3D array as a single TIFF stack."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(volume))
    return path


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
