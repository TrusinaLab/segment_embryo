"""Load TIFF z-stacks and attach them to a Napari viewer."""

import re
from dataclasses import dataclass
from pathlib import Path

import napari
import numpy as np
import tifffile

IMAGE_DIR_NAME = "22A_E1_Wnt3"
# TIFF filename channel index (e.g. ..._z50c1...): 1 = Wnt, 2 = DAPI
WNT_CHANNEL = 1
DAPI_CHANNEL = 2
WNT_LAYER_NAME = f"channel {WNT_CHANNEL}"
DAPI_LAYER_NAME = f"channel {DAPI_CHANNEL}"
SEGMENT_OUTPUT_DIR = Path("data") / "test segment embryo"
CELL_LABELS_DIR = Path("data") / "test_cell_labels"
EPI_VE_OUTPUT_DIR = Path("data") / "epi_ve"
EMBRYO_CUP_MASK_DIR = Path("data") / "embryo_cup_mask"
EMBRYO_CUP_SEGMENT_DIR = Path("data") / "embryo_cup_segment"
MIDDLE_Z_SLICE_COUNT = 10
# Fraction of the stack (from lowest Z) used for quick segmentation tests.
BOTTOM_Z_FRACTION = 1.0 / 3.0
# Napari layer scale (Z, Y, X) for anisotropic voxels (Z step larger than XY).
LAYER_SCALE_ZYX = (4.0, 1.0, 1.0)
# Usable z-range in 22A_E1_Wnt3 (empty slices below/above were removed from disk).
STACK_Z_MIN = 18
STACK_Z_MAX = 112
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


def bottom_z_slice_count(z_size: int, fraction: float = BOTTOM_Z_FRACTION) -> int:
    """Number of Z planes in the bottom ``fraction`` of a stack (at least 1)."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    return max(1, int(z_size * fraction))


def subset_bottom_z_fraction(
    channels: dict[int, np.ndarray], fraction: float = BOTTOM_Z_FRACTION
) -> dict[int, np.ndarray]:
    """Keep the lowest-Z ``fraction`` of each channel volume (axis 0)."""
    subset: dict[int, np.ndarray] = {}
    for channel_idx, volume in channels.items():
        z_size = volume.shape[0]
        n = bottom_z_slice_count(z_size, fraction=fraction)
        subset[channel_idx] = volume[:n]
        print(
            f"channel {channel_idx}: bottom {n}/{z_size} z-slices "
            f"(local indices 0–{n - 1})"
        )
    return subset


def bottom_z_index_list(
    z_size: int,
    fraction: float = BOTTOM_Z_FRACTION,
    *,
    absolute_z: list[int] | None = None,
) -> list[int]:
    """Local or absolute z indices for the bottom ``fraction`` of a stack."""
    n = bottom_z_slice_count(z_size, fraction=fraction)
    if absolute_z is not None:
        if len(absolute_z) < n:
            raise ValueError(
                f"absolute_z has {len(absolute_z)} entries but need {n} for fraction {fraction}"
            )
        return absolute_z[:n]
    return list(range(n))


def load_segment_channels_bottom_third(
    root: Path | None = None, fraction: float = BOTTOM_Z_FRACTION
) -> dict[int, np.ndarray]:
    """Plane-split segment channels, lowest-Z ``fraction`` only (for test runs)."""
    return subset_bottom_z_fraction(load_segment_channels(root), fraction=fraction)


def get_segment_bottom_z_index_list(
    root: Path | None = None, fraction: float = BOTTOM_Z_FRACTION
) -> list[int]:
    """Absolute z indices for the bottom ``fraction`` of the segment stack."""
    z_all = discover_z_index_list(find_segment_dir(root))
    n = bottom_z_slice_count(len(z_all), fraction=fraction)
    return z_all[:n]


def parse_z_range_spec(spec: str) -> tuple[int, int]:
    """
    Parse an inclusive absolute z range from the CLI (e.g. ``50-80`` or ``50:80``).
    """
    text = spec.strip()
    for sep in ("-", ":", ".."):
        if sep in text:
            left, right = text.split(sep, 1)
            z_min, z_max = int(left.strip()), int(right.strip())
            if z_min > z_max:
                raise ValueError(
                    f"Invalid z range '{spec}': start ({z_min}) must be <= end ({z_max})."
                )
            return z_min, z_max
    raise ValueError(
        f"Invalid z range '{spec}'. Use inclusive absolute indices, e.g. 50-80."
    )


def segment_z_index_summary(root: Path | None = None) -> str:
    """Human-readable list of absolute z indices in the segment folder."""
    z_all = discover_z_index_list(find_segment_dir(root))
    if not z_all:
        return "no z indices found in segment TIFFs"
    return f"z {z_all[0]}–{z_all[-1]} ({len(z_all)} slices)"


@dataclass(frozen=True)
class ZRangeSelection:
    """Channels cropped to a z subset plus index bookkeeping."""

    channels: dict[int, np.ndarray]
    absolute_z: list[int]
    local_indices: list[int]


def subset_channels_by_absolute_z_range(
    channels: dict[int, np.ndarray],
    absolute_z: list[int],
    z_min: int,
    z_max: int,
) -> ZRangeSelection:
    """Keep planes whose TIFF z index lies in ``[z_min, z_max]`` (inclusive)."""
    if z_min > z_max:
        raise ValueError(f"z_min ({z_min}) must be <= z_max ({z_max})")

    selected = [z for z in absolute_z if z_min <= z <= z_max]
    if not selected:
        raise ValueError(
            f"No segment slices with z in [{z_min}, {z_max}]. "
            f"Segment stack has {segment_z_index_summary()}."
        )

    z_to_local = {z: i for i, z in enumerate(absolute_z)}
    local_indices = [z_to_local[z] for z in selected]
    subset: dict[int, np.ndarray] = {}
    for channel_idx, volume in channels.items():
        if volume.shape[0] != len(absolute_z):
            raise ValueError(
                f"channel {channel_idx} has {volume.shape[0]} z-planes but "
                f"expected {len(absolute_z)} from TIFF names."
            )
        subset[channel_idx] = volume[local_indices]
        print(
            f"channel {channel_idx}: absolute z {selected[0]}–{selected[-1]} "
            f"({len(selected)} slices)"
        )

    return ZRangeSelection(
        channels=subset,
        absolute_z=selected,
        local_indices=local_indices,
    )


def load_segment_channels_z_range(
    z_min: int,
    z_max: int,
    root: Path | None = None,
) -> ZRangeSelection:
    """Load plane-split segment channels limited to absolute z ``z_min``–``z_max``."""
    z_all = discover_z_index_list(find_segment_dir(root))
    full = load_segment_channels(root)
    return subset_channels_by_absolute_z_range(full, z_all, z_min, z_max)


def align_labels_to_z_local_indices(
    labels: np.ndarray,
    reference_shape: tuple[int, ...],
    local_indices: list[int],
    *,
    full_z_count: int | None = None,
) -> np.ndarray:
    """
    Crop labels to the same local z indices as a channel subset.

    ``full_z_count`` is the z depth before subsetting (defaults to ``labels.shape[0]``).
    """
    if labels.shape == reference_shape:
        return labels

    z_full = full_z_count if full_z_count is not None else labels.shape[0]
    if (
        len(labels.shape) == 3
        and len(reference_shape) == 3
        and labels.shape[1:] == reference_shape[1:]
        and labels.shape[0] == z_full
        and len(local_indices) == reference_shape[0]
    ):
        print(
            f"Aligning labels: local z indices {local_indices[0]}–{local_indices[-1]} "
            f"({labels.shape[0]} → {reference_shape[0]} slices)."
        )
        return labels[local_indices]

    return align_label_volume_to_reference(labels, reference_shape)


def align_labels_to_volume_start(
    labels: np.ndarray, reference_shape: tuple[int, ...]
) -> np.ndarray:
    """
    Match labels to a reference volume, taking the first Z planes when shapes differ.

    Used when the reference is a bottom-Z subset of a longer label stack.
    Falls back to :func:`align_label_volume_to_reference` for full-stack alignment.
    """
    if labels.shape == reference_shape:
        return labels
    if (
        len(labels.shape) == 3
        and len(reference_shape) == 3
        and labels.shape[1:] == reference_shape[1:]
        and labels.shape[0] >= reference_shape[0]
    ):
        z_ref = reference_shape[0]
        print(
            f"Aligning labels: first {z_ref} z-slices "
            f"({labels.shape[0]} → {z_ref}) to match reference volume."
        )
        return labels[:z_ref]
    return align_label_volume_to_reference(labels, reference_shape)


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


def discover_z_index_list(data_dir: Path | None = None) -> list[int]:
    """
    Sorted absolute z indices from TIFF filenames in the image folder.

    Matches the z ordering used by ``load_image_stack``.
    """
    data_dir = data_dir or find_image_dir()
    z_indices: set[int] = set()
    for path in data_dir.glob("*.tif"):
        match = _ZC_PATTERN.search(path.stem)
        if match:
            z_indices.add(int(match.group(1)))

    if not z_indices:
        raise ValueError(f"No TIFF files matched pattern in {data_dir}")

    return sorted(z_indices)


def get_stack_z_index_list(root: Path | None = None) -> list[int]:
    """Absolute z indices for every slice in the default image stack."""
    return discover_z_index_list(find_image_dir(root))


def align_label_volume_to_reference(
    labels: np.ndarray,
    reference_shape: tuple[int, ...],
    *,
    z_min: int = STACK_Z_MIN,
    z_max: int = STACK_Z_MAX,
) -> np.ndarray:
    """
    Crop or pass through a label volume so its shape matches a reference stack.

    When labels were saved on the full z-stack (e.g. 133 slices, z 0–132) but
    segment TIFFs were trimmed to ``z_min``–``z_max`` (95 slices), take
    ``labels[z_min : z_max + 1]``.
    """
    if labels.shape == reference_shape:
        return labels

    if len(labels.shape) != 3 or len(reference_shape) != 3:
        raise ValueError(
            f"Label shape {labels.shape} does not match reference {reference_shape}."
        )

    z_labels, y_labels, x_labels = labels.shape
    z_ref, y_ref, x_ref = reference_shape
    if (y_labels, x_labels) != (y_ref, x_ref):
        raise ValueError(
            f"Label YX {labels.shape[1:]} does not match reference {reference_shape[1:]}."
        )

    expected_z = z_max - z_min + 1
    if z_ref == expected_z and z_labels > z_ref:
        if z_labels == z_max + 1 or z_labels >= z_max + 1:
            end = z_min + z_ref
            if end <= z_labels:
                print(
                    f"Aligning labels: subset z indices {z_min}–{z_max} "
                    f"({labels.shape[0]} → {z_ref} slices) to match segment stack."
                )
                return labels[z_min:end]

    raise ValueError(
        f"Label shape {labels.shape} does not match reference {reference_shape}. "
        f"Re-save labels on the same z-stack as the segment (expected {z_ref} z-slices "
        f"for z {z_min}–{z_max}), or use a label file with matching shape."
    )


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


def layer_scale_for_shape(shape: tuple[int, ...]) -> tuple[float, ...]:
    """Napari scale tuple matching array dimensions (Z, Y, X) or (Y, X)."""
    ndim = len(shape)
    if ndim == 3:
        return LAYER_SCALE_ZYX
    if ndim == 2:
        return LAYER_SCALE_ZYX[1], LAYER_SCALE_ZYX[2]
    raise ValueError(f"Unsupported volume ndim {ndim}: shape {shape}")


def apply_layer_scale(layer: napari.layers.Layer) -> None:
    """Set layer.scale from LAYER_SCALE_ZYX for the layer's data shape."""
    layer.scale = layer_scale_for_shape(layer.data.shape)


def viewer_add_image(
    viewer: napari.Viewer, data: np.ndarray, *, name: str | None = None, **kwargs
):
    """Add an image layer with project default voxel scale."""
    data = np.asarray(data)
    return viewer.add_image(
        data, name=name, scale=layer_scale_for_shape(data.shape), **kwargs
    )


def viewer_add_labels(
    viewer: napari.Viewer, data: np.ndarray, *, name: str | None = None, **kwargs
):
    """Add a labels layer with project default voxel scale."""
    data = np.asarray(data)
    return viewer.add_labels(
        data, name=name, scale=layer_scale_for_shape(data.shape), **kwargs
    )


def apply_channels_to_viewer(
    viewer: napari.Viewer, channels: dict[int, np.ndarray]
) -> None:
    """Add or update image layers without restarting the viewer."""
    for channel_idx in sorted(channels):
        name = f"channel {channel_idx}"
        volume = channels[channel_idx]
        if name in viewer.layers:
            layer = viewer.layers[name]
            layer.data = volume
            apply_layer_scale(layer)
        else:
            viewer_add_image(viewer, volume, name=name)
        print(f"{name}: shape {volume.shape}, dtype {volume.dtype}")
