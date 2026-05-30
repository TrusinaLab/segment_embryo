"""Fit a dividing plane or interpolated surface from sparse line annotations."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Callable

import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Image, Labels, Shapes
from qtpy.QtCore import Qt

from image_io import (
    IMAGE_DIR_NAME,
    SEGMENT_OUTPUT_DIR,
    apply_channels_to_viewer,
    get_stack_z_index_list,
    load_project_channels,
    project_root,
    save_channel_stack_as_tiffs,
)

DIVIDER_LINES_LAYER = "divider_lines"
PLANE_SPLIT_LABELS_LAYER = "plane_split"
REFERENCE_IMAGE_LAYER = "channel 1"
KEEP_SPLIT_LABEL = 1  # positive side (shown as the first/red-ish label in napari)


class SplitMode(str, Enum):
    PLANE = "plane"
    SURFACE = "interpolated surface"


_KEEP_LABEL_CHOICE_RE = re.compile(r"^Label\s+(\d+)")
_SAVE_LABEL_UI_REFRESH: Callable[[napari.Viewer], None] | None = None


def ensure_divider_lines_layer(viewer: napari.Viewer) -> Shapes:
    """Create an empty 3D shapes layer for divider lines if needed."""
    if DIVIDER_LINES_LAYER in viewer.layers:
        return viewer.layers[DIVIDER_LINES_LAYER]

    return viewer.add_shapes(
        name=DIVIDER_LINES_LAYER,
        ndim=3,
        edge_color="red",
        face_color="transparent",
        edge_width=2,
    )


def _reference_image_layer(viewer: napari.Viewer) -> Image:
    if REFERENCE_IMAGE_LAYER not in viewer.layers:
        raise ValueError(
            f"Reference layer '{REFERENCE_IMAGE_LAYER}' not found. "
            "Load image channels first."
        )
    layer = viewer.layers[REFERENCE_IMAGE_LAYER]
    if not isinstance(layer, Image):
        raise TypeError(f"Layer '{REFERENCE_IMAGE_LAYER}' is not an image layer.")
    return layer


def collect_line_points(shapes_layer: Shapes) -> np.ndarray:
    """
    Gather (z, y, x) coordinates from all line shapes.

    Each shape is an (N, 3) array; line tools typically give N=2 endpoints.
    """
    if len(shapes_layer.data) == 0:
        raise ValueError(
            f"Layer '{shapes_layer.name}' is empty. "
            "Draw at least one line on a z-slice before building the split."
        )

    chunks = [np.asarray(shape, dtype=float) for shape in shapes_layer.data]
    points = np.vstack(chunks)
    if points.shape[1] != 3:
        raise ValueError(f"Expected 3D points (z, y, x); got shape {points.shape}.")

    return points


def collect_lines_by_z(shapes_layer: Shapes) -> dict[int, np.ndarray]:
    """
    Map z-index -> line endpoints with shape (2, 2) holding (y, x) pairs.

    One line per z-slice is expected. Multiple lines on the same z-slice are
    averaged.
    """
    if len(shapes_layer.data) == 0:
        raise ValueError(
            f"Layer '{shapes_layer.name}' is empty. "
            "Draw at least one line on a z-slice before building the split."
        )

    grouped: dict[int, list[np.ndarray]] = {}
    for shape in shapes_layer.data:
        arr = np.asarray(shape, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Expected line points (z, y, x); got shape {arr.shape}.")
        if arr.shape[0] < 2:
            continue

        p_start, p_end = arr[0], arr[-1]
        z_idx = int(round(0.5 * (p_start[0] + p_end[0])))
        if abs(p_start[0] - p_end[0]) > 1.0:
            print(
                f"Warning: line spans z={p_start[0]:.1f}–{p_end[0]:.1f}; "
                f"using z={z_idx} for surface interpolation."
            )
        endpoints = np.array([p_start[1:3], p_end[1:3]])
        grouped.setdefault(z_idx, []).append(endpoints)

    if not grouped:
        raise ValueError(
            f"Layer '{shapes_layer.name}' has no valid lines. "
            "Each line needs two endpoints."
        )

    lines_by_z: dict[int, np.ndarray] = {}
    for z_idx, endpoint_list in grouped.items():
        lines_by_z[z_idx] = (
            endpoint_list[0]
            if len(endpoint_list) == 1
            else np.mean(endpoint_list, axis=0)
        )

    return lines_by_z


def rgba_to_hex(rgba: np.ndarray) -> str:
    """Convert an RGBA float array to ``#rrggbb``."""
    channels = [int(round(255 * float(c))) for c in rgba[:3]]
    return "#{:02x}{:02x}{:02x}".format(*channels)


def _label_color_name(rgba: np.ndarray) -> str:
    """Short human-readable color name for legend and dropdown text."""
    red, green, blue = (float(c) for c in rgba[:3])
    if red > 0.35 and green < 0.25 and blue < 0.15:
        return "red-brown"
    if blue > 0.75 and green > 0.55:
        return "cyan"
    if green > red and green > blue:
        return "green"
    if red > green and red > blue:
        return "red"
    if blue > red and blue > green:
        return "blue"
    return "colored"


def split_label_color_info(labels_layer: Labels, label: int) -> tuple[str, str, str]:
    """Return ``(hex, color_name, side)`` for a split label id."""
    rgba = np.asarray(labels_layer.get_color(label), dtype=float)
    side = "positive side" if label == 1 else "negative side"
    return rgba_to_hex(rgba), _label_color_name(rgba), side


def keep_label_choice_text(labels_layer: Labels, label: int) -> str:
    """Dropdown entry showing label id, napari color, and side."""
    hex_color, color_name, side = split_label_color_info(labels_layer, label)
    return f"Label {label} — {color_name} ({hex_color}), {side}"


def keep_label_choices(labels_layer: Labels) -> list[str]:
    return [keep_label_choice_text(labels_layer, 1), keep_label_choice_text(labels_layer, 2)]


def label_colors_legend_html(labels_layer: Labels) -> str:
    """HTML color swatches matching the ``plane_split`` layer in napari."""
    parts: list[str] = []
    for label in (1, 2):
        hex_color, color_name, side = split_label_color_info(labels_layer, label)
        parts.append(
            f'<span style="background:{hex_color}; padding:2px 14px; '
            f'border:1px solid #666; margin-right:4px;">&nbsp;</span>'
            f"<b>Label {label}</b> — {color_name} ({hex_color}), {side}"
        )
    return "<br>".join(parts)


def parse_keep_label_choice(choice: str) -> int:
    match = _KEEP_LABEL_CHOICE_RE.match(choice.strip())
    if not match:
        raise ValueError(f"Could not parse label from choice: {choice!r}")
    return int(match.group(1))


def refresh_save_label_ui(viewer: napari.Viewer) -> None:
    if _SAVE_LABEL_UI_REFRESH is not None:
        _SAVE_LABEL_UI_REFRESH(viewer)


def interpolate_line_endpoints(
    z_query: float, lines_by_z: dict[int, np.ndarray]
) -> np.ndarray:
    """Linearly interpolate line endpoints in z; clamp outside the annotated range."""
    z_keys = sorted(lines_by_z)
    if z_query <= z_keys[0]:
        return lines_by_z[z_keys[0]].copy()
    if z_query >= z_keys[-1]:
        return lines_by_z[z_keys[-1]].copy()

    for z0, z1 in zip(z_keys[:-1], z_keys[1:]):
        if z0 <= z_query <= z1:
            if z1 == z0:
                return lines_by_z[z0].copy()
            t = (z_query - z0) / (z1 - z0)
            return (1.0 - t) * lines_by_z[z0] + t * lines_by_z[z1]

    return lines_by_z[z_keys[-1]].copy()


def fit_plane_from_points(points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Least-squares plane through ``points`` in (z, y, x) coordinates.

    Returns unit normal ``n`` and offset ``d`` such that ``n @ p + d = 0`` on
    the plane, with ``p = (z, y, x)``.
    """
    if points.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 points to fit a plane; got {points.shape[0]}. "
            "Draw lines on two or more z-slices (each line adds two points)."
        )

    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("Could not fit a plane: all points are identical.")

    normal = normal / norm
    d = -float(np.dot(normal, centroid))
    return normal, d


def signed_distance_field(
    volume_shape: tuple[int, ...], normal: np.ndarray, d: float
) -> np.ndarray:
    """Signed distance to plane for every voxel in a (Z, Y, X) volume."""
    zz, yy, xx = np.ogrid[: volume_shape[0], : volume_shape[1], : volume_shape[2]]
    return normal[0] * zz + normal[1] * yy + normal[2] * xx + d


def split_volume_by_plane(
    volume_shape: tuple[int, ...],
    normal: np.ndarray,
    d: float,
) -> np.ndarray:
    """
    Label volume: 0=background (unused), 1=positive side, 2=negative side.

    Positive side is where ``n @ (z,y,x) + d >= 0``.
    """
    signed = signed_distance_field(volume_shape, normal, d)
    labels = np.zeros(volume_shape, dtype=np.uint8)
    labels[signed >= 0] = 1
    labels[signed < 0] = 2
    return labels


def split_volume_by_interpolated_surface(
    volume_shape: tuple[int, ...],
    lines_by_z: dict[int, np.ndarray],
) -> np.ndarray:
    """
    Label volume by interpolating divider lines between annotated z-slices.

    On each z-slice the split follows a 2D line; endpoints are linearly
    interpolated in z between key slices. This yields a curved dividing
    surface (a ruled surface) rather than a single flat plane.
    """
    if len(lines_by_z) < 2:
        raise ValueError(
            "Interpolated surface mode needs divider lines on at least "
            f"2 different z-slices; got {len(lines_by_z)}."
        )

    z_size, y_size, x_size = volume_shape
    labels = np.zeros(volume_shape, dtype=np.uint8)

    yy = np.arange(y_size, dtype=float)[:, None]
    xx = np.arange(x_size, dtype=float)[None, :]

    for z in range(z_size):
        line = interpolate_line_endpoints(float(z), lines_by_z)
        y0, x0 = line[0]
        y1, x1 = line[1]
        length_sq = (y1 - y0) ** 2 + (x1 - x0) ** 2
        if length_sq < 1e-6:
            raise ValueError(
                f"Degenerate zero-length divider line at interpolated z={z}."
            )
        signed = (yy - y0) * (x1 - x0) - (xx - x0) * (y1 - y0)
        labels[z, signed >= 0] = 1
        labels[z, signed < 0] = 2

    return labels


def keep_mask_from_split_labels(labels: np.ndarray, keep_label: int = KEEP_SPLIT_LABEL) -> np.ndarray:
    """Boolean mask for voxels to keep (default: label 1 / positive side)."""
    return labels == keep_label


def mask_volume(volume: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Zero voxels outside ``keep_mask``; preserve dtype."""
    masked = np.zeros_like(volume)
    masked[keep_mask] = volume[keep_mask]
    return masked


def iter_channel_image_layers(viewer: napari.Viewer) -> list[tuple[int, Image]]:
    """Yield ``(channel_index, layer)`` for loaded ``channel N`` image layers."""
    layers: list[tuple[int, Image]] = []
    for layer in viewer.layers:
        if not isinstance(layer, Image):
            continue
        if not layer.name.startswith("channel "):
            continue
        channel_idx = int(layer.name.rsplit(" ", 1)[-1])
        layers.append((channel_idx, layer))
    return sorted(layers, key=lambda item: item[0])


def apply_masked_channel_layers(viewer: napari.Viewer, keep_mask: np.ndarray) -> None:
    """Add or update ``channel N (masked)`` preview layers."""
    for channel_idx, layer in iter_channel_image_layers(viewer):
        name = f"channel {channel_idx} (masked)"
        masked = mask_volume(layer.data, keep_mask)
        if name in viewer.layers:
            viewer.layers[name].data = masked
        else:
            viewer.add_image(masked, name=name, opacity=layer.opacity)


def save_masked_segment_tiffs(
    viewer: napari.Viewer,
    output_dir: Path | None = None,
    keep_label: int = KEEP_SPLIT_LABEL,
) -> list[Path]:
    """
    Mask all channel layers to the kept split region and save per-slice TIFFs.

    Keeps voxels where ``plane_split == keep_label`` (default label 1).
    """
    if PLANE_SPLIT_LABELS_LAYER not in viewer.layers:
        raise ValueError(
            f"Layer '{PLANE_SPLIT_LABELS_LAYER}' not found. "
            "Click 'Build split' before saving."
        )

    labels = np.asarray(viewer.layers[PLANE_SPLIT_LABELS_LAYER].data)
    keep_mask = keep_mask_from_split_labels(labels, keep_label=keep_label)
    if not keep_mask.any():
        raise ValueError("Keep mask is empty. Check the split labels layer.")

    out_dir = output_dir or (project_root() / SEGMENT_OUTPUT_DIR)
    z_indices = get_stack_z_index_list()

    written: list[Path] = []
    for channel_idx, layer in iter_channel_image_layers(viewer):
        masked = mask_volume(layer.data, keep_mask)
        written.extend(
            save_channel_stack_as_tiffs(
                masked,
                channel_idx,
                z_indices,
                out_dir,
                prefix=IMAGE_DIR_NAME,
            )
        )

    apply_masked_channel_layers(viewer, keep_mask)
    return written


def apply_plane_split_labels(viewer: napari.Viewer, labels: np.ndarray) -> Labels:
    """Show or update the plane-split labels layer."""
    if PLANE_SPLIT_LABELS_LAYER in viewer.layers:
        layer = viewer.layers[PLANE_SPLIT_LABELS_LAYER]
        layer.data = labels
        return layer

    return viewer.add_labels(
        labels,
        name=PLANE_SPLIT_LABELS_LAYER,
        opacity=0.4,
        blending="translucent",
    )


def build_volume_split(viewer: napari.Viewer, mode: SplitMode = SplitMode.PLANE) -> None:
    """Fit a divider (plane or interpolated surface) and update labels."""
    if DIVIDER_LINES_LAYER not in viewer.layers:
        raise ValueError(
            f"Add shapes to layer '{DIVIDER_LINES_LAYER}' before building the split."
        )

    shapes_layer = viewer.layers[DIVIDER_LINES_LAYER]
    ref = _reference_image_layer(viewer)

    if mode == SplitMode.PLANE:
        points = collect_line_points(shapes_layer)
        normal, d = fit_plane_from_points(points)
        labels = split_volume_by_plane(ref.data.shape, normal, d)
        detail = f"fitted plane n={normal}, d={d:.3f}"
    else:
        lines_by_z = collect_lines_by_z(shapes_layer)
        labels = split_volume_by_interpolated_surface(ref.data.shape, lines_by_z)
        z_keys = sorted(lines_by_z)
        detail = f"interpolated surface through z-slices {z_keys}"

    apply_plane_split_labels(viewer, labels)

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 2).sum())
    print(
        f"{PLANE_SPLIT_LABELS_LAYER}: {detail}\n"
        f"  label 1 (positive side): {n_pos} voxels\n"
        f"  label 2 (negative side): {n_neg} voxels"
    )
    refresh_save_label_ui(viewer)


def build_plane_split(viewer: napari.Viewer) -> None:
    """Fit a plane from divider lines and update the split labels layer."""
    build_volume_split(viewer, mode=SplitMode.PLANE)


def build_surface_split(viewer: napari.Viewer) -> None:
    """Build split from z-interpolated divider lines."""
    build_volume_split(viewer, mode=SplitMode.SURFACE)


def setup_plane_split_viewer(viewer: napari.Viewer) -> None:
    """Load full z-stack, image channels, and the divider lines layer."""
    channels = load_project_channels()
    apply_channels_to_viewer(viewer, channels)
    ensure_divider_lines_layer(viewer)
    z_count = channels[min(channels)].shape[0]
    z_keys = get_stack_z_index_list()
    print(
        "Plane / surface split workflow:\n"
        f"  Loaded {z_count} z-slices (absolute z {z_keys[0]}–{z_keys[-1]}).\n"
        f"  1. Select layer '{DIVIDER_LINES_LAYER}' and the line tool.\n"
        "  2. On several z-slices, draw lines along the embryo / trophectoderm boundary.\n"
        "  3. Choose split mode and click 'Build split':\n"
        "       plane — single flat surface (best when lines are nearly coplanar)\n"
        "       interpolated surface — lines move smoothly between z-slices\n"
        "         (use when lines on different z do not lie on one plane)\n"
        "  4. Choose label to keep and click 'Save masked TIFFs'\n"
        f"     to write masked TIFFs for all z to {SEGMENT_OUTPUT_DIR.as_posix()}/"
    )


@magicgui(call_button="Reload images from disk")
def reload_images_widget(viewer: napari.Viewer) -> None:
    """Reload the full z-stack from 22A_E1_Wnt3."""
    apply_channels_to_viewer(viewer, load_project_channels())


@magicgui(
    split_mode={"choices": [mode.value for mode in SplitMode], "label": "Split mode"},
    call_button="Build split",
)
def build_split_widget(
    viewer: napari.Viewer,
    split_mode: str = SplitMode.PLANE.value,
) -> None:
    """Fit a plane or interpolated surface through divider lines."""
    mode = SplitMode(split_mode)
    build_volume_split(viewer, mode=mode)


@magicgui(
    label_colors={"widget_type": "Label", "label": "Label colors in napari"},
    keep_label={
        "choices": ["Label 1 — build split first", "Label 2 — build split first"],
        "label": "Label to keep",
    },
    call_button="Save masked TIFFs",
)
def save_masked_tiffs_widget(
    viewer: napari.Viewer,
    label_colors: str = "Build split to show label colors.",
    keep_label: str = "Label 1 — build split first",
) -> None:
    """Keep the chosen split label, zero the rest, save to data/test segment embryo."""
    if PLANE_SPLIT_LABELS_LAYER not in viewer.layers:
        raise ValueError(
            f"Layer '{PLANE_SPLIT_LABELS_LAYER}' not found. "
            "Click 'Build split' before saving."
        )

    label_value = parse_keep_label_choice(keep_label)
    paths = save_masked_segment_tiffs(viewer, keep_label=label_value)
    out_dir = paths[0].parent if paths else project_root() / SEGMENT_OUTPUT_DIR
    print(f"Saved {len(paths)} TIFFs (kept label {label_value}) to {out_dir}")
    for path in paths[:6]:
        print(f"  {path.name}")
    if len(paths) > 6:
        print(f"  ... and {len(paths) - 6} more")


def _configure_save_label_ui(viewer: napari.Viewer) -> None:
    """Update color legend and dropdown to match the active ``plane_split`` layer."""
    legend = save_masked_tiffs_widget.label_colors
    legend.native.setTextFormat(Qt.RichText)
    legend.native.setWordWrap(True)

    if PLANE_SPLIT_LABELS_LAYER not in viewer.layers:
        legend.native.setText("<i>Build split to show label colors.</i>")
        return

    labels_layer = viewer.layers[PLANE_SPLIT_LABELS_LAYER]
    if not isinstance(labels_layer, Labels):
        legend.native.setText("<i>Build split to show label colors.</i>")
        return

    choices = keep_label_choices(labels_layer)
    current = save_masked_tiffs_widget.keep_label.value
    save_masked_tiffs_widget.keep_label.choices = choices
    if current in choices:
        save_masked_tiffs_widget.keep_label.value = current
    else:
        save_masked_tiffs_widget.keep_label.value = choices[0]

    legend.native.setText(label_colors_legend_html(labels_layer))


def add_plane_split_widgets(viewer: napari.Viewer) -> None:
    """Add dock widgets for reload and volume splitting."""
    global _SAVE_LABEL_UI_REFRESH
    _SAVE_LABEL_UI_REFRESH = _configure_save_label_ui

    viewer.window.add_dock_widget(reload_images_widget)
    viewer.window.add_dock_widget(build_split_widget)
    viewer.window.add_dock_widget(save_masked_tiffs_widget)
    _configure_save_label_ui(viewer)
