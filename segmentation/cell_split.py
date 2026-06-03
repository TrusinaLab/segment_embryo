"""Split a fused instance label using 3D-interpolated divider lines (Shapes)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import napari
import numpy as np
from magicgui import magicgui
from skimage.draw import line as skimage_line
from napari.layers import Labels, Shapes
from napari.utils.notifications import show_info, show_warning

from image_io import (
    CELL_LABELS_DIR,
    apply_channels_to_viewer,
    save_volume_tiff,
    viewer_add_labels,
)
from segmentation.cell_features import (
    CELL_LABELS_LAYER,
    attach_label_hover_features,
    attach_label_hover_status,
    minimal_label_features_table,
)
from segmentation.label_focus import (
    LabelFocusState,
    _apply_labels_display_style,
    add_label_focus_widget,
    apply_initial_focus,
)
from segmentation.plane_split import (
    collect_line_points,
    collect_lines_by_z,
    ensure_divider_lines_layer,
    fit_plane_from_points,
    keep_label_choices,
    parse_keep_label_choice,
    split_volume_by_interpolated_surface,
    split_volume_by_plane,
)

CELL_SPLIT_DIVIDER_LAYER = "cell_split_divider_lines"
CELL_SPLIT_PREVIEW_LAYER = "cell_split_preview"


class CellSplitMode(str, Enum):
    """How to cut a fused instance apart."""

    SURFACE = "xy lines per z (neck visible in slice)"
    PLANE = "plane (3d, any orientation)"
    Z_CUT = "z cut (nuclei stacked along z)"


def _next_label_ids(labels: np.ndarray, count: int = 2) -> tuple[int, ...]:
    max_id = int(labels.max())
    return tuple(range(max_id + 1, max_id + 1 + count))


def collect_marker_points_zyx(shapes_layer: Shapes) -> np.ndarray:
    """
    All shape vertices as (z, y, x).

    Use for z-cut mode: one line along z (same y, x), or several points on the neck.
    """
    if len(shapes_layer.data) == 0:
        raise ValueError(
            f"Layer '{shapes_layer.name}' is empty. "
            "Add points or a line along z on the cell_split_divider_lines layer."
        )
    chunks = [np.asarray(shape, dtype=float) for shape in shapes_layer.data]
    points = np.vstack(chunks)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points in (z, y, x); got {points.shape}.")
    return points


def _shape_polylines_zyx(shapes_layer: Shapes) -> list[np.ndarray]:
    """Each shape as (N, 3) float in (z, y, x)."""
    if len(shapes_layer.data) == 0:
        raise ValueError(
            f"Layer '{shapes_layer.name}' is empty. "
            "Draw a divider line on cell_split_divider_lines first."
        )
    polylines: list[np.ndarray] = []
    for shape in shapes_layer.data:
        arr = np.asarray(shape, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) points in (z, y, x); got {arr.shape}.")
        if arr.shape[0] >= 1:
            polylines.append(arr)
    if not polylines:
        raise ValueError(
            f"Layer '{shapes_layer.name}' has no valid points. "
            "Draw at least one line through the fused cell."
        )
    return polylines


def _sample_segment_zyx(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    z_max: int,
    y_max: int,
    x_max: int,
    n: int = 256,
) -> np.ndarray:
    """(N, 3) int samples along a segment, clipped to volume bounds."""
    t = np.linspace(0.0, 1.0, n)
    pts = p0 + t[:, None] * (p1 - p0)
    pts = np.round(pts).astype(np.int64)
    pts[:, 0] = np.clip(pts[:, 0], 0, z_max)
    pts[:, 1] = np.clip(pts[:, 1], 0, y_max)
    pts[:, 2] = np.clip(pts[:, 2], 0, x_max)
    return pts


def _sample_line_yx(
    y0: float,
    x0: float,
    y1: float,
    x1: float,
    *,
    y_max: int,
    x_max: int,
) -> np.ndarray:
    """(N, 2) int (y, x) along a 2D line segment."""
    y0i, x0i = int(round(y0)), int(round(x0))
    y1i, x1i = int(round(y1)), int(round(x1))
    rr, cc = skimage_line(
        np.clip(y0i, 0, y_max),
        np.clip(x0i, 0, x_max),
        np.clip(y1i, 0, y_max),
        np.clip(x1i, 0, x_max),
    )
    rr = np.clip(rr, 0, y_max)
    cc = np.clip(cc, 0, x_max)
    return np.column_stack([rr, cc])


def detect_target_cell_from_divider(
    labels: np.ndarray,
    shapes_layer: Shapes,
    mode: CellSplitMode,
) -> tuple[int, dict[int, int]]:
    """
    Infer which instance label the divider crosses by sampling ``labels`` along shapes.

    Returns the label ID with the most hits along the divider and per-ID hit counts.
    """
    z_max, y_max, x_max = (int(d) - 1 for d in labels.shape)
    counts: dict[int, int] = {}

    if mode == CellSplitMode.SURFACE:
        lines_by_z = collect_lines_by_z(shapes_layer)
        for z_idx, endpoints in lines_by_z.items():
            z = int(np.clip(z_idx, 0, z_max))
            y0, x0 = endpoints[0]
            y1, x1 = endpoints[1]
            for y, x in _sample_line_yx(y0, x0, y1, x1, y_max=y_max, x_max=x_max):
                lid = int(labels[z, y, x])
                if lid > 0:
                    counts[lid] = counts.get(lid, 0) + 1
    else:
        for arr in _shape_polylines_zyx(shapes_layer):
            if arr.shape[0] == 1:
                samples = _sample_segment_zyx(
                    arr[0],
                    arr[0],
                    z_max=z_max,
                    y_max=y_max,
                    x_max=x_max,
                    n=1,
                )
            else:
                chunks = [
                    _sample_segment_zyx(
                        arr[i],
                        arr[i + 1],
                        z_max=z_max,
                        y_max=y_max,
                        x_max=x_max,
                    )
                    for i in range(arr.shape[0] - 1)
                ]
                samples = np.vstack(chunks)
            for z, y, x in samples:
                lid = int(labels[int(z), int(y), int(x)])
                if lid > 0:
                    counts[lid] = counts.get(lid, 0) + 1

    if not counts:
        raise ValueError(
            "Divider line does not cross any labeled cell. "
            "Draw the line through the fused cell on cell_split_divider_lines."
        )

    best_id = max(counts, key=counts.get)
    total = sum(counts.values())
    if len(counts) > 1 and counts[best_id] / total < 0.6:
        show_warning(
            f"Line crosses multiple cells {counts}; using label {best_id} "
            f"({counts[best_id]}/{total} samples along the divider). "
            "Set ① Override cell ID if this is wrong."
        )
    return best_id, counts


def split_volume_by_z_cut(
    volume_shape: tuple[int, int, int],
    points_zyx: np.ndarray,
) -> np.ndarray:
    """
    Split with a cut surface z = f(y, x) (horizontal-ish, separates z-stacked nuclei).

    Label 1 = lower z (smaller index); label 2 = upper z.
    """
    points_zyx = np.asarray(points_zyx, dtype=float)
    if points_zyx.shape[0] < 1:
        raise ValueError("Need at least one 3D marker point for z cut.")

    z_size, y_size, x_size = volume_shape
    zz, yy, xx = np.ogrid[:z_size, :y_size, :x_size]

    if points_zyx.shape[0] == 1:
        z_surface = np.full((y_size, x_size), float(points_zyx[0, 0]))
    elif (
        points_zyx.shape[0] == 2
        and np.allclose(points_zyx[:, 1:], points_zyx[0, 1:], atol=1.0)
    ):
        z_surface = np.full((y_size, x_size), float(points_zyx[:, 0].mean()))
    else:
        design = np.column_stack(
            [points_zyx[:, 1], points_zyx[:, 2], np.ones(points_zyx.shape[0])]
        )
        a, b, c = np.linalg.lstsq(design, points_zyx[:, 0], rcond=None)[0]
        z_surface = a * yy + b * xx + c

    signed = zz - z_surface
    labels = np.zeros(volume_shape, dtype=np.uint8)
    labels[signed < 0] = 1
    labels[signed >= 0] = 2
    return labels


def build_split_field(
    volume_shape: tuple[int, int, int],
    lines_by_z: dict[int, np.ndarray],
    mode: CellSplitMode,
    *,
    plane_points: np.ndarray | None = None,
    marker_points_zyx: np.ndarray | None = None,
) -> np.ndarray:
    """Per-voxel split field: 0=unset, 1=positive side, 2=negative side."""
    if mode == CellSplitMode.Z_CUT:
        if marker_points_zyx is None:
            raise ValueError("Z cut mode requires 3D marker points.")
        return split_volume_by_z_cut(volume_shape, marker_points_zyx)
    if mode == CellSplitMode.PLANE:
        if plane_points is None:
            raise ValueError("Plane mode requires line points.")
        normal, d = fit_plane_from_points(plane_points)
        return split_volume_by_plane(volume_shape, normal, d)
    return split_volume_by_interpolated_surface(volume_shape, lines_by_z)


def split_preview_volume(
    labels: np.ndarray,
    source_id: int,
    split_field: np.ndarray,
) -> np.ndarray:
    """Preview labels 1/2 only inside ``source_id`` mask."""
    mask = labels == source_id
    preview = np.zeros(labels.shape, dtype=np.uint8)
    preview[mask & (split_field == 1)] = 1
    preview[mask & (split_field == 2)] = 2
    return preview


def apply_split_to_labels(
    labels: np.ndarray,
    source_id: int,
    split_field: np.ndarray,
    *,
    keep_sub_label: int = 1,
) -> tuple[np.ndarray, int, int]:
    """
    Split ``source_id`` into two cells.

    The chosen preview side **keeps** ``source_id``; the other side gets a new unique ID.
    ``keep_sub_label`` is 1 or 2 from the preview layer.
    """
    labels = np.asarray(labels, dtype=np.uint32).copy()
    mask = labels == source_id
    if not mask.any():
        raise ValueError(f"Label ID {source_id} is not present in the volume.")

    keep_mask = mask & (split_field == keep_sub_label)
    other_mask = mask & (split_field != keep_sub_label) & (split_field > 0)

    if not keep_mask.any() or not other_mask.any():
        raise ValueError(
            "Split does not cut through the selected cell — adjust divider lines "
            f"(kept side voxels={int(keep_mask.sum())}, other side={int(other_mask.sum())})."
        )

    existing = set(np.unique(labels).astype(int))
    new_id = int(labels.max()) + 1
    while new_id == source_id or new_id in existing:
        new_id += 1

    labels[keep_mask] = source_id
    labels[other_mask] = new_id
    return labels, source_id, new_id


@dataclass
class CellSplitState:
    """Working copy of instance labels and last preview split field."""

    labels: np.ndarray
    source_id: int = 0
    split_field: np.ndarray | None = field(default=None, repr=False)
    last_new_ids: tuple[int, int] | None = None
    applied_splits: list[tuple[int, int, int]] = field(default_factory=list)

    def record_split(self, old_id: int, kept_id: int, new_id: int) -> None:
        self.applied_splits.append((old_id, kept_id, new_id))

    def set_source_id(self, label_id: int) -> bool:
        if label_id <= 0:
            return False
        if label_id not in set(np.unique(self.labels).astype(int)):
            return False
        self.source_id = label_id
        self.split_field = None
        return True


def ensure_cell_split_divider_layer(viewer: napari.Viewer) -> Shapes:
    """3D Shapes layer for divider lines (separate name from embryo plane split)."""
    if CELL_SPLIT_DIVIDER_LAYER in viewer.layers:
        return viewer.layers[CELL_SPLIT_DIVIDER_LAYER]

    return viewer.add_shapes(
        name=CELL_SPLIT_DIVIDER_LAYER,
        ndim=3,
        edge_color="yellow",
        face_color="transparent",
        edge_width=2,
    )


def _sync_preview_layer(viewer: napari.Viewer, preview: np.ndarray) -> Labels:
    if CELL_SPLIT_PREVIEW_LAYER in viewer.layers:
        layer = viewer.layers[CELL_SPLIT_PREVIEW_LAYER]
        layer.data = preview
        layer.refresh()
    else:
        layer = viewer_add_labels(
            viewer, preview, name=CELL_SPLIT_PREVIEW_LAYER, opacity=0.85
        )
    layer.editable = False
    layer.mode = "pan_zoom"
    return layer


def _sync_working_labels_layer(viewer: napari.Viewer, labels: np.ndarray) -> Labels:
    data = np.asarray(labels, dtype=np.uint32)
    if CELL_LABELS_LAYER in viewer.layers:
        layer = viewer.layers[CELL_LABELS_LAYER]
        layer.data = data
        layer.refresh()
    else:
        layer = viewer_add_labels(viewer, data, name=CELL_LABELS_LAYER)
        _apply_labels_display_style(layer)
    layer.editable = False
    layer.mode = "pan_zoom"
    attach_label_hover_features(layer)
    return layer


def _refresh_working_labels(
    viewer: napari.Viewer, state: CellSplitState
) -> napari.layers.Labels:
    """Keep cell_labels layer data in sync with in-memory state (all splits preserved)."""
    return _sync_working_labels_layer(viewer, state.labels)


def _clear_divider_shapes(viewer: napari.Viewer) -> None:
    """Remove all divider markers (napari-safe: clear data only, not shape_type)."""
    if CELL_SPLIT_DIVIDER_LAYER not in viewer.layers:
        return
    layer = viewer.layers[CELL_SPLIT_DIVIDER_LAYER]
    layer.data = []


def _activate_split_target(
    state: CellSplitState,
    focus_state: LabelFocusState,
    viewer: napari.Viewer,
    source_id: int,
    *,
    refresh_layer: Callable[[], None] | None = None,
) -> bool:
    if source_id <= 0:
        return False
    if not state.set_source_id(source_id):
        show_warning(f"Label ID {source_id} not found in cell_labels.")
        return False
    focus_state.full_by_layer[CELL_LABELS_LAYER] = state.labels.copy()
    if focus_state.set_solo(source_id):
        focus_state.apply_display(viewer)
        if refresh_layer is not None:
            refresh_layer()
    print(
        f"Target cell for split: {source_id} "
        f"({len(state.applied_splits)} split(s) already applied this session)."
    )
    return True


def add_cell_split_widgets(
    viewer: napari.Viewer,
    state: CellSplitState,
    focus_state: LabelFocusState,
    *,
    refresh_layer: Callable[[], None],
) -> object:
    """Dock widgets: target ID, build preview, apply split, save."""

    @magicgui(
        source_id={
            "label": "① Override cell ID (0 = auto from line)",
            "min": 0,
            "value": 0,
        },
        call_button="Set target cell (optional)",
    )
    def set_target_cell(source_id: int = 0) -> None:
        if source_id <= 0:
            show_info(
                "Leave at 0 to auto-detect from the divider line when you build preview. "
                "Or enter an ID and click **Set target cell** to override."
            )
            return
        _activate_split_target(
            state, focus_state, viewer, source_id, refresh_layer=refresh_layer
        )

    def _resolve_target_for_build(
        shapes: Shapes, mode: CellSplitMode
    ) -> int | None:
        """Manual override, else detect label ID from divider geometry."""
        refresh_layer()
        manual = int(set_target_cell.source_id.value)
        if manual > 0:
            return manual
        try:
            detected, counts = detect_target_cell_from_divider(
                state.labels, shapes, mode
            )
        except ValueError as exc:
            show_warning(str(exc))
            return None
        print(f"Auto-detected fused cell from divider: label {detected} ({counts})")
        return detected

    def _ensure_target_for_apply() -> bool:
        if state.source_id > 0:
            return True
        show_warning(
            "Build split preview first — the target cell is set from your divider line."
        )
        return False

    @magicgui(
        split_mode={"choices": [m.value for m in CellSplitMode], "label": "Split mode"},
        call_button="Build split preview",
    )
    def build_split_preview(
        split_mode: str = CellSplitMode.Z_CUT.value,
    ) -> None:
        refresh_layer()
        shapes = ensure_cell_split_divider_layer(viewer)
        if len(shapes.data) == 0:
            show_warning(
                f"Draw a divider line on **{CELL_SPLIT_DIVIDER_LAYER}** first, "
                "then click Build split preview."
            )
            return
        mode = CellSplitMode(split_mode)
        target_id = _resolve_target_for_build(shapes, mode)
        if target_id is None:
            return
        if not _activate_split_target(
            state,
            focus_state,
            viewer,
            target_id,
            refresh_layer=refresh_layer,
        ):
            return
        set_target_cell.source_id.value = target_id
        try:
            if mode == CellSplitMode.Z_CUT:
                markers = collect_marker_points_zyx(shapes)
                split_field = build_split_field(
                    state.labels.shape,
                    {},
                    mode,
                    marker_points_zyx=markers,
                )
            elif mode == CellSplitMode.PLANE:
                points = collect_line_points(shapes)
                split_field = build_split_field(
                    state.labels.shape, {}, mode, plane_points=points
                )
            else:
                lines_by_z = collect_lines_by_z(shapes)
                split_field = build_split_field(state.labels.shape, lines_by_z, mode)
        except ValueError as exc:
            show_warning(str(exc))
            return

        state.split_field = split_field
        preview = split_preview_volume(state.labels, state.source_id, split_field)
        _sync_preview_layer(viewer, preview)
        n1 = int((preview == 1).sum())
        n2 = int((preview == 2).sum())
        layer = viewer.layers[CELL_SPLIT_PREVIEW_LAYER]
        apply_split.keep_side.choices = keep_label_choices(layer)
        print(
            f"Split preview for label {state.source_id} ({mode.value}): "
            f"side 1 = {n1} voxels, side 2 = {n2} voxels. "
            "Check cell_split_preview, then Apply split."
        )

    @magicgui(
        keep_side={
            "choices": ["Label 1 — build preview first", "Label 2 — build preview first"],
            "label": "Side that keeps original label ID",
        },
        call_button="Apply split to cell_labels",
    )
    def apply_split(
        keep_side: str = "Label 1 — build preview first",
    ) -> None:
        if not _ensure_target_for_apply():
            return
        if state.split_field is None:
            show_info("Click 'Build split preview' first.")
            return
        if "build preview first" in keep_side.lower():
            show_info(
                "Build preview first, then choose which side keeps the original label ID."
            )
            return

        keep_sub = parse_keep_label_choice(keep_side)
        old_id = state.source_id
        try:
            state.labels, kept_id, new_id = apply_split_to_labels(
                state.labels,
                old_id,
                state.split_field,
                keep_sub_label=keep_sub,
            )
        except ValueError as exc:
            show_warning(str(exc))
            return

        state.record_split(old_id, kept_id, new_id)
        state.last_new_ids = (kept_id, new_id)
        state.source_id = 0
        state.split_field = None
        set_target_cell.source_id.value = 0
        focus_state.full_by_layer[CELL_LABELS_LAYER] = state.labels.copy()
        focus_state.clear_solo()
        _refresh_working_labels(viewer, state)
        _clear_divider_shapes(viewer)
        if CELL_SPLIT_PREVIEW_LAYER in viewer.layers:
            viewer.layers[CELL_SPLIT_PREVIEW_LAYER].data = np.zeros(
                state.labels.shape, dtype=np.uint8
            )
            viewer.layers[CELL_SPLIT_PREVIEW_LAYER].refresh()
        n = len(state.applied_splits)
        print(
            f"Split {old_id}: kept ID {kept_id}, new ID {new_id}. "
            f"{n} split(s) applied this session — draw a new divider line, "
            "Build preview, Apply again (or Save when all done)."
        )

    @magicgui(call_button="Clear divider markers")
    def clear_divider_markers() -> None:
        _clear_divider_shapes(viewer)
        state.split_field = None
        print("Cleared cell_split_divider_lines for the next cell.")

    @magicgui(
        output_name={
            "label": "Output filename",
            "value": "cell_labels_after_split.tif",
        },
        call_button="Save cell_labels TIFF",
    )
    def save_labels(output_name: str = "cell_labels_after_split.tif") -> None:
        refresh_layer()
        out_dir = Path(CELL_LABELS_DIR)
        path = out_dir / output_name
        save_volume_tiff(state.labels, path)
        print(f"Saved {path} ({len(state.applied_splits)} split(s) in this session).")

    viewer.window.add_dock_widget(set_target_cell)
    viewer.window.add_dock_widget(build_split_preview)
    viewer.window.add_dock_widget(apply_split)
    viewer.window.add_dock_widget(clear_divider_markers)
    viewer.window.add_dock_widget(save_labels)
    return set_target_cell


def print_cell_split_workflow(label_path: Path) -> None:
    print(
        "Split fused cell with divider markers:\n"
        f"  Labels: {label_path.name}\n"
        f"  STEP 1: {CELL_SPLIT_DIVIDER_LAYER} — draw the cut line through the fused cell\n"
        "     (target label ID is auto-detected from the line; use ① only to override)\n"
        "  STEP 2: pick split mode:\n"
        "     z cut (default): nuclei stacked along z — draw ONE line through z\n"
        "       (same y,x at both ends; use xz/yz view in napari) or place neck points\n"
        "       Preview: label 1 = lower z, label 2 = upper z\n"
        "     xy lines per z: neck visible in each xy slice — line per z-slice (2+ z)\n"
        "     plane: general 3D plane through line points\n"
        "  STEP 3: **Build split preview** → solo view + auto target ID + check cell_split_preview\n"
        "  STEP 4: **Apply split** — repeat for each fused cell (splits accumulate in memory)\n"
        "  STEP 5: **Clear divider markers** before the next cell if needed\n"
        "  STEP 6: **Save cell_labels TIFF** once when all splits are done\n"
    )


def launch_cell_split_viewer(
    channels: dict[int, np.ndarray],
    labels: np.ndarray,
    *,
    target_label_id: int | None = None,
) -> None:
    import napari

    viewer = napari.Viewer()
    apply_channels_to_viewer(viewer, channels)
    label_data = np.asarray(labels, dtype=np.uint32)
    _sync_working_labels_layer(viewer, label_data)
    ensure_cell_split_divider_layer(viewer)

    split_state = CellSplitState(labels=label_data.copy())
    focus_state = LabelFocusState(
        full_by_layer={CELL_LABELS_LAYER: split_state.labels.copy()}
    )
    cell_layer = viewer.layers[CELL_LABELS_LAYER]
    attach_label_hover_status(cell_layer, viewer)

    def _refresh_layer() -> None:
        _refresh_working_labels(viewer, split_state)
        focus_state.full_by_layer[CELL_LABELS_LAYER] = split_state.labels.copy()

    set_target_widget = add_cell_split_widgets(
        viewer, split_state, focus_state, refresh_layer=_refresh_layer
    )

    def _on_cell_selected(label_id: int) -> None:
        _activate_split_target(
            split_state,
            focus_state,
            viewer,
            label_id,
            refresh_layer=_refresh_layer,
        )
        set_target_widget.source_id.value = label_id

    add_label_focus_widget(
        viewer,
        focus_state,
        pick_layer_name=CELL_LABELS_LAYER,
        on_cell_selected=_on_cell_selected,
        on_after_display=_refresh_layer,
    )

    if target_label_id is not None:
        if _activate_split_target(
            split_state,
            focus_state,
            viewer,
            target_label_id,
            refresh_layer=_refresh_layer,
        ):
            set_target_widget.source_id.value = target_label_id
        else:
            print(f"Warning: --target-label {target_label_id} not found.")

    napari.run()
