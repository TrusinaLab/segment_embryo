"""Inspect instance labels: isolate one cell ID for fused-nucleus QC."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from image_io import SEGMENT_OUTPUT_DIR, apply_channels_to_viewer, viewer_add_labels
from segmentation.cell_features import (
    CELL_LABELS_LAYER,
    attach_label_hover_features,
    attach_label_hover_status,
    minimal_label_features_table,
)


def _apply_labels_display_style(layer, *, opacity: float = 0.65) -> None:
    """Contour mode is not supported on all napari versions; set when available."""
    layer.opacity = opacity
    layer.blending = "translucent"
    if hasattr(layer, "contour"):
        layer.contour = 2


def mask_labels_to_single_id(labels: np.ndarray, label_id: int) -> np.ndarray:
    """Return a label volume containing only ``label_id`` (background = 0)."""
    if label_id <= 0:
        return np.zeros_like(labels, dtype=np.uint32)
    out = np.zeros_like(labels, dtype=np.uint32)
    out[labels == label_id] = label_id
    return out


def _label_ids_in_volume(labels: np.ndarray) -> set[int]:
    ids = set(np.unique(labels).astype(int))
    ids.discard(0)
    return ids


@dataclass
class LabelFocusState:
    """Full label volumes; optional solo mode showing one instance ID."""

    full_by_layer: dict[str, np.ndarray]
    solo_label_id: int | None = None

    def set_solo(self, label_id: int) -> bool:
        """Show only ``label_id``. Returns False if the ID is not in any layer."""
        if label_id <= 0:
            return False
        for full in self.full_by_layer.values():
            if label_id in _label_ids_in_volume(full):
                self.solo_label_id = label_id
                return True
        return False

    def clear_solo(self) -> None:
        self.solo_label_id = None

    def volume_for_display(self, layer_name: str) -> np.ndarray:
        full = self.full_by_layer[layer_name]
        if self.solo_label_id is None:
            return full
        return mask_labels_to_single_id(full, self.solo_label_id)

    def apply_display(self, viewer) -> None:
        for name in self.full_by_layer:
            if name not in viewer.layers:
                continue
            layer = viewer.layers[name]
            layer.data = self.volume_for_display(name)
            layer.refresh()


def add_label_focus_widget(
    viewer,
    focus_state: LabelFocusState,
    *,
    pick_layer_name: str = CELL_LABELS_LAYER,
    on_cell_selected: Callable[[int], None] | None = None,
    on_after_display: Callable[[], None] | None = None,
) -> None:
    """Dock widgets to show one label ID or restore the full segmentation."""
    from magicgui import magicgui
    from napari.utils.notifications import show_info

    pick_layer = (
        viewer.layers[pick_layer_name] if pick_layer_name in viewer.layers else None
    )

    @magicgui(
        cell_id={"label": "Cell label ID", "min": 0, "value": 0},
        call_button="Show only this ID",
    )
    def show_only_this_id(cell_id: int = 0) -> None:
        if cell_id <= 0:
            show_info("Enter a positive label ID (click a cell to set it).")
            return
        if not focus_state.set_solo(cell_id):
            show_info(f"Label ID {cell_id} not found in the segmentation.")
            return
        focus_state.apply_display(viewer)
        if on_after_display is not None:
            on_after_display()
        print(f"Solo view: showing only label {cell_id}.")
        if on_cell_selected is not None:
            on_cell_selected(cell_id)

    @magicgui(call_button="Show all labels")
    def show_all_labels() -> None:
        focus_state.clear_solo()
        focus_state.apply_display(viewer)
        if on_after_display is not None:
            on_after_display()
        print("Showing all cell labels.")

    if pick_layer is not None:

        @pick_layer.mouse_drag_callbacks.append
        def on_label_pick(layer, event) -> None:
            if event.type != "mouse_press":
                return
            value = layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            if value and int(value) > 0:
                lid = int(value)
                show_only_this_id.cell_id.value = lid
                if focus_state.set_solo(lid):
                    focus_state.apply_display(viewer)
                    if on_after_display is not None:
                        on_after_display()
                    print(f"Solo view: label {lid} (from click on {pick_layer_name}).")
                    if on_cell_selected is not None:
                        on_cell_selected(lid)

    viewer.window.add_dock_widget(show_only_this_id)
    viewer.window.add_dock_widget(show_all_labels)


def apply_initial_focus(
    viewer, focus_state: LabelFocusState, focus_label_id: int
) -> None:
    if focus_state.set_solo(focus_label_id):
        focus_state.apply_display(viewer)
        print(f"Started in solo view for label {focus_label_id}.")
    else:
        print(f"Warning: --focus-label {focus_label_id} not found; showing all labels.")


def print_inspect_workflow(label_path: Path, shape: tuple[int, ...]) -> None:
    print(
        "Inspect cell labels (fused-nucleus QC):\n"
        f"  DAPI + channels from {SEGMENT_OUTPUT_DIR.as_posix()}/\n"
        f"  Labels: {label_path.name}  shape {shape}\n"
        f"  Layer: {CELL_LABELS_LAYER}\n"
        "  Dock: enter Cell label ID → Show only this ID\n"
        "  Or click a cell on the labels layer to isolate it\n"
        "  Show all labels — restore full segmentation\n"
        "  For re-segmenting fixes use:  python run_micro_sam_resegment.py\n"
    )


def launch_inspect_labels_viewer(
    channels: dict[int, np.ndarray],
    labels: np.ndarray,
    *,
    focus_label_id: int | None = None,
) -> None:
    """Napari viewer with segment channels and solo-cell inspection (no micro-SAM)."""
    import napari

    viewer = napari.Viewer()
    apply_channels_to_viewer(viewer, channels)
    label_data = np.asarray(labels, dtype=np.uint32)
    layer = viewer_add_labels(viewer, label_data, name=CELL_LABELS_LAYER)
    _apply_labels_display_style(layer)
    attach_label_hover_features(layer)
    attach_label_hover_status(layer, viewer)
    layer.editable = False
    layer.mode = "pan_zoom"
    viewer.layers.selection.active = layer

    focus_state = LabelFocusState(
        full_by_layer={CELL_LABELS_LAYER: label_data.copy()}
    )
    add_label_focus_widget(viewer, focus_state, pick_layer_name=CELL_LABELS_LAYER)
    if focus_label_id is not None:
        apply_initial_focus(viewer, focus_state, focus_label_id)

    napari.run()
