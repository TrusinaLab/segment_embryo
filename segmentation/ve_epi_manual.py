"""Manual VE / EPI assignment using Napari pick mode (tool 5) on cell_labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from magicgui import magicgui
from napari.utils.notifications import show_info

from image_io import (
    CELL_LABELS_DIR,
    EPI_VE_OUTPUT_DIR,
    align_label_volume_to_reference,
    apply_channels_to_viewer,
    discover_label_volume_path,
    load_label_volume,
    load_segment_channels,
    project_root,
    save_volume_tiff,
)
from segmentation.cell_features import CELL_LABELS_LAYER, save_features_csv

CLASS_UNLABELED = 0
CLASS_VE = 1
CLASS_EPI = 2
CLASS_NAMES = {CLASS_UNLABELED: "unlabeled", CLASS_VE: "VE", CLASS_EPI: "EPI"}

VE_EPI_MANUAL_LAYER = "ve_epi_manual"
MANUAL_CSV_NAME = "ve_epi_manual.csv"
MANUAL_LABELS_TIFF = "ve_epi_manual_labels.tif"

COLOR_VE = np.array([1.0, 0.15, 0.15, 0.85], dtype=np.float32)
COLOR_EPI = np.array([0.15, 0.45, 1.0, 0.85], dtype=np.float32)

ASSIGN_VE = "Mark picked cell as VE (red)"
ASSIGN_EPI = "Mark picked cell as EPI (blue)"
ASSIGN_OFF = "Off (pick to inspect only)"


@dataclass
class ManualClassState:
    """Per-cell manual_class assignments keyed by instance label id."""

    instance_labels: np.ndarray
    assignments: dict[int, int] = field(default_factory=dict)
    _cell_ids: set[int] = field(init=False, repr=False)
    _class_volume: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ids = np.unique(self.instance_labels)
        self._cell_ids = set(ids.astype(int)) - {0}
        self._class_volume = np.zeros(self.instance_labels.shape, dtype=np.uint16)
        if self.assignments:
            self.rebuild_class_volume()

    def all_cell_ids(self) -> set[int]:
        return self._cell_ids

    def counts(self) -> dict[str, int]:
        ve = sum(1 for c in self.assignments.values() if c == CLASS_VE)
        epi = sum(1 for c in self.assignments.values() if c == CLASS_EPI)
        labeled = ve + epi
        total = len(self._cell_ids)
        return {
            "total_cells": total,
            "ve": ve,
            "epi": epi,
            "unlabeled": total - labeled,
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for label_id in sorted(self._cell_ids):
            cls = self.assignments.get(label_id, CLASS_UNLABELED)
            rows.append(
                {
                    "label": label_id,
                    "manual_class": cls if cls != CLASS_UNLABELED else np.nan,
                    "class_name": CLASS_NAMES.get(cls, "unlabeled"),
                }
            )
        return pd.DataFrame(rows)

    def patch_cell_class(self, label_id: int, class_id: int) -> None:
        """Update overlay voxels for one cell only (fast path for picks)."""
        value = 0 if class_id == CLASS_UNLABELED else class_id
        self._class_volume[self.instance_labels == label_id] = value

    def rebuild_class_volume(self) -> np.ndarray:
        """Rebuild full overlay in one pass over the volume (bulk updates)."""
        self._class_volume.fill(0)
        if not self.assignments:
            return self._class_volume
        max_label = int(self.instance_labels.max())
        lut = np.zeros(max_label + 1, dtype=np.uint16)
        for label_id, class_id in self.assignments.items():
            if class_id != CLASS_UNLABELED:
                lut[label_id] = class_id
        np.copyto(self._class_volume, lut[self.instance_labels])
        return self._class_volume

    def class_volume(self) -> np.ndarray:
        return self._class_volume


def default_manual_csv_path() -> Path:
    return project_root() / EPI_VE_OUTPUT_DIR / MANUAL_CSV_NAME


def default_manual_labels_path() -> Path:
    return project_root() / EPI_VE_OUTPUT_DIR / MANUAL_LABELS_TIFF


def load_assignments_from_csv(path: Path) -> dict[int, int]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"CSV must contain 'label' column: {path}")
    class_col = "manual_class" if "manual_class" in df.columns else "prediction"
    if class_col not in df.columns:
        raise ValueError(f"CSV must contain '{class_col}' column: {path}")

    assignments: dict[int, int] = {}
    for _, row in df.iterrows():
        label_id = int(row["label"])
        if pd.isna(row[class_col]):
            continue
        cls = int(row[class_col])
        if cls in (CLASS_VE, CLASS_EPI):
            assignments[label_id] = cls
    return assignments


def _apply_class_colormap(layer: napari.layers.Labels) -> None:
    layer.color = {CLASS_VE: COLOR_VE, CLASS_EPI: COLOR_EPI}
    layer.opacity = 0.75
    layer.blending = "translucent"


def _sync_manual_layer(
    viewer: napari.Viewer, state: ManualClassState, *, in_place: bool = False
) -> None:
    volume = state.class_volume()
    if VE_EPI_MANUAL_LAYER in viewer.layers:
        layer = viewer.layers[VE_EPI_MANUAL_LAYER]
        if in_place:
            layer.events.data()
        else:
            layer.data = volume
    else:
        layer = viewer.add_labels(volume, name=VE_EPI_MANUAL_LAYER, blending="translucent")
    _apply_class_colormap(layer)
    viewer.layers.move(viewer.layers.index(layer), len(viewer.layers) - 1)


def _print_counts(state: ManualClassState) -> None:
    c = state.counts()
    print(
        f"Manual labels: {c['ve']} VE, {c['epi']} EPI, "
        f"{c['unlabeled']} unlabeled / {c['total_cells']} cells"
    )


def _enable_pick_mode(cell_layer: napari.layers.Labels) -> None:
    """Activate labels pick mode (same as keyboard tool 5)."""
    cell_layer.mode = "pick"
    cell_layer.opacity = 0.6
    cell_layer.blending = "translucent"


def setup_ve_epi_manual_viewer(
    *,
    label_path: Path | None = None,
    resume_csv: Path | None = None,
) -> tuple[napari.Viewer, ManualClassState, napari.layers.Labels]:
    print("Loading cell labels...")
    labels = load_label_volume(label_path)
    print("Loading segment channels (TIFF stack)...")
    channels = load_segment_channels()

    ref = channels[min(channels)]
    labels = align_label_volume_to_reference(labels, ref.shape)
    n_cells = len(set(np.unique(labels).astype(int)) - {0})
    print(f"Opening Napari ({ref.shape[0]} z-slices, {n_cells} cells)...")

    csv_path = resume_csv or default_manual_csv_path()
    assignments = load_assignments_from_csv(csv_path)
    if assignments:
        print(f"Resumed {len(assignments)} manual assignments from {csv_path.name}")

    state = ManualClassState(instance_labels=labels, assignments=assignments)

    viewer = napari.Viewer()
    apply_channels_to_viewer(viewer, channels)
    cell_layer = viewer.add_labels(labels, name=CELL_LABELS_LAYER)
    _enable_pick_mode(cell_layer)
    viewer.layers.selection.active = cell_layer

    _sync_manual_layer(viewer, state)

    print(
        "Manual VE / EPI labeling (pick mode):\n"
        f"  Cells: {n_cells}\n"
        "  1. Layer 'cell_labels' is selected — pick mode is ON (same as tool 5).\n"
        f"  2. Dock dropdown: default '{ASSIGN_VE}'.\n"
        "  3. Click a nucleus → that cell turns red on 've_epi_manual'.\n"
        "  4. When done → 'Assign EPI to all remaining cells', then Save.\n"
    )
    _print_counts(state)

    return viewer, state, cell_layer


def add_ve_epi_manual_widgets(
    viewer: napari.Viewer,
    state: ManualClassState,
    cell_layer: napari.layers.Labels,
) -> None:
    ui = {"assign_mode": ASSIGN_VE}

    def _assign_picked_cell(label_id: int, class_id: int, class_name: str) -> None:
        if label_id == 0:
            show_info("Background — click a nucleus in cell_labels.")
            return
        if label_id not in state.all_cell_ids():
            show_info(f"Label {label_id} is not a segmented cell.")
            return
        state.assignments[label_id] = class_id
        state.patch_cell_class(label_id, class_id)
        _sync_manual_layer(viewer, state, in_place=True)
        print(f"Cell {label_id} → {class_name} (red)" if class_id == CLASS_VE else f"Cell {label_id} → {class_name} (blue)")
        _print_counts(state)

    @cell_layer.mouse_drag_callbacks.append
    def on_pick_click(layer, event) -> None:
        """Run when user clicks in pick mode (tool 5) on cell_labels."""
        if event.type != "mouse_press":
            return
        if layer.mode != "pick":
            return
        mode = ui["assign_mode"]
        if mode == ASSIGN_OFF:
            return
        label_id = (
            layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
            or 0
        )
        class_id = CLASS_VE if mode == ASSIGN_VE else CLASS_EPI
        class_name = "VE" if class_id == CLASS_VE else "EPI"
        _assign_picked_cell(int(label_id), class_id, class_name)

    @magicgui(
        assign_mode={
            "choices": [ASSIGN_VE, ASSIGN_EPI, ASSIGN_OFF],
            "label": "On each pick (tool 5)",
            "value": ASSIGN_VE,
        },
        auto_call=True,
    )
    def labeling_mode(assign_mode: str = ASSIGN_VE) -> None:
        """Choose what happens when you pick a cell on cell_labels (updates immediately)."""
        ui["assign_mode"] = assign_mode

    @magicgui(call_button="Assign EPI to all remaining cells")
    def assign_epi_remaining() -> None:
        to_epi = [
            lid
            for lid in state.all_cell_ids()
            if state.assignments.get(lid, CLASS_UNLABELED) == CLASS_UNLABELED
        ]
        if not to_epi:
            print("No unlabeled cells left.")
            return
        for label_id in to_epi:
            state.assignments[label_id] = CLASS_EPI
        state.rebuild_class_volume()
        _sync_manual_layer(viewer, state)
        print(f"Assigned EPI (blue) to {len(to_epi)} remaining cells.")
        _print_counts(state)

    @magicgui(call_button="Save manual labels (CSV + TIFF)")
    def save_manual() -> None:
        csv_out = default_manual_csv_path()
        tiff_out = default_manual_labels_path()
        save_features_csv(state.to_dataframe(), csv_out)
        save_volume_tiff(state.class_volume(), tiff_out)
        print(f"Saved {csv_out}")
        print(f"Saved {tiff_out}")

    @magicgui(call_button="Reload assignments from CSV")
    def reload_csv() -> None:
        state.assignments = load_assignments_from_csv(default_manual_csv_path())
        state.rebuild_class_volume()
        _sync_manual_layer(viewer, state)
        print("Reloaded assignments from CSV.")
        _print_counts(state)

    viewer.window.add_dock_widget(labeling_mode)
    viewer.window.add_dock_widget(assign_epi_remaining)
    viewer.window.add_dock_widget(save_manual)
    viewer.window.add_dock_widget(reload_csv)
