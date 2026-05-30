"""Manual VE / EPI assignment by clicking cells in Napari."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from magicgui import magicgui

from image_io import (
    CELL_LABELS_DIR,
    EPI_VE_OUTPUT_DIR,
    apply_channels_to_viewer,
    discover_label_volume_path,
    load_label_volume,
    load_segment_channels,
    project_root,
    save_volume_tiff,
)
from segmentation.cell_features import (
    CELL_LABELS_LAYER,
    compute_cell_features,
    predictions_to_label_volume,
    save_features_csv,
)

CLASS_UNLABELED = 0
CLASS_VE = 1
CLASS_EPI = 2
CLASS_NAMES = {CLASS_UNLABELED: "unlabeled", CLASS_VE: "VE", CLASS_EPI: "EPI"}

VE_EPI_MANUAL_LAYER = "ve_epi_manual"
MANUAL_CSV_NAME = "ve_epi_manual.csv"
MANUAL_LABELS_TIFF = "ve_epi_manual_labels.tif"


@dataclass
class ManualClassState:
    """Per-cell manual_class assignments keyed by instance label id."""

    instance_labels: np.ndarray
    assignments: dict[int, int] = field(default_factory=dict)

    def all_cell_ids(self) -> set[int]:
        ids = set(np.unique(self.instance_labels).astype(int))
        ids.discard(0)
        return ids

    def counts(self) -> dict[str, int]:
        ve = sum(1 for c in self.assignments.values() if c == CLASS_VE)
        epi = sum(1 for c in self.assignments.values() if c == CLASS_EPI)
        labeled = ve + epi
        total = len(self.all_cell_ids())
        return {
            "total_cells": total,
            "ve": ve,
            "epi": epi,
            "unlabeled": total - labeled,
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for label_id in sorted(self.all_cell_ids()):
            cls = self.assignments.get(label_id, CLASS_UNLABELED)
            rows.append(
                {
                    "label": label_id,
                    "manual_class": cls if cls != CLASS_UNLABELED else np.nan,
                    "class_name": CLASS_NAMES.get(cls, "unlabeled"),
                }
            )
        return pd.DataFrame(rows)

    def class_volume(self) -> np.ndarray:
        df = self.to_dataframe()
        df_plot = df.copy()
        df_plot["manual_class"] = df_plot["manual_class"].fillna(0).astype(int)
        return predictions_to_label_volume(
            self.instance_labels, df_plot, class_column="manual_class"
        )


def default_manual_csv_path() -> Path:
    return project_root() / EPI_VE_OUTPUT_DIR / MANUAL_CSV_NAME


def default_manual_labels_path() -> Path:
    return project_root() / EPI_VE_OUTPUT_DIR / MANUAL_LABELS_TIFF


def load_assignments_from_csv(path: Path) -> dict[int, int]:
    """Load ``label`` → ``manual_class`` from a saved CSV."""
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


def label_id_at_viewer_cursor(viewer: napari.Viewer, layer_name: str = CELL_LABELS_LAYER) -> int:
    """Instance label id under the current crosshair (0 if background)."""
    if layer_name not in viewer.layers:
        raise ValueError(f"Layer '{layer_name}' not found.")

    layer = viewer.layers[layer_name]
    if not isinstance(layer, napari.layers.Labels):
        raise TypeError(f"Layer '{layer_name}' is not a Labels layer.")

    data_pos = layer.world_to_data(viewer.dims.point)
    indices = tuple(int(round(v)) for v in data_pos)
    if len(indices) != layer.data.ndim:
        raise ValueError(f"Expected {layer.data.ndim}D position, got {indices}")

    for i, size in enumerate(layer.data.shape):
        if indices[i] < 0 or indices[i] >= size:
            return 0

    return int(layer.data[indices])


def _sync_manual_layer(viewer: napari.Viewer, state: ManualClassState) -> None:
    volume = state.class_volume()
    if VE_EPI_MANUAL_LAYER in viewer.layers:
        viewer.layers[VE_EPI_MANUAL_LAYER].data = volume
    else:
        viewer.add_labels(
            volume,
            name=VE_EPI_MANUAL_LAYER,
            opacity=0.55,
            blending="translucent",
        )


def _sync_label_features(viewer: napari.Viewer, state: ManualClassState) -> None:
    if CELL_LABELS_LAYER not in viewer.layers:
        return
    df = state.to_dataframe()
    features = compute_cell_features(state.instance_labels)
    merged = features.merge(df[["label", "manual_class", "class_name"]], on="label", how="left")
    viewer.layers[CELL_LABELS_LAYER].features = merged


def _print_counts(state: ManualClassState) -> None:
    c = state.counts()
    print(
        f"Manual labels: {c['ve']} VE, {c['epi']} EPI, "
        f"{c['unlabeled']} unlabeled / {c['total_cells']} cells"
    )


def setup_ve_epi_manual_viewer(
    *,
    label_path: Path | None = None,
    resume_csv: Path | None = None,
) -> tuple[napari.Viewer, ManualClassState]:
    """Load segment channels + instance labels; optional resume from CSV."""
    labels = load_label_volume(label_path)
    channels = load_segment_channels()

    ref = channels[min(channels)]
    if labels.shape != ref.shape:
        raise ValueError(
            f"Label shape {labels.shape} does not match segment shape {ref.shape}. "
            "Re-save labels from the same z-stack as step 1."
        )

    csv_path = resume_csv or default_manual_csv_path()
    assignments = load_assignments_from_csv(csv_path)
    if assignments:
        print(f"Resumed {len(assignments)} manual assignments from {csv_path.name}")

    state = ManualClassState(instance_labels=labels, assignments=assignments)

    viewer = napari.Viewer()
    apply_channels_to_viewer(viewer, channels)
    viewer.add_labels(labels, name=CELL_LABELS_LAYER)
    _sync_manual_layer(viewer, state)
    _sync_label_features(viewer, state)

    print(
        "Manual VE / EPI labeling:\n"
        f"  Segment + labels shape {labels.shape}\n"
        f"  Cells: {state.counts()['total_cells']}\n"
        "  Move crosshair onto a nucleus, then use the dock buttons.\n"
        "  Pick mode (keyboard 5) helps confirm the cell id in the status bar."
    )
    _print_counts(state)

    return viewer, state


def add_ve_epi_manual_widgets(viewer: napari.Viewer, state: ManualClassState) -> None:
    """Dock widgets for click-based VE assignment and bulk EPI."""

    def _assign(label_id: int, class_id: int, class_name: str) -> None:
        if label_id == 0:
            print("Click on a cell (non-zero label), not background.")
            return
        if label_id not in state.all_cell_ids():
            print(f"Label {label_id} is not a segmented cell id.")
            return
        state.assignments[label_id] = class_id
        _sync_manual_layer(viewer, state)
        _sync_label_features(viewer, state)
        print(f"Label {label_id} → {class_name}")
        _print_counts(state)

    @magicgui(call_button="Mark cell at crosshair as VE")
    def mark_ve() -> None:
        """Assign VE (class 1) to the instance under the crosshair."""
        _assign(label_id_at_viewer_cursor(viewer), CLASS_VE, "VE")

    @magicgui(call_button="Mark cell at crosshair as EPI")
    def mark_epi() -> None:
        """Assign EPI (class 2) to the instance under the crosshair."""
        _assign(label_id_at_viewer_cursor(viewer), CLASS_EPI, "EPI")

    @magicgui(call_button="Clear class for cell at crosshair")
    def clear_cell() -> None:
        """Remove VE/EPI assignment for the instance under the crosshair."""
        label_id = label_id_at_viewer_cursor(viewer)
        if label_id == 0:
            print("Click on a cell, not background.")
            return
        state.assignments.pop(label_id, None)
        _sync_manual_layer(viewer, state)
        _sync_label_features(viewer, state)
        print(f"Label {label_id} → unlabeled")
        _print_counts(state)

    @magicgui(call_button="Assign EPI to all remaining cells")
    def assign_epi_remaining() -> None:
        """Set EPI on every cell not already marked VE or EPI."""
        all_ids = state.all_cell_ids()
        to_epi = [
            lid
            for lid in all_ids
            if state.assignments.get(lid, CLASS_UNLABELED) == CLASS_UNLABELED
        ]
        if not to_epi:
            print("No unlabeled cells left.")
            return
        for label_id in to_epi:
            state.assignments[label_id] = CLASS_EPI
        _sync_manual_layer(viewer, state)
        _sync_label_features(viewer, state)
        print(f"Assigned EPI to {len(to_epi)} remaining cells.")
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
        path = default_manual_csv_path()
        state.assignments = load_assignments_from_csv(path)
        _sync_manual_layer(viewer, state)
        _sync_label_features(viewer, state)
        print(f"Reloaded from {path}")
        _print_counts(state)

    viewer.window.add_dock_widget(mark_ve)
    viewer.window.add_dock_widget(mark_epi)
    viewer.window.add_dock_widget(clear_cell)
    viewer.window.add_dock_widget(assign_epi_remaining)
    viewer.window.add_dock_widget(save_manual)
    viewer.window.add_dock_widget(reload_csv)
