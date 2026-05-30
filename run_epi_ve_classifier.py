"""
VE / EPI cell classification — feature table + Napari training.

Loads cell labels from ``data/test_cell_labels/``, masked segment channels (or raw
middle z-subset), computes morphological and spatial features, and attaches them
to the label layer for napari-feature-classifier or in-app Random Forest training.

Run (micro-sam-napari conda env):
    conda activate micro-sam-napari
    python run_epi_ve_classifier.py

See ``notes/epi_ve_workflow.md`` and ``notes/EPI_VE_classifier.md``.
"""

from __future__ import annotations

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
    load_middle_z_channels,
    load_segment_channels,
    project_root,
)
from segmentation.cell_features import (
    CELL_LABELS_LAYER,
    DEFAULT_PAD_CELLS_RADIUS,
    FEATURE_COLUMNS,
    VE_EPI_PREDICTIONS_LAYER,
    CellFeatureParams,
    compute_cell_features,
    predictions_to_label_volume,
    save_features_csv,
)

CLASS_VE = 1
CLASS_EPI = 2


def _load_channels(use_segment: bool) -> dict[int, np.ndarray]:
    if use_segment:
        try:
            return load_segment_channels()
        except FileNotFoundError:
            print("Segment TIFFs not found; falling back to raw middle z-subset.")
    return load_middle_z_channels()


def _attach_features(label_layer: napari.layers.Labels, df: pd.DataFrame) -> None:
    label_layer.features = df.copy()


def _shapes_match(labels: np.ndarray, channels: dict[int, np.ndarray]) -> bool:
    ref = channels[min(channels)]
    return labels.shape == ref.shape


def setup_ve_epi_viewer(
    use_segment_channels: bool = True,
    z_spacing: float = 1.0,
    xy_spacing: float = 1.0,
    pad_cells_radius: int = DEFAULT_PAD_CELLS_RADIUS,
) -> tuple[napari.Viewer, np.ndarray, pd.DataFrame]:
    labels = load_label_volume()
    channels = _load_channels(use_segment_channels)

    if not _shapes_match(labels, channels):
        raise ValueError(
            f"Label shape {labels.shape} does not match image shape "
            f"{channels[min(channels)].shape}. Reload the same z-range used for segmentation."
        )

    params = CellFeatureParams(
        z_spacing=z_spacing,
        xy_spacing=xy_spacing,
        pad_cells_radius=pad_cells_radius,
    )
    features = compute_cell_features(labels, params=params)

    viewer = napari.Viewer()
    apply_channels_to_viewer(viewer, channels)
    label_layer = viewer.add_labels(labels, name=CELL_LABELS_LAYER)
    _attach_features(label_layer, features)

    com = features.attrs.get("embryo_com_zyx_scaled")
    mask_voxels = features.attrs.get("embryo_mask_voxels")
    if com is not None:
        print(f"Embryo center of mass (scaled z,y,x): {com}")
    if mask_voxels is not None:
        print(
            f"Embryo reference mask: per-cell dilation r={pad_cells_radius}, "
            f"holes filled, {mask_voxels:,} voxels"
        )

    return viewer, labels, features


def add_ve_epi_widgets(
    viewer: napari.Viewer,
    labels: np.ndarray,
    features: pd.DataFrame,
) -> None:
    state = {"features": features, "labels": labels}

    @magicgui(
        z_spacing={"min": 0.01, "max": 10.0, "step": 0.01},
        xy_spacing={"min": 0.01, "max": 10.0, "step": 0.01},
        pad_cells_radius={"min": 0, "max": 20, "step": 1},
        call_button="Recompute features",
    )
    def recompute_features(
        z_spacing: float = 1.0,
        xy_spacing: float = 1.0,
        pad_cells_radius: int = DEFAULT_PAD_CELLS_RADIUS,
    ) -> None:
        params = CellFeatureParams(
            z_spacing=z_spacing,
            xy_spacing=xy_spacing,
            pad_cells_radius=pad_cells_radius or None,
        )
        df = compute_cell_features(state["labels"], params=params)
        state["features"] = df
        layer = viewer.layers[CELL_LABELS_LAYER]
        _attach_features(layer, df)
        print(f"Recomputed features for {len(df)} cells.")

    @magicgui(call_button="Save features CSV")
    def save_features() -> None:
        out = project_root() / EPI_VE_OUTPUT_DIR / "cell_features.csv"
        save_features_csv(state["features"], out)
        print(f"Saved {out}")

    @magicgui(
        training_csv={
            "widget_type": "FileEdit",
            "mode": "r",
            "label": "Training CSV (label, manual_class)",
        },
        call_button="Train RF & predict",
    )
    def train_and_predict(training_csv: str = "") -> None:
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as exc:
            raise ImportError(
                "Install scikit-learn: conda install -c conda-forge scikit-learn"
            ) from exc

        csv_path = Path(training_csv) if training_csv else None
        if csv_path is None or not csv_path.is_file():
            raise ValueError(
                "Provide a CSV with columns: label, manual_class "
                "(use 1=VE, 2=EPI). Annotate examples in napari-feature-classifier "
                "or edit the saved features CSV."
            )

        train_df = pd.read_csv(csv_path)
        if "label" not in train_df.columns or "manual_class" not in train_df.columns:
            raise ValueError("Training CSV must contain 'label' and 'manual_class'.")

        merged = state["features"].merge(
            train_df[["label", "manual_class"]], on="label", how="left"
        )
        train_rows = merged.dropna(subset=["manual_class"])
        if train_rows["manual_class"].nunique() < 2:
            raise ValueError("Need at least two classes in manual_class for training.")

        cols = [c for c in FEATURE_COLUMNS if c in train_rows.columns]
        X = train_rows[cols].fillna(train_rows[cols].median())
        y = train_rows["manual_class"].astype(int)

        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0)
        clf.fit(X, y)

        all_X = merged[cols].fillna(merged[cols].median())
        merged["prediction"] = clf.predict(all_X)
        merged["class_name"] = merged["prediction"].map({CLASS_VE: "VE", CLASS_EPI: "EPI"})
        state["features"] = merged
        _attach_features(viewer.layers[CELL_LABELS_LAYER], merged)

        pred_vol = predictions_to_label_volume(state["labels"], merged, "prediction")
        if VE_EPI_PREDICTIONS_LAYER in viewer.layers:
            viewer.layers[VE_EPI_PREDICTIONS_LAYER].data = pred_vol
        else:
            viewer.add_labels(pred_vol, name=VE_EPI_PREDICTIONS_LAYER)

        out = project_root() / EPI_VE_OUTPUT_DIR / "cell_features_predicted.csv"
        save_features_csv(merged, out)
        print(f"Predicted {len(merged)} cells; saved {out}")

    viewer.window.add_dock_widget(recompute_features)
    viewer.window.add_dock_widget(save_features)
    viewer.window.add_dock_widget(train_and_predict)


def main() -> None:
    label_path = discover_label_volume_path()
    viewer, labels, features = setup_ve_epi_viewer()
    add_ve_epi_widgets(viewer, labels, features)

    print(
        "VE / EPI classifier:\n"
        f"  Labels: {label_path}\n"
        f"  Cells: {len(features)}\n"
        f"  Features: {', '.join(FEATURE_COLUMNS)}\n"
        "\n"
        "Interactive training (recommended):\n"
        "  Plugins → napari-feature-classifier → Initialize (classes VE, EPI)\n"
        "  Click example cells → Run Classifier\n"
        "\n"
        "Or: add manual_class (1=VE, 2=EPI) to a CSV → Train RF & predict dock.\n"
        f"\nOutput folder: {EPI_VE_OUTPUT_DIR.as_posix()}/\n"
        f"Label dir: {CELL_LABELS_DIR.as_posix()}/"
    )
    napari.run()


if __name__ == "__main__":
    main()
