"""
View cell labels in Napari colored by a numeric feature from a features CSV.

Loads labels from ``data/test_cell_labels/``, optional segment channels, and
maps each label to a colormap value (default feature: ``radial_alignment``).
Use the dock widget or ``--feature`` to color by any numeric column in the CSV.

Run (micro-sam-napari conda env):
    conda activate micro-sam-napari
    python run_view_radial_alignment.py
    python run_view_radial_alignment.py --feature elongation
    python run_view_radial_alignment.py --list-features

Or:
    scripts\\run_view_radial_alignment.bat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from magicgui import magicgui
from napari.utils.colormaps import DirectLabelColormap

from image_io import (
    EPI_VE_OUTPUT_DIR,
    apply_channels_to_viewer,
    viewer_add_labels,
    discover_label_volume_path,
    load_label_volume,
    load_middle_z_channels,
    load_segment_channels,
    project_root,
)
from segmentation.cell_features import CELL_LABELS_LAYER, FEATURE_COLUMNS

DEFAULT_CMAP = "viridis"
DEFAULT_FEATURE = "radial_alignment"
MISSING_RGBA = (0.55, 0.55, 0.55, 1.0)
BACKGROUND_RGBA = (0.0, 0.0, 0.0, 0.0)
NON_FEATURE_COLUMNS = frozenset({"label", "roi_id", "class_name", "prediction", "manual_class"})


def load_features_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"CSV must contain a 'label' column: {path}")
    return df


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric CSV columns suitable for label coloring (stable order)."""
    from_csv = [c for c in FEATURE_COLUMNS if c in df.columns]
    extra = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS
        and c not in from_csv
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return from_csv + sorted(extra)


def feature_value_range(
    features: pd.DataFrame,
    column: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float]:
    values = features[column].dropna()
    if values.empty:
        raise ValueError(f"No valid values for feature '{column}'.")
    lo = float(values.min()) if vmin is None else vmin
    hi = float(values.max()) if vmax is None else vmax
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def feature_colormap(
    features: pd.DataFrame,
    column: str,
    *,
    cmap_name: str = DEFAULT_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[DirectLabelColormap, float, float]:
    """Map each label id to an RGBA color from a numeric feature column."""
    import matplotlib.pyplot as plt

    if column not in features.columns:
        raise ValueError(f"Feature '{column}' not in CSV. Available: {numeric_feature_columns(features)}")
    if not pd.api.types.is_numeric_dtype(features[column]):
        raise ValueError(f"Feature '{column}' is not numeric.")

    lo, hi = feature_value_range(features, column, vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    span = hi - lo

    color_dict: dict[int | None, tuple[float, float, float, float]] = {
        0: BACKGROUND_RGBA,
        None: MISSING_RGBA,
    }

    for _, row in features.iterrows():
        label = int(row["label"])
        value = row[column]
        if pd.isna(value):
            color_dict[label] = MISSING_RGBA
        else:
            t = float(np.clip((value - lo) / span, 0.0, 1.0))
            rgba = cmap(t)
            color_dict[label] = (
                float(rgba[0]),
                float(rgba[1]),
                float(rgba[2]),
                float(rgba[3]),
            )

    return DirectLabelColormap(color_dict=color_dict), lo, hi


def _load_channels(use_segment: bool) -> dict[int, np.ndarray] | None:
    if not use_segment:
        return None
    try:
        return load_segment_channels()
    except FileNotFoundError:
        print("Segment TIFFs not found; falling back to raw middle z-subset.")
        return load_middle_z_channels()


def add_feature_color_widgets(
    viewer: napari.Viewer,
    features: pd.DataFrame,
    *,
    initial_feature: str,
    cmap_name: str,
) -> None:
    choices = numeric_feature_columns(features)
    state = {"cmap_name": cmap_name}

    @magicgui(
        color_by={"choices": choices, "value": initial_feature, "label": "Color by"},
        cmap_name={"choices": ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"]},
        call_button="Apply colors",
    )
    def color_by_feature(color_by: str, cmap_name: str = DEFAULT_CMAP) -> None:
        colormap, lo, hi = feature_colormap(features, color_by, cmap_name=cmap_name)
        state["cmap_name"] = cmap_name
        layer = viewer.layers[CELL_LABELS_LAYER]
        layer.colormap = colormap
        n_valid = features[color_by].notna().sum()
        print(
            f"Colored by {color_by} ({cmap_name}), range [{lo:.4g}, {hi:.4g}]; "
            f"{n_valid}/{len(features)} cells with values; gray = missing."
        )

    viewer.window.add_dock_widget(color_by_feature)


def setup_feature_colored_viewer(
    features_csv: Path,
    *,
    color_by: str = DEFAULT_FEATURE,
    use_segment_channels: bool = True,
    cmap_name: str = DEFAULT_CMAP,
) -> napari.Viewer:
    labels = load_label_volume()
    features = load_features_csv(features_csv)
    choices = numeric_feature_columns(features)
    if color_by not in choices:
        raise ValueError(
            f"Unknown feature '{color_by}'. Choose one of: {', '.join(choices)}"
        )

    colormap, lo, hi = feature_colormap(features, color_by, cmap_name=cmap_name)

    viewer = napari.Viewer()
    channels = _load_channels(use_segment_channels)
    if channels is not None:
        if labels.shape != channels[min(channels)].shape:
            raise ValueError(
                f"Label shape {labels.shape} does not match image shape "
                f"{channels[min(channels)].shape}."
            )
        apply_channels_to_viewer(viewer, channels)

    viewer_add_labels(
        viewer,
        labels,
        name=CELL_LABELS_LAYER,
        colormap=colormap,
        features=features,
        opacity=0.65,
    )

    add_feature_color_widgets(
        viewer, features, initial_feature=color_by, cmap_name=cmap_name
    )

    present = set(np.unique(labels)) - {0}
    in_csv = set(features["label"].astype(int))
    missing_features = sorted(present - in_csv)
    if missing_features:
        print(f"Labels without CSV row (shown gray): {missing_features[:20]}", end="")
        if len(missing_features) > 20:
            print(f" ... +{len(missing_features) - 20} more")
        else:
            print()

    n_valid = features[color_by].notna().sum()
    print(
        f"Label colors: {color_by} ({cmap_name}), range [{lo:.4g}, {hi:.4g}]; "
        f"gray = missing.\n"
        f"  Dock: change 'Color by' + Apply colors\n"
        f"  Numeric features: {', '.join(choices)}\n"
        f"  Labels layer: {CELL_LABELS_LAYER}\n"
        f"  Cells in CSV: {len(features)} ({n_valid} with {color_by})\n"
        f"  Features CSV: {features_csv}"
    )
    return viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=project_root() / EPI_VE_OUTPUT_DIR / "cell_features.csv",
        help="Features table with a label column and numeric feature columns",
    )
    parser.add_argument(
        "--feature",
        default=DEFAULT_FEATURE,
        help=f"Initial feature to color by (default: {DEFAULT_FEATURE})",
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="Print numeric feature columns from the CSV and exit",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Show labels only (no segment/raw channels)",
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        help=f"Matplotlib colormap name (default: {DEFAULT_CMAP})",
    )
    args = parser.parse_args()

    features = load_features_csv(args.csv)
    choices = numeric_feature_columns(features)

    if args.list_features:
        print(f"Numeric features in {args.csv}:")
        for name in choices:
            valid = features[name].notna().sum()
            lo = features[name].min()
            hi = features[name].max()
            print(f"  {name}: {valid} values, range [{lo:.4g}, {hi:.4g}]")
        return

    label_path = discover_label_volume_path()
    viewer = setup_feature_colored_viewer(
        args.csv,
        color_by=args.feature,
        use_segment_channels=not args.no_images,
        cmap_name=args.cmap,
    )
    print(f"  Label volume: {label_path}")
    napari.run()


if __name__ == "__main__":
    main()
