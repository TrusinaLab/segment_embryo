"""Per-cell morphological and spatial features for VE / EPI classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, center_of_mass, distance_transform_edt
from skimage.morphology import ball, closing, dilation

CELL_LABELS_LAYER = "cell_labels"
VE_EPI_PREDICTIONS_LAYER = "ve_epi_predictions"
DEFAULT_ROI_ID = "embryo1"
DEFAULT_PAD_CELLS_RADIUS = 6


@dataclass(frozen=True)
class CellFeatureParams:
    """Voxel scaling and optional mask for embryo geometry."""

    z_spacing: float = 1.0
    xy_spacing: float = 1.0
    embryo_mask: np.ndarray | None = None
    """If set, dilate each cell by this radius (voxels), merge, then fill enclosed holes."""
    pad_cells_radius: int | None = None
    fill_embryo_holes: bool = True


def cell_union_mask(labels: np.ndarray) -> np.ndarray:
    """Binary mask of all segmented cell voxels (no padding)."""
    return np.asarray(labels) > 0


def _fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill enclosed voids in 2D (each z-slice) and 3D, then lightly close narrow gaps.

    ``binary_fill_holes`` only fills cavities that do not connect to the background.
    Gaps between cells in the Y–X view are often open to the image border; a small
    closing step bridges those after all slice-wise and volumetric hole fills.
    """
    filled = mask.copy()
    for z in range(filled.shape[0]):
        filled[z] = binary_fill_holes(filled[z])
    filled = binary_fill_holes(filled)
    return closing(filled, ball(2))


def _dilate_cell_bbox(
    cell: np.ndarray, footprint: np.ndarray, margin: int, out: np.ndarray
) -> None:
    """Dilate one cell inside its bounding box only (much faster than full volume)."""
    coords = np.argwhere(cell)
    if coords.size == 0:
        return
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    pad = margin + 1
    z0 = max(z0 - pad, 0)
    y0 = max(y0 - pad, 0)
    x0 = max(x0 - pad, 0)
    z1 = min(z1 + pad, cell.shape[0])
    y1 = min(y1 + pad, cell.shape[1])
    x1 = min(x1 + pad, cell.shape[2])
    slab = cell[z0:z1, y0:y1, x0:x1]
    out[z0:z1, y0:y1, x0:x1] |= dilation(slab, footprint)


def padded_embryo_mask_from_cells(
    labels: np.ndarray,
    radius: int,
    *,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Dilate each label separately (bounding-box crops), union, then fill holes.

    Hole fill: per-z ``binary_fill_holes``, 3D ``binary_fill_holes``, then
    ``binary_closing`` (ball r=2) for narrow gaps still open to the border.
    """
    labels = np.asarray(labels)
    if radius < 1:
        merged = cell_union_mask(labels)
    else:
        footprint = ball(radius)
        merged = np.zeros(labels.shape, dtype=bool)
        for label_id in np.unique(labels):
            if label_id == 0:
                continue
            _dilate_cell_bbox(labels == label_id, footprint, radius, merged)

    if fill_holes and merged.any():
        merged = _fill_enclosed_holes(merged)

    return merged


def embryo_cup_mask_from_cells(
    labels: np.ndarray,
    pad_radius: int = DEFAULT_PAD_CELLS_RADIUS,
    *,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    3D binary mask separating the embryo cup (True) from background (False).

    Built by dilating each cell, merging, then hole fill + closing — same as the
    padded mask used for VE/EPI surface features.
    """
    return padded_embryo_mask_from_cells(labels, pad_radius, fill_holes=fill_holes)


def embryo_cup_labels(mask: np.ndarray) -> np.ndarray:
    """Label volume: 0 = background, 1 = inside embryo cup."""
    return np.asarray(mask, dtype=np.uint8)


def reference_embryo_mask(labels: np.ndarray, params: CellFeatureParams) -> np.ndarray:
    """Mask used for embryo COM, surface EDT, and radial alignment."""
    if params.embryo_mask is not None:
        return np.asarray(params.embryo_mask, dtype=bool)
    if params.pad_cells_radius is not None:
        return padded_embryo_mask_from_cells(
            labels,
            params.pad_cells_radius,
            fill_holes=params.fill_embryo_holes,
        )
    return cell_union_mask(labels)


def _scale_coordinates(coords_zyx: np.ndarray, params: CellFeatureParams) -> np.ndarray:
    """Physical-ish coordinates: Z scaled by z_spacing, Y/X by xy_spacing."""
    scaled = coords_zyx.astype(np.float64, copy=True)
    scaled[:, 0] *= params.z_spacing
    scaled[:, 1] *= params.xy_spacing
    scaled[:, 2] *= params.xy_spacing
    return scaled


def _pca_shape_metrics(coords_scaled: np.ndarray) -> dict[str, float | np.ndarray]:
    """PCA on cell voxels; return shape metrics and major axis unit vector."""
    if coords_scaled.shape[0] < 4:
        nan = float("nan")
        return {
            "elongation": nan,
            "flatness": nan,
            "sphericity": nan,
            "major_axis_x": nan,
            "major_axis_y": nan,
            "major_axis_z": nan,
        }

    centered = coords_scaled - coords_scaled.mean(axis=0)
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 1e-12)
    major = evecs[:, order[0]]
    major = major / (np.linalg.norm(major) + 1e-12)

    elongation = float(np.sqrt(evals[0] / evals[1]))
    flatness = float(np.sqrt(evals[1] / evals[2]))
    sphericity = float(evals[2] / evals[0])

    return {
        "elongation": elongation,
        "flatness": flatness,
        "sphericity": sphericity,
        "major_axis_z": float(major[0]),
        "major_axis_y": float(major[1]),
        "major_axis_x": float(major[2]),
    }


def _embryo_center_of_mass(
    labels: np.ndarray, params: CellFeatureParams
) -> np.ndarray:
    """Center of mass of embryo tissue (scaled coordinates)."""
    mask = reference_embryo_mask(labels, params)

    if not mask.any():
        raise ValueError("Cannot compute embryo center of mass: empty mask.")

    com_zyx = np.array(center_of_mass(mask), dtype=np.float64)
    scaled = com_zyx.copy()
    scaled[0] *= params.z_spacing
    scaled[1] *= params.xy_spacing
    scaled[2] *= params.xy_spacing
    return scaled


def _outward_normal_at_point(
    edt: np.ndarray, point_zyx: np.ndarray
) -> np.ndarray:
    """Unit vector pointing toward increasing distance (outward from tissue)."""
    z, y, x = np.clip(np.round(point_zyx).astype(int), 0, np.array(edt.shape) - 1)
    gz, gy, gx = np.gradient(edt.astype(np.float64))
    grad = np.array([gz[z, y, x], gy[z, y, x], gx[z, y, x]], dtype=np.float64)
    norm = np.linalg.norm(grad)
    if norm < 1e-8:
        return np.array([np.nan, np.nan, np.nan])
    return grad / norm


def compute_cell_features(
    labels: np.ndarray,
    params: CellFeatureParams | None = None,
    roi_id: str = DEFAULT_ROI_ID,
) -> pd.DataFrame:
    """
    Build a feature table with one row per label ID (background excluded).

    Features include morphological PCA metrics, distance from embryo center of
    mass, distance from the tissue surface (EDT), and radial alignment.
    """
    params = params or CellFeatureParams()
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected 3D label volume (Z, Y, X), got shape {labels.shape}")

    embryo_com = _embryo_center_of_mass(labels, params)
    tissue = reference_embryo_mask(labels, params)
    edt = distance_transform_edt(tissue)

    rows: list[dict[str, float | int | str]] = []
    for label_id in np.unique(labels):
        if label_id == 0:
            continue

        coords = np.argwhere(labels == label_id)
        n_voxels = int(coords.shape[0])
        coords_scaled = _scale_coordinates(coords, params)
        centroid = coords_scaled.mean(axis=0)

        dist_embryo_com = float(np.linalg.norm(centroid - embryo_com))

        shape = _pca_shape_metrics(coords_scaled)
        major_axis = np.array(
            [shape["major_axis_z"], shape["major_axis_y"], shape["major_axis_x"]]
        )

        centroid_voxel = coords.mean(axis=0)
        dist_surface = float(
            edt[
                int(np.clip(round(centroid_voxel[0]), 0, edt.shape[0] - 1)),
                int(np.clip(round(centroid_voxel[1]), 0, edt.shape[1] - 1)),
                int(np.clip(round(centroid_voxel[2]), 0, edt.shape[2] - 1)),
            ]
        )

        outward = _outward_normal_at_point(edt, centroid_voxel)
        if np.isnan(outward).any():
            radial_alignment = float("nan")
        else:
            outward_scaled = outward.copy()
            outward_scaled[0] *= params.z_spacing
            outward_scaled[1] *= params.xy_spacing
            outward_scaled[2] *= params.xy_spacing
            outward_scaled /= np.linalg.norm(outward_scaled) + 1e-12
            radial_alignment = float(abs(np.dot(major_axis, outward_scaled)))

        rows.append(
            {
                "label": int(label_id),
                "roi_id": roi_id,
                "n_voxels": n_voxels,
                "volume": float(n_voxels * params.z_spacing * params.xy_spacing**2),
                "centroid_z": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_x": float(centroid[2]),
                "distance_from_embryo_com": dist_embryo_com,
                "distance_from_surface": dist_surface,
                "radial_alignment": radial_alignment,
                "elongation": shape["elongation"],
                "flatness": shape["flatness"],
                "sphericity": shape["sphericity"],
            }
        )

    df = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    df.attrs["embryo_com_zyx_scaled"] = embryo_com
    df.attrs["embryo_mask_voxels"] = int(tissue.sum())
    return df


FEATURE_COLUMNS = [
    "distance_from_embryo_com",
    "distance_from_surface",
    "radial_alignment",
    "elongation",
    "flatness",
    "sphericity",
    "volume",
    "n_voxels",
]


def minimal_label_features_table(labels: np.ndarray) -> pd.DataFrame:
    """
    One row per instance ID for napari hover text.

    Napari matches rows via an ``index`` column (label value), not the DataFrame index.
    """
    ids = np.unique(labels)
    ids = ids[ids != 0].astype(int)
    if ids.size == 0:
        return pd.DataFrame(columns=["index", "label", "cell_id", "name"])
    return pd.DataFrame(
        {
            "index": ids,
            "label": ids,
            "cell_id": ids,
            "name": [f"cell {i}" for i in ids],
        }
    )


def features_for_label_layer(
    labels: np.ndarray,
    extra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a napari-compatible features table for all non-zero labels in ``labels``.

    Ensures every present ID has ``index``, ``label``, ``cell_id``, and ``name``.
    """
    present = sorted(set(np.unique(labels).astype(int)) - {0})
    if not present:
        return minimal_label_features_table(labels)

    if extra is not None and not extra.empty and "label" in extra.columns:
        df = extra.copy()
        if "index" not in df.columns:
            df["index"] = df["label"].astype(int)
        df = df[df["index"].astype(int).isin(present)]
        known = set(df["index"].astype(int))
        for label_id in present:
            if label_id not in known:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [
                                {
                                    "index": label_id,
                                    "label": label_id,
                                    "cell_id": label_id,
                                    "name": f"cell {label_id}",
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
        if "cell_id" not in df.columns:
            df["cell_id"] = df["index"].astype(int)
        if "name" not in df.columns:
            df["name"] = df["index"].astype(int).map(lambda i: f"cell {i}")
        return df.reset_index(drop=True)

    return minimal_label_features_table(labels)


def attach_label_hover_features(
    layer: napari.layers.Labels,
    features: pd.DataFrame | None = None,
) -> None:
    """Attach per-label metadata so label IDs show on mouse hover."""
    df = features_for_label_layer(layer.data, features)
    layer.features = df


def attach_label_hover_status(
    layer: napari.layers.Labels,
    viewer: napari.Viewer,
) -> None:
    """Show instance label id in the viewer status bar while moving the cursor."""

    @layer.mouse_move_callbacks.append
    def _on_move(layer, event) -> None:
        value = layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True,
        )
        if value and int(value) > 0:
            lid = int(value)
            viewer.status = f"cell label: {lid}"
        elif getattr(viewer, "status", None) != "ready":
            viewer.status = "ready"


def save_features_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def predictions_to_label_volume(
    labels: np.ndarray, df: pd.DataFrame, class_column: str = "prediction"
) -> np.ndarray:
    """Map per-label class IDs onto a new label image (0 = background)."""
    mapping = dict(zip(df["label"].astype(int), df[class_column].astype(int)))
    out = np.zeros_like(labels, dtype=np.uint16)
    for label_id, class_id in mapping.items():
        if class_id == 0:
            continue
        out[labels == label_id] = class_id
    return out
