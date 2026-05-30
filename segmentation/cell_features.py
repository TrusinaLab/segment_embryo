"""Per-cell morphological and spatial features for VE / EPI classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, distance_transform_edt

CELL_LABELS_LAYER = "cell_labels"
VE_EPI_PREDICTIONS_LAYER = "ve_epi_predictions"
DEFAULT_ROI_ID = "embryo1"


@dataclass(frozen=True)
class CellFeatureParams:
    """Voxel scaling and optional mask for embryo geometry."""

    z_spacing: float = 1.0
    xy_spacing: float = 1.0
    embryo_mask: np.ndarray | None = None


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
    if params.embryo_mask is not None:
        mask = np.asarray(params.embryo_mask, dtype=bool)
    else:
        mask = labels > 0

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
    union = labels > 0
    edt = distance_transform_edt(union)

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
