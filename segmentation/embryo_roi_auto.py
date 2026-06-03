"""Step 1 (automatic): embryo ROI from DAPI + WNT channel fusion."""

from __future__ import annotations

from dataclasses import dataclass

import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import ball, closing, remove_small_objects

from image_io import (
    DAPI_CHANNEL,
    WNT_CHANNEL,
    apply_channels_to_viewer,
    load_project_channels,
)
from segmentation.embryo_roi import (
    EMBRYO_ROI_LABELS_LAYER,
    apply_embryo_roi_labels,
)


@dataclass(frozen=True)
class AutoEmbryoRoiParams:
    """Tunable parameters for DAPI + WNT fusion segmentation."""

    norm_percentile_lo: float = 1.0
    norm_percentile_hi: float = 99.5
    blur_sigma_z: float = 1.0
    blur_sigma_xy: float = 3.0
    otsu_scale_dapi: float = 0.85
    otsu_scale_wnt: float = 0.85
    closing_radius: int = 3
    min_object_size: int = 1000
    fill_holes: bool = True
    prefer_upper_component: bool = True


def normalize_percentile(
    volume: np.ndarray,
    lo: float = 1.0,
    hi: float = 99.5,
) -> np.ndarray:
    """Clip by percentiles and scale to [0, 1]."""
    low, high = np.percentile(volume, [lo, hi])
    if high <= low:
        return np.zeros_like(volume, dtype=np.float32)
    scaled = (volume.astype(np.float32) - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0)


def _otsu_threshold(volume: np.ndarray, scale: float) -> float:
    """Otsu threshold with scale factor; fall back when Otsu is ill-defined."""
    flat = volume.ravel()
    if flat.size == 0 or flat.max() <= flat.min():
        return 1.0
    positive = flat[flat > 0.01]
    if positive.size < 32:
        return float(np.max(positive)) if positive.size else 1.0
    return float(threshold_otsu(positive) * scale)


def _keep_embryo_component(
    mask: np.ndarray,
    *,
    prefer_upper: bool,
) -> np.ndarray:
    """Keep one 3D connected component (largest, or uppermost among large blobs)."""
    labeled = label(mask)
    if labeled.max() == 0:
        return mask

    regions = regionprops(labeled)
    if not prefer_upper or len(regions) == 1:
        keep_label = max(regions, key=lambda r: r.area).label
        return labeled == keep_label

    areas = np.array([r.area for r in regions], dtype=np.int64)
    area_cutoff = max(areas.max() // 10, 1)
    candidates = [r for r in regions if r.area >= area_cutoff]
    if not candidates:
        candidates = regions

    # Image y increases downward; smaller centroid y -> upper structure.
    keep_label = min(candidates, key=lambda r: r.centroid[1]).label
    return labeled == keep_label


def segment_embryo_roi_auto(
    dapi: np.ndarray,
    wnt: np.ndarray,
    params: AutoEmbryoRoiParams | None = None,
) -> np.ndarray:
    """
    Build a binary embryo ROI mask (Z, Y, X) from DAPI and WNT volumes.

    Pipeline: normalize → blur → per-channel Otsu → union → close →
    remove small objects → keep embryo component → optional hole fill.
    """
    if dapi.shape != wnt.shape:
        raise ValueError(f"Channel shape mismatch: DAPI {dapi.shape}, WNT {wnt.shape}")

    params = params or AutoEmbryoRoiParams()

    dapi_n = normalize_percentile(
        dapi, params.norm_percentile_lo, params.norm_percentile_hi
    )
    wnt_n = normalize_percentile(
        wnt, params.norm_percentile_lo, params.norm_percentile_hi
    )

    sigma = (params.blur_sigma_z, params.blur_sigma_xy, params.blur_sigma_xy)
    dapi_blur = ndi.gaussian_filter(dapi_n, sigma=sigma)
    wnt_blur = ndi.gaussian_filter(wnt_n, sigma=sigma)

    t_dapi = _otsu_threshold(dapi_blur, params.otsu_scale_dapi)
    t_wnt = _otsu_threshold(wnt_blur, params.otsu_scale_wnt)
    mask = (dapi_blur > t_dapi) | (wnt_blur > t_wnt)

    if params.closing_radius > 0:
        mask = closing(mask, ball(params.closing_radius))

    if params.min_object_size > 0:
        mask = remove_small_objects(mask, max_size=params.min_object_size - 1)

    mask = _keep_embryo_component(
        mask, prefer_upper=params.prefer_upper_component
    )

    if params.fill_holes:
        mask = ndi.binary_fill_holes(mask)

    if not mask.any():
        raise ValueError(
            "Automatic embryo ROI is empty. Loosen thresholds or blur settings."
        )

    return mask


def _channel_volume(viewer: napari.Viewer, channel_idx: int) -> np.ndarray:
    layer_name = f"channel {channel_idx}"
    if layer_name not in viewer.layers:
        raise ValueError(
            f"Layer '{layer_name}' not found. Load image channels first."
        )
    layer = viewer.layers[layer_name]
    if not isinstance(layer, Image):
        raise TypeError(f"Layer '{layer_name}' is not an image layer.")
    return np.asarray(layer.data)


def build_embryo_roi_auto(
    viewer: napari.Viewer,
    params: AutoEmbryoRoiParams | None = None,
) -> np.ndarray:
    """Run automatic segmentation and update the ``embryo_roi`` labels layer."""
    dapi = _channel_volume(viewer, DAPI_CHANNEL)
    wnt = _channel_volume(viewer, WNT_CHANNEL)
    mask = segment_embryo_roi_auto(dapi, wnt, params)
    apply_embryo_roi_labels(viewer, mask)
    print(
        f"{EMBRYO_ROI_LABELS_LAYER} (auto): shape {mask.shape}, "
        f"{int(mask.sum())} voxels inside ROI"
    )
    return mask


def setup_embryo_roi_auto_viewer(viewer: napari.Viewer) -> None:
    """Load channels for automatic embryo ROI segmentation."""
    apply_channels_to_viewer(viewer, load_project_channels())
    print(
        "Step 1 — Automatic embryo ROI:\n"
        f"  channel {WNT_CHANNEL}: WNT (membranes)\n"
        f"  channel {DAPI_CHANNEL}: DAPI (nuclei)\n"
        "  Adjust parameters in the dock widget, then click "
        "'Build embryo ROI (auto)'."
    )


@magicgui(call_button="Reload images from disk")
def reload_images_auto_widget(viewer: napari.Viewer) -> None:
    """Reload TIFF channels from 22A_E1_Wnt3."""
    apply_channels_to_viewer(viewer, load_project_channels())


@magicgui(
    call_button="Build embryo ROI (auto)",
    blur_sigma_z={"min": 0.0, "max": 10.0, "step": 0.5},
    blur_sigma_xy={"min": 0.0, "max": 15.0, "step": 0.5},
    otsu_scale_dapi={"min": 0.1, "max": 1.5, "step": 0.05},
    otsu_scale_wnt={"min": 0.1, "max": 1.5, "step": 0.05},
    closing_radius={"min": 0, "max": 10},
    min_object_size={"min": 0, "max": 50000, "step": 100},
)
def build_embryo_roi_auto_widget(
    viewer: napari.Viewer,
    blur_sigma_z: float = 1.0,
    blur_sigma_xy: float = 3.0,
    otsu_scale_dapi: float = 0.85,
    otsu_scale_wnt: float = 0.85,
    closing_radius: int = 3,
    min_object_size: int = 1000,
    fill_holes: bool = True,
    prefer_upper_component: bool = True,
) -> None:
    """Fuse DAPI + WNT, threshold, and keep the embryo blob."""
    params = AutoEmbryoRoiParams(
        blur_sigma_z=blur_sigma_z,
        blur_sigma_xy=blur_sigma_xy,
        otsu_scale_dapi=otsu_scale_dapi,
        otsu_scale_wnt=otsu_scale_wnt,
        closing_radius=closing_radius,
        min_object_size=min_object_size,
        fill_holes=fill_holes,
        prefer_upper_component=prefer_upper_component,
    )
    build_embryo_roi_auto(viewer, params)


def add_embryo_roi_auto_widgets(viewer: napari.Viewer) -> None:
    """Add dock widgets for reload and automatic ROI building."""
    viewer.window.add_dock_widget(reload_images_auto_widget)
    viewer.window.add_dock_widget(build_embryo_roi_auto_widget)
