"""Step 1: draw embryo outline and rasterize to a 3D ROI mask."""

from __future__ import annotations

from pathlib import Path

import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Image, Labels, Shapes

from image_io import apply_channels_to_viewer, load_project_channels

EMBRYO_OUTLINE_LAYER = "embryo_outline"
EMBRYO_ROI_LABELS_LAYER = "embryo_roi"
REFERENCE_IMAGE_LAYER = "channel 1"


def ensure_embryo_outline_layer(viewer: napari.Viewer) -> Shapes:
    """Create an empty shapes layer for the embryo polygon if needed."""
    if EMBRYO_OUTLINE_LAYER in viewer.layers:
        return viewer.layers[EMBRYO_OUTLINE_LAYER]

    return viewer.add_shapes(
        name=EMBRYO_OUTLINE_LAYER,
        ndim=3,
        edge_color="lime",
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


def rasterize_embryo_outline(viewer: napari.Viewer) -> np.ndarray:
    """
    Convert polygons in ``embryo_outline`` to a binary 3D mask (Z, Y, X).

    Draw closed polygons on individual z-slices (or 3D polygons) before
    calling this function.
    """
    if EMBRYO_OUTLINE_LAYER not in viewer.layers:
        raise ValueError(
            f"Add shapes to layer '{EMBRYO_OUTLINE_LAYER}' before building the ROI."
        )

    shapes_layer = viewer.layers[EMBRYO_OUTLINE_LAYER]
    if len(shapes_layer.data) == 0:
        raise ValueError(
            f"Layer '{EMBRYO_OUTLINE_LAYER}' is empty. "
            "Draw at least one polygon around the embryo."
        )

    ref = _reference_image_layer(viewer)
    mask_shape = ref.data.shape

    # One boolean mask per shape; union -> single embryo ROI.
    masks = shapes_layer.to_masks(mask_shape=mask_shape)
    embryo_mask = np.any(masks, axis=0)

    if not embryo_mask.any():
        raise ValueError("Embryo mask is empty. Check that polygons cover the embryo.")

    return embryo_mask


def apply_embryo_roi_labels(viewer: napari.Viewer, embryo_mask: np.ndarray) -> Labels:
    """Show the rasterized embryo ROI as a labels layer (0=background, 1=embryo)."""
    labels = embryo_mask.astype(np.uint8)

    if EMBRYO_ROI_LABELS_LAYER in viewer.layers:
        layer = viewer.layers[EMBRYO_ROI_LABELS_LAYER]
        layer.data = labels
        return layer

    return viewer.add_labels(
        labels,
        name=EMBRYO_ROI_LABELS_LAYER,
        opacity=0.35,
        blending="translucent",
    )


def build_embryo_roi(viewer: napari.Viewer) -> None:
    """Rasterize ``embryo_outline`` and update the ``embryo_roi`` labels layer."""
    mask = rasterize_embryo_outline(viewer)
    apply_embryo_roi_labels(viewer, mask)
    n_voxels = int(mask.sum())
    print(f"{EMBRYO_ROI_LABELS_LAYER}: {mask.shape}, {n_voxels} voxels inside ROI")


def setup_embryo_roi_viewer(viewer: napari.Viewer) -> None:
    """Load channels and prepare the shapes layer for step 1."""
    channels = load_project_channels()
    apply_channels_to_viewer(viewer, channels)
    ensure_embryo_outline_layer(viewer)
    print(
        "Step 1 — Embryo ROI:\n"
        f"  1. Select layer '{EMBRYO_OUTLINE_LAYER}' and the polygon tool.\n"
        "  2. Move through z and draw closed polygons around the embryo.\n"
        "  3. Click 'Build embryo ROI' to create the mask."
    )


@magicgui(call_button="Reload images from disk")
def reload_images_widget(viewer: napari.Viewer) -> None:
    """Reload TIFF channels from 22A_E1_Wnt3."""
    apply_channels_to_viewer(viewer, load_project_channels())


@magicgui(call_button="Build embryo ROI")
def build_embryo_roi_widget(viewer: napari.Viewer) -> None:
    """Rasterize embryo_outline shapes into the embryo_roi labels layer."""
    build_embryo_roi(viewer)


def add_embryo_roi_widgets(viewer: napari.Viewer) -> None:
    """Add dock widgets for reload and ROI building."""
    viewer.window.add_dock_widget(reload_images_widget)
    viewer.window.add_dock_widget(build_embryo_roi_widget)
