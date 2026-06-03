"""Napari dock widget to choose absolute z range for micro-SAM nuclei segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from magicgui import magicgui

from image_io import (
    apply_channels_to_viewer,
    subset_channels_by_absolute_z_range,
)
from image_io import DAPI_CHANNEL, DAPI_LAYER_NAME
from segmentation.micro_sam_nuclei import (
    _bind_sam_to_dapi_layer,
    allocate_embedding_cache,
    default_embedding_path,
    release_embedding_handles,
    resolve_embedding_cache_path,
    sync_embedding_widget_save_path,
    z_range_tag,
)


@dataclass
class NucleiZRangeContext:
    """Mutable state shared by the z-range dock and micro-SAM reload."""

    viewer: object
    annotator: object
    full_channels: dict[int, np.ndarray]
    absolute_z_all: list[int]
    model_type: str
    embedding_path_override: Path | None
    resume_labels_full: np.ndarray | None
    current_z_min: int
    current_z_max: int


def _embedding_path_for_range(
    ctx: NucleiZRangeContext, z_min: int, z_max: int
) -> Path:
    if ctx.embedding_path_override is not None:
        return ctx.embedding_path_override
    return default_embedding_path(ctx.model_type, z_tag=z_range_tag(z_min, z_max))


def apply_nuclei_z_range(
    ctx: NucleiZRangeContext,
    z_min: int,
    z_max: int,
) -> Path:
    """
    Subset channels and point micro-SAM at the matching embedding cache path.

    Does not re-load the SAM weights (already loaded at Napari startup).
    Click **Compute Embeddings** after applying.
    """
    from micro_sam.sam_annotator._state import AnnotatorState
    from micro_sam.sam_annotator.util import _sync_embedding_widget

    if z_min > z_max:
        raise ValueError(f"z_min ({z_min}) must be <= z_max ({z_max})")

    selection = subset_channels_by_absolute_z_range(
        ctx.full_channels, ctx.absolute_z_all, z_min, z_max
    )
    channels = selection.channels
    dapi = channels[DAPI_CHANNEL]
    tag = z_range_tag(z_min, z_max)
    embedding_path = allocate_embedding_cache(
        resolve_embedding_cache_path(
            _embedding_path_for_range(ctx, z_min, z_max),
            model_type=ctx.model_type,
            z_tag=tag,
        )
    )

    segmentation_result = None
    if ctx.resume_labels_full is not None:
        from image_io import align_labels_to_z_local_indices

        segmentation_result = align_labels_to_z_local_indices(
            ctx.resume_labels_full,
            dapi.shape,
            selection.local_indices,
            full_z_count=len(ctx.absolute_z_all),
        )

    apply_channels_to_viewer(ctx.viewer, channels)

    release_embedding_handles()
    state = AnnotatorState()
    state.image_shape = dapi.shape
    state.embedding_path = str(embedding_path)

    _bind_sam_to_dapi_layer(ctx.viewer, dapi_layer_name=DAPI_LAYER_NAME)
    ctx.annotator._update_image(segmentation_result=segmentation_result)

    embed_widget = state.widgets["embeddings"]
    _sync_embedding_widget(
        embed_widget,
        ctx.model_type,
        save_path=str(embedding_path),
        checkpoint_path=None,
        device=None,
        tile_shape=None,
        halo=None,
    )
    sync_embedding_widget_save_path(embed_widget, embedding_path)

    ctx.current_z_min = z_min
    ctx.current_z_max = z_max
    print(
        f"Z range applied: absolute z {z_min}–{z_max} → DAPI shape {dapi.shape}; "
        f"embeddings → {embedding_path.name}. "
        "Click **Compute Embeddings** in Annotator 3d."
    )
    return embedding_path


def add_nuclei_z_range_dock(ctx: NucleiZRangeContext) -> None:
    """Add a dock widget to set inclusive absolute z and reload micro-SAM."""
    z_lo = ctx.absolute_z_all[0]
    z_hi = ctx.absolute_z_all[-1]
    z_summary = f"{z_lo}–{z_hi} ({len(ctx.absolute_z_all)} slices in segment)"

    @magicgui(
        z_min={"min": z_lo, "max": z_hi, "step": 1, "label": "Z min (absolute)"},
        z_max={"min": z_lo, "max": z_hi, "step": 1, "label": "Z max (absolute)"},
        call_button="Apply Z range",
    )
    def z_range_panel(
        z_min: int = ctx.current_z_min,
        z_max: int = ctx.current_z_max,
    ) -> None:
        apply_nuclei_z_range(ctx, z_min, z_max)

    z_range_panel.z_min.value = ctx.current_z_min
    z_range_panel.z_max.value = ctx.current_z_max
    z_range_panel.native.setToolTip(
        f"Inclusive z indices from segment TIFF names. Stack on disk: {z_summary}. "
        "After Apply, click Compute Embeddings. Incomplete old .zarr folders are "
        "left alone; a new _v2, _v3, … cache name is used automatically."
    )
    ctx.viewer.window.add_dock_widget(
        z_range_panel,
        name="Z range (segmentation)",
        area="right",
    )
