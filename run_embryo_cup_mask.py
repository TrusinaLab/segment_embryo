"""
Embryo cup segmentation from cell labels (dilation + hole fill + closing).

Builds a 3D inside/outside mask of the embryo cup from ``data/test_cell_labels/``,
shows it in Napari, and can save the mask and masked fluorescence channels.

Run (micro-sam-napari conda env):
    conda activate micro-sam-napari
    python run_embryo_cup_mask.py
"""

from __future__ import annotations

from pathlib import Path

import napari
import numpy as np
from magicgui import magicgui

from image_io import (
    EMBRYO_CUP_MASK_DIR,
    EMBRYO_CUP_SEGMENT_DIR,
    apply_channels_to_viewer,
    get_middle_z_index_list,
    load_label_volume,
    load_middle_z_channels,
    load_segment_channels,
    project_root,
    save_channel_stack_as_tiffs,
    save_volume_tiff,
)
from segmentation.cell_features import (
    DEFAULT_PAD_CELLS_RADIUS,
    embryo_cup_labels,
    embryo_cup_mask_from_cells,
)
from segmentation.plane_split import mask_volume

CELL_LABELS_LAYER = "cell_labels"
EMBRYO_CUP_LAYER = "embryo_cup"


def _load_channels() -> dict[int, np.ndarray]:
    try:
        return load_segment_channels()
    except FileNotFoundError:
        print("Segment TIFFs not found; using raw middle z-subset.")
        return load_middle_z_channels()


def setup_embryo_cup_viewer(
    pad_radius: int = DEFAULT_PAD_CELLS_RADIUS,
) -> tuple[napari.Viewer, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    labels = load_label_volume()
    channels = _load_channels()

    if labels.shape != channels[min(channels)].shape:
        raise ValueError(
            f"Label shape {labels.shape} != image shape {channels[min(channels)].shape}"
        )

    cup_mask = embryo_cup_mask_from_cells(labels, pad_radius=pad_radius)
    cup_labels = embryo_cup_labels(cup_mask)

    viewer = napari.Viewer()
    apply_channels_to_viewer(viewer, channels)
    viewer.add_labels(labels, name=CELL_LABELS_LAYER, opacity=0.35)
    viewer.add_labels(cup_labels, name=EMBRYO_CUP_LAYER, opacity=0.45)

    print(
        f"Embryo cup mask: {int(cup_mask.sum()):,} voxels inside "
        f"(pad r={pad_radius}, holes filled + closing)"
    )
    return viewer, labels, cup_mask, channels


def add_embryo_cup_widgets(
    viewer: napari.Viewer,
    labels: np.ndarray,
    channels: dict[int, np.ndarray],
    initial_mask: np.ndarray,
) -> None:
    state = {"mask": initial_mask, "labels": labels, "channels": channels}

    @magicgui(
        pad_radius={"min": 0, "max": 20, "step": 1},
        call_button="Rebuild embryo cup mask",
    )
    def rebuild_mask(pad_radius: int = DEFAULT_PAD_CELLS_RADIUS) -> None:
        state["mask"] = embryo_cup_mask_from_cells(
            state["labels"], pad_radius=pad_radius
        )
        viewer.layers[EMBRYO_CUP_LAYER].data = embryo_cup_labels(state["mask"])
        print(f"Rebuilt cup mask: {int(state['mask'].sum()):,} interior voxels")

    @magicgui(call_button="Save embryo cup mask (TIFF)")
    def save_mask() -> None:
        path = project_root() / EMBRYO_CUP_MASK_DIR / "embryo_cup_mask.tif"
        save_volume_tiff(embryo_cup_labels(state["mask"]), path)
        print(f"Saved {path}")

    @magicgui(call_button="Save masked channels (TIFFs)")
    def save_masked_channels() -> None:
        z_indices = get_middle_z_index_list()
        out_dir = project_root() / EMBRYO_CUP_SEGMENT_DIR
        written: list[Path] = []
        for channel_idx, volume in sorted(state["channels"].items()):
            masked = mask_volume(volume, state["mask"])
            written.extend(
                save_channel_stack_as_tiffs(
                    masked, channel_idx, z_indices, out_dir
                )
            )
        print(f"Saved {len(written)} masked TIFFs to {out_dir}/")

    viewer.window.add_dock_widget(rebuild_mask)
    viewer.window.add_dock_widget(save_mask)
    viewer.window.add_dock_widget(save_masked_channels)


def main() -> None:
    viewer, labels, cup_mask, channels = setup_embryo_cup_viewer()
    add_embryo_cup_widgets(viewer, labels, channels, cup_mask)

    print(
        "\nEmbryo cup segmentation:\n"
        f"  Mask output: {EMBRYO_CUP_MASK_DIR.as_posix()}/\n"
        f"  Masked channels: {EMBRYO_CUP_SEGMENT_DIR.as_posix()}/\n"
        "  Layer 'embryo_cup' = 1 inside cup, 0 = background"
    )
    napari.run()


if __name__ == "__main__":
    main()
