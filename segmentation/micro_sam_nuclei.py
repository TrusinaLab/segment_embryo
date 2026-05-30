"""Launch micro-SAM Annotator 3d on DAPI from the plane-split segment."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import zarr

from image_io import (
    CELL_LABELS_DIR,
    SEGMENT_OUTPUT_DIR,
    discover_label_volume_path,
    load_label_volume,
    load_segment_channels,
    project_root,
)

DAPI_CHANNEL = 1
DEFAULT_MODEL = "vit_b_lm"
EMBEDDINGS_SUBDIR = Path("data") / "embeddings"
DEFAULT_EMBEDDING_STEM = "segment_dapi_vit_b_lm"


def default_embedding_path(model_type: str = DEFAULT_MODEL) -> Path:
    """Zarr path for precomputed SAM embeddings (under ``data/embeddings/``)."""
    stem = DEFAULT_EMBEDDING_STEM if model_type == DEFAULT_MODEL else f"segment_dapi_{model_type}"
    return project_root() / EMBEDDINGS_SUBDIR / f"{stem}.zarr"


def load_dapi_from_segment(root: Path | None = None) -> np.ndarray:
    """DAPI volume (Z, Y, X) from masked segment TIFFs written in step 1."""
    channels = load_segment_channels(root)
    if DAPI_CHANNEL not in channels:
        raise KeyError(
            f"Channel {DAPI_CHANNEL} (DAPI) not found in segment stack. "
            f"Available channels: {sorted(channels)}"
        )
    return channels[DAPI_CHANNEL]


def try_load_resume_labels(root: Path | None = None) -> np.ndarray | None:
    """Load an existing label volume for ``committed_objects`` if present."""
    labels_dir = project_root() / CELL_LABELS_DIR if root is None else root / CELL_LABELS_DIR
    if not labels_dir.is_dir():
        return None
    try:
        path = discover_label_volume_path(labels_dir)
    except FileNotFoundError:
        return None
    labels = load_label_volume(path)
    print(f"Resuming labels from {path.name} (shape {labels.shape})")
    return labels


def embeddings_cache_complete(embedding_path: Path) -> bool:
    """True if the zarr store has finished 3D embeddings (micro-SAM signature)."""
    if not embedding_path.exists():
        return False
    try:
        with zarr.open(embedding_path, mode="r") as store:
            return "input_size" in store.attrs and "features" in store
    except Exception:
        return False


def precompute_embeddings(
    image: np.ndarray,
    embedding_path: Path,
    model_type: str = DEFAULT_MODEL,
    *,
    force: bool = False,
) -> None:
    """
    Compute SAM embeddings in the terminal with a tqdm progress bar.

    Writes the same zarr cache that Annotator 3d uses, so Napari can load them
    without clicking **Compute Embeddings**.
    """
    from micro_sam import util as ms_util

    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    if force and embedding_path.exists():
        import shutil

        shutil.rmtree(embedding_path)
    elif embeddings_cache_complete(embedding_path):
        print(f"Embeddings already complete at {embedding_path}")
        return

    print(
        f"Precomputing embeddings ({model_type}) for {image.shape[0]} z-slices — "
        "progress in this terminal:"
    )
    predictor, _ = ms_util.get_sam_model(model_type=model_type, return_state=True)
    ms_util.precompute_image_embeddings(
        predictor=predictor,
        input_=image,
        save_path=str(embedding_path),
        ndim=3,
        verbose=True,
    )
    print(f"Embeddings saved to {embedding_path}")


def apply_embedding_progress_patch() -> None:
    """
    Let Napari's progress bar repaint during **Compute Embeddings**.

    micro-SAM runs embedding computation on the main Qt thread (thread workers
    disabled in napari >= 0.5). Without flushing events, the progress bar stays
    at 0% until the run finishes.
    """
    from qtpy.QtWidgets import QApplication
    from micro_sam.sam_annotator import _widgets as ms_widgets

    if getattr(ms_widgets, "_gastrul_progress_patch", False):
        return

    original_create: Callable = ms_widgets._create_pbar_for_threadworker

    def create_pbar_with_ui_refresh():
        pbar, pbar_signals = original_create()
        original_update = pbar.update

        def update_and_refresh(*args, **kwargs):
            original_update(*args, **kwargs)
            app = QApplication.instance()
            if app is not None:
                app.processEvents()

        pbar.update = update_and_refresh
        return pbar, pbar_signals

    ms_widgets._create_pbar_for_threadworker = create_pbar_with_ui_refresh
    ms_widgets._gastrul_progress_patch = True


def print_nuclei_workflow(
    dapi: np.ndarray,
    embedding_path: Path,
    *,
    resumed: bool,
    embeddings_ready: bool,
) -> None:
    z_size = dapi.shape[0]
    embed_step = (
        "    1. Embeddings are cached — in Annotator 3d click Compute Embeddings "
        "(loads cache) or start segmenting if already loaded\n"
        if embeddings_ready
        else (
            "    1. Embeddings: precomputed in terminal, or Compute Embeddings in Napari "
            "(progress bar at bottom; may look frozen on CPU — watch terminal if precomputing)\n"
        )
    )
    print(
        "Step 2 — micro-SAM 3D nuclei:\n"
        f"  Image: DAPI from {SEGMENT_OUTPUT_DIR.as_posix()}/\n"
        f"  Shape (Z, Y, X): {dapi.shape}\n"
        f"  Embeddings cache: {embedding_path}\n"
        "  In the Annotator 3d dock:\n"
        f"{embed_step}"
        f"    2. Model: {DEFAULT_MODEL} — green/red points → Segment slice\n"
        "    3. Segment All Slices (Shift+S) to propagate in Z\n"
        "    4. Commit each object → committed_objects layer\n"
        "    5. File → Save Selected Layer(s)… → committed_objects only\n"
        f"       → {CELL_LABELS_DIR.as_posix()}/\n"
        "  Tip: run with default (no flags) to precompute in the terminal with a "
        "slice-by-slice progress bar before Napari opens.\n"
    )
    if z_size > 40 and not embeddings_ready:
        print(
            f"  Note: {z_size} z-slices on CPU — first-time embedding is slow; "
            "use terminal precompute or set embeddings_save_path in the dock."
        )
    if resumed:
        print("  Existing labels were loaded into committed_objects for editing.")


def launch_nuclei_annotator(
    image: np.ndarray,
    *,
    embedding_path: Path | None = None,
    segmentation_result: np.ndarray | None = None,
    model_type: str = DEFAULT_MODEL,
) -> None:
    """Open napari with micro-SAM Annotator 3d on ``image``."""
    from micro_sam.sam_annotator.annotator_3d import annotator_3d

    apply_embedding_progress_patch()
    cache = embedding_path or default_embedding_path(model_type)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if segmentation_result is not None and segmentation_result.shape != image.shape:
        raise ValueError(
            f"Label shape {segmentation_result.shape} does not match DAPI {image.shape}. "
            "Use labels from the same z-stack as the current segment."
        )

    annotator_3d(
        image,
        embedding_path=str(cache),
        segmentation_result=segmentation_result,
        model_type=model_type,
    )
