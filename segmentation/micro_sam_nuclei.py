"""Launch micro-SAM Annotator 3d on DAPI from the plane-split segment."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import zarr

from image_io import (
    CELL_LABELS_DIR,
    DAPI_CHANNEL,
    DAPI_LAYER_NAME,
    SEGMENT_OUTPUT_DIR,
    WNT_CHANNEL,
    WNT_LAYER_NAME,
    apply_channels_to_viewer,
    viewer_add_labels,
    discover_label_volume_path,
    load_label_volume,
    load_segment_channels,
    project_root,
)
DEFAULT_MODEL = "vit_b_lm"
EMBEDDINGS_SUBDIR = Path("data") / "embeddings"
DEFAULT_EMBEDDING_STEM = "segment_dapi_vit_b_lm"


def default_embedding_path(
    model_type: str = DEFAULT_MODEL, *, z_tag: str | None = None
) -> Path:
    """Zarr path for precomputed SAM embeddings (under ``data/embeddings/``)."""
    stem = DEFAULT_EMBEDDING_STEM if model_type == DEFAULT_MODEL else f"segment_dapi_{model_type}"
    if z_tag:
        stem = f"{stem}_{z_tag}"
    return project_root() / EMBEDDINGS_SUBDIR / f"{stem}.zarr"


def bottom_third_z_tag() -> str:
    """Embedding cache suffix for bottom-third Z test runs."""
    return "bottom_z3"


def z_range_tag(z_min: int, z_max: int) -> str:
    """Embedding cache suffix for an absolute z range (inclusive)."""
    return f"z{z_min}_{z_max}"


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


def embeddings_dir() -> Path:
    """Default folder for per-run ``*.zarr`` embedding caches."""
    return project_root() / EMBEDDINGS_SUBDIR


def release_embedding_handles() -> None:
    """Drop in-memory embedding references (e.g. after changing z range)."""
    try:
        from micro_sam.sam_annotator._state import AnnotatorState

        state = AnnotatorState()
        state.image_embeddings = None
    except Exception:
        pass


def allocate_embedding_cache(embedding_path: Path) -> Path:
    """
    Pick a writable ``*.zarr`` path for micro-SAM without deleting old caches.

    - **Complete** cache → reuse it.
    - **Missing** → use that path (empty folder created so the widget can validate).
    - **Incomplete** → use ``<name>_v2.zarr``, ``_v3.zarr``, … and leave the broken folder alone
      (avoids WinError 32 / "embeddings are incomplete" on the same path).
    """
    cache = Path(embedding_path)
    cache.parent.mkdir(parents=True, exist_ok=True)

    if embeddings_cache_complete(cache):
        return cache

    if cache.exists():
        stem = cache.stem
        for n in range(2, 200):
            alt = cache.parent / f"{stem}_v{n}.zarr"
            if embeddings_cache_complete(alt):
                print(f"Using complete embedding cache: {alt.name}")
                return alt
            if not alt.exists():
                cache = alt
                print(
                    f"Incomplete cache at {embedding_path.name} — "
                    f"using new path {cache.name} (you can delete the old folder later)."
                )
                break
        else:
            raise RuntimeError(
                f"Too many embedding cache variants under {cache.parent}. "
                "Delete unused folders in data/embeddings/ manually."
            )

    if not cache.exists():
        cache.mkdir(parents=True, exist_ok=True)
    return cache


def resolve_embedding_cache_path(
    path: Path | str | None,
    *,
    model_type: str = DEFAULT_MODEL,
    z_tag: str | None = None,
) -> Path:
    """
    Normalize the path micro-SAM expects: a single ``*.zarr`` store, not a parent folder.

    Picking ``data/embeddings`` in the file dialog fails because that folder already
    contains other caches and is treated as one incomplete zarr group.
    """
    if path is None:
        return default_embedding_path(model_type, z_tag=z_tag)

    path = Path(path)
    if path.suffix.lower() == ".zarr":
        return path

    if path.is_dir():
        if path.resolve() == embeddings_dir().resolve():
            resolved = default_embedding_path(model_type, z_tag=z_tag)
            print(
                f"Embedding path: using leaf cache {resolved.name} "
                f"(not the parent folder {path})."
            )
            return resolved
        resolved = path / f"{path.name}.zarr"
        print(f"Embedding path: using {resolved}")
        return resolved

    return path.with_suffix(".zarr")


def sync_embedding_widget_save_path(widget, cache_path: Path) -> None:
    """Set Annotator 3d embedding save path to a concrete ``*.zarr`` file."""
    resolved = str(cache_path.resolve())
    widget.embeddings_save_path_param.setText(resolved)
    widget.embeddings_save_path = resolved


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


def prepare_sam_for_napari(
    state,
    image: np.ndarray,
    *,
    model_type: str = DEFAULT_MODEL,
    embedding_path: Path,
    prefer_decoder: bool = True,
) -> bool:
    """
    Load the SAM model only — do not compute embeddings.

    Call **Compute Embeddings** in the Annotator 3d dock to load a cache or run
    the Napari progress bar. Returns True if a complete zarr cache already exists.
    """
    from micro_sam import util as ms_util
    from micro_sam.instance_segmentation import get_decoder

    cache = allocate_embedding_cache(Path(embedding_path))

    state.predictor, sam_state = ms_util.get_sam_model(
        model_type=model_type, return_state=True
    )
    state.decoder = None
    if prefer_decoder and "decoder_state" in sam_state:
        state.decoder = get_decoder(
            image_encoder=state.predictor.model.image_encoder,
            decoder_state=sam_state["decoder_state"],
            device=None,
        )

    state.image_shape = image.shape
    state.image_embeddings = None
    state.embedding_path = str(cache)
    state.data_signature = ms_util._compute_data_signature(image)

    ready = embeddings_cache_complete(cache)
    if ready:
        print(
            f"Embedding cache ready at {cache}. "
            "Click **Compute Embeddings** in Annotator 3d to load it."
        )
    else:
        print(
            f"No embedding cache at {cache}. "
            "Set z range if needed, then click **Compute Embeddings**."
        )
    return ready


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
        "    1. Embeddings cached — click **Compute Embeddings** in Annotator 3d to load, "
        "then segment\n"
        if embeddings_ready
        else (
            "    1. Click **Compute Embeddings** in Annotator 3d "
            "(Napari progress bar at bottom; CPU may look slow)\n"
        )
    )
    print(
        "Step 2 — micro-SAM 3D nuclei (from scratch):\n"
        f"  Channels: {SEGMENT_OUTPUT_DIR.as_posix()}/ "
        f"({WNT_LAYER_NAME} = Wnt, {DAPI_LAYER_NAME} = DAPI)\n"
        f"  SAM input: DAPI  shape (Z, Y, X) {dapi.shape}\n"
        f"  Embeddings cache: {embedding_path}\n"
        "  In the Annotator 3d dock:\n"
        f"{embed_step}"
        f"    2. Image Layer: **{DAPI_LAYER_NAME}** (DAPI) — model {DEFAULT_MODEL}\n"
        "       green/red points → Segment slice\n"
        "    3. Segment All Slices (Shift+S) to propagate in Z\n"
        "    4. Commit each object → committed_objects layer\n"
        "    5. File → Save Selected Layer(s)… → committed_objects only\n"
        f"       → {CELL_LABELS_DIR.as_posix()}/\n"
        "  Z range: **Z range (segmentation)** dock (right) — set z min/max, "
        "**Apply Z range**, then **Compute Embeddings** if needed.\n"
        "  Embeddings save path must be a **.zarr** file (e.g. "
        f".../data/embeddings/{DEFAULT_EMBEDDING_STEM}_z50_80.zarr), "
        "not the parent ``data/embeddings`` folder.\n"
    )
    if z_size > 40 and not embeddings_ready:
        print(
            f"  Note: {z_size} z-slices on CPU — first-time **Compute Embeddings** is slow."
        )
    if resumed:
        print("  Existing labels were loaded into committed_objects for editing.")


REFERENCE_LABELS_LAYER = "reference_labels_qc"
COMMITTED_OBJECTS_LAYER = "committed_objects"


def add_reference_labels_layer(
    viewer,
    labels: np.ndarray,
    *,
    name: str = REFERENCE_LABELS_LAYER,
) -> None:
    """
    Read-only overlay of the current segmentation for QC (spot fused nuclei).

    Shown as label contours so DAPI stays visible underneath.
    """
    from segmentation.label_focus import _apply_labels_display_style

    layer = viewer_add_labels(
        viewer, np.asarray(labels, dtype=np.uint32), name=name
    )
    _apply_labels_display_style(layer, opacity=0.7)
    layer.editable = False
    layer.mode = "pan_zoom"
    viewer.layers.move(viewer.layers.index(layer), len(viewer.layers) - 1)


def print_resegment_workflow(
    dapi: np.ndarray,
    embedding_path: Path,
    label_path: Path,
    *,
    embeddings_ready: bool,
) -> None:
    embed_step = (
        "    1. Embeddings cached — Compute Embeddings loads the zarr cache\n"
        if embeddings_ready
        else "    1. Click **Compute Embeddings** in Annotator 3d\n"
    )
    print(
        "micro-SAM re-segment (QC + fix fused cells):\n"
        f"  DAPI: {SEGMENT_OUTPUT_DIR.as_posix()}/  shape {dapi.shape}\n"
        f"  Reference overlay: {REFERENCE_LABELS_LAYER} (frozen, from {label_path.name})\n"
        f"  Edit layer: {COMMITTED_OBJECTS_LAYER} (loaded from same file — commit/save here)\n"
        f"  Embeddings: {embedding_path}\n"
        "  Workflow:\n"
        f"{embed_step}"
        "    2. Toggle reference overlay to find fused blobs (one ID, two nuclei in DAPI)\n"
        "       For solo-cell inspection use:  python run_inspect_cell_labels.py\n"
        "    3. Select point_prompts — green inside each nucleus, red on background\n"
        "    4. Segment slice → Segment All Slices (Shift+S) for that object\n"
        "    5. Commit from current_object (or auto_segmentation) → committed_objects\n"
        "    6. Erase old fused ID in committed_objects if needed (labels paint/erase)\n"
        "    7. File → Save Selected Layer(s)… → committed_objects only\n"
        f"       → {CELL_LABELS_DIR.as_posix()}/\n"
    )


def launch_resegment_annotator(
    image: np.ndarray,
    reference_labels: np.ndarray,
    *,
    embedding_path: Path | None = None,
    model_type: str = DEFAULT_MODEL,
) -> None:
    """
    Annotator 3d with DAPI, editable ``committed_objects``, and a QC reference overlay.

    ``reference_labels`` is shown read-only on top; ``committed_objects`` starts as a
    copy for editing and re-commit.
    """
    import napari
    from micro_sam.sam_annotator.annotator_3d import annotator_3d

    apply_embedding_progress_patch()
    cache = allocate_embedding_cache(
        resolve_embedding_cache_path(embedding_path, model_type=model_type)
    )

    if reference_labels.shape != image.shape:
        raise ValueError(
            f"Label shape {reference_labels.shape} does not match DAPI {image.shape}. "
            "Use labels from the same z-stack as the current segment."
        )

    viewer = annotator_3d(
        image,
        embedding_path=str(cache),
        segmentation_result=reference_labels,
        model_type=model_type,
        return_viewer=True,
    )
    add_reference_labels_layer(viewer, reference_labels)
    napari.run()


def _bind_sam_to_dapi_layer(viewer, *, dapi_layer_name: str = DAPI_LAYER_NAME) -> None:
    """Point micro-SAM's Image Layer dropdown at DAPI (no separate ``image`` layer)."""
    from micro_sam.sam_annotator._state import AnnotatorState

    if dapi_layer_name not in viewer.layers:
        raise KeyError(
            f"Layer '{dapi_layer_name}' not found. Available: {list(viewer.layers)}"
        )
    dapi_layer = viewer.layers[dapi_layer_name]
    embedding_widget = AnnotatorState().widgets["embeddings"]
    embedding_widget.image_selection.value = dapi_layer
    embedding_widget._initialize_image()


def launch_nuclei_annotator(
    *,
    segment_channels: dict[int, np.ndarray] | None = None,
    embedding_path: Path | None = None,
    segmentation_result: np.ndarray | None = None,
    model_type: str = DEFAULT_MODEL,
    full_segment_channels: dict[int, np.ndarray] | None = None,
    absolute_z_all: list[int] | None = None,
    initial_z_min: int | None = None,
    initial_z_max: int | None = None,
    resume_labels_full: np.ndarray | None = None,
    embedding_path_override: Path | None = None,
) -> None:
    """
    Open Annotator 3d on step-1 segment channels only (**channel 1**, **channel 2**, …).

    micro-SAM uses **channel 2** (DAPI) via the dock **Image Layer** selector.
    No extra ``image`` layer is added.
    """
    import napari
    from micro_sam.sam_annotator.annotator_3d import Annotator3d
    from micro_sam.sam_annotator.util import _sync_embedding_widget
    from micro_sam.sam_annotator._state import AnnotatorState

    apply_embedding_progress_patch()
    channels = segment_channels if segment_channels is not None else load_segment_channels()
    if DAPI_CHANNEL not in channels:
        raise KeyError(
            f"Channel {DAPI_CHANNEL} (DAPI) not found. Available: {sorted(channels)}"
        )
    dapi = channels[DAPI_CHANNEL]

    cache = allocate_embedding_cache(
        resolve_embedding_cache_path(embedding_path, model_type=model_type)
    )

    if segmentation_result is not None and segmentation_result.shape != dapi.shape:
        raise ValueError(
            f"Label shape {segmentation_result.shape} does not match DAPI {dapi.shape}. "
            "Use labels from the same z-stack as the current segment."
        )

    viewer = napari.Viewer()
    apply_channels_to_viewer(viewer, channels)

    state = AnnotatorState()
    prepare_sam_for_napari(
        state, dapi, model_type=model_type, embedding_path=cache
    )

    annotator = Annotator3d(viewer)
    viewer.window.add_dock_widget(annotator)
    _bind_sam_to_dapi_layer(viewer)
    annotator._update_image(segmentation_result=segmentation_result)
    embed_widget = state.widgets["embeddings"]
    _sync_embedding_widget(
        embed_widget,
        model_type,
        save_path=str(cache),
        checkpoint_path=None,
        device=None,
        tile_shape=None,
        halo=None,
    )
    sync_embedding_widget_save_path(embed_widget, cache)

    if full_segment_channels is not None and absolute_z_all is not None:
        from segmentation.nuclei_z_range import NucleiZRangeContext, add_nuclei_z_range_dock

        z_min = initial_z_min if initial_z_min is not None else absolute_z_all[0]
        z_max = initial_z_max if initial_z_max is not None else absolute_z_all[-1]
        z_ctx = NucleiZRangeContext(
            viewer=viewer,
            annotator=annotator,
            full_channels=full_segment_channels,
            absolute_z_all=absolute_z_all,
            model_type=model_type,
            embedding_path_override=embedding_path_override,
            resume_labels_full=resume_labels_full,
            current_z_min=z_min,
            current_z_max=z_max,
        )
        add_nuclei_z_range_dock(z_ctx)

    napari.run()
