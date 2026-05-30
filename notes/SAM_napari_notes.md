# SAM for Napari — chat summary (setup & tool choice)

Notes from the first part of the conversation: choosing and installing Segment Anything for interactive 3D segmentation in napari.

## Goal

Interactive 3D segmentation (e.g. nuclei in light-microscopy z-stacks) with **click-to-segment** and **positive/negative point refinement** in napari.

## Recommended tool: **micro-sam (μSAM)**

Best fit for mouse gastrula / light-microscopy nuclei:

- Published, widely used in microscopy ([Nature Methods](https://www.nature.com/articles/s41592-024-02580-4))
- **Annotator 3d** with `point_prompts` layer (green = inside, red = background)
- **Segment All Slices** (`Shift+S`) propagates through Z
- Fine-tuned light-microscopy models:
  - **`vit_b_lm`** — good default (speed vs quality)
  - **`vit_l_lm`** — slightly better, heavier
  - **`vit_t_lm`** — fastest, lower quality

Install via **conda-forge only** (officially **not supported with pip**).

## Alternatives considered

| Plugin | 3D nuclei? | Notes |
|--------|------------|-------|
| **micro-sam** | ✅ | Primary recommendation |
| **napari-sam3-assistant** | ✅ (CUDA for 3D) | Newer SAM3; less microscopy-specific |
| **napari-sam** (MIC-DKFZ) | partial | Mainly 2D; needs `segment_anything` |
| **napari-segment-anything** (Royer) | ❌ | 2D only |

## Environment strategy: conda only (Option B)

**Do not merge conda and uv in one environment.** Use one package manager per env.

- **`micro-sam-napari`** conda env: napari + micro_sam + PyTorch (CPU)
- Project code (`main.py`) runs from that env
- Old **uv / `.venv`** retired for napari work (optional to keep for other scripts)

Created files:

| File | Purpose |
|------|---------|
| `environment.yml` | CPU conda env (Windows pins: `nifty=1.2.1=*_4`, `protobuf<5`) |
| `scripts/run_napari.bat` | `conda run -n micro-sam-napari python main.py` |
| `scripts/run_micro_sam_3d.bat` | Opens `micro_sam.annotator_3d` directly |

Verified env: micro_sam 1.3.1, napari 0.7.0, torch 2.5.1, **CPU** (`cuda False`).

## How to run

1. **`scripts/run_napari.bat`** (or `conda activate micro-sam-napari` → `python main.py`) — loads TIFF channels
2. **Plugins → micro-sam → Annotator 3d** (or `scripts/run_micro_sam_3d.bat`)
3. Model **`vit_b_lm`** → **Compute Embeddings** → point prompts → segment → **Segment All Slices**

Do **not** use `uv run main.py` for SAM work.

## Embeddings (CPU)

- No real **Stop** button; napari may show “not responding” during long CPU runs — wait or kill the process (`Ctrl+C` / close window).
- **Restart:** click **Compute Embeddings** again.
- Set **`embeddings_save_path`** in Embedding Settings to cache and avoid recomputing.
- Incomplete cache folder → delete it or use a new path before recomputing.

## Workflow recap (nuclei, CPU)

1. Load stack in napari
2. Annotator 3d → **`vit_b_lm`**
3. Compute embeddings (slow on CPU; downloads weights on first run)
4. Green points on nuclei, red on background
5. Segment slice → refine with points
6. **Segment All Slices** (`Shift+S`) for Z propagation

Expect slow embedding and 3D propagation on CPU; use a middle z-subset for tests.
