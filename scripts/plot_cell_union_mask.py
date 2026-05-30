"""Export PNG: MIP Y–X of raw vs padded+hole-filled embryo masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_io import discover_label_volume_path, load_label_volume, project_root
from segmentation.cell_features import (
    DEFAULT_PAD_CELLS_RADIUS,
    _fill_enclosed_holes,
    cell_union_mask,
    padded_embryo_mask_from_cells,
)


def _mip_yx(mask) -> "object":
    return mask.max(axis=0)


def plot_union_comparison(
    labels,
    out_path: Path,
    *,
    pad_radius: int,
    show: bool = False,
) -> None:
    raw = cell_union_mask(labels)
    dilated = padded_embryo_mask_from_cells(labels, pad_radius, fill_holes=False)
    padded = _fill_enclosed_holes(dilated) if dilated.any() else dilated
    only_fill = padded & ~dilated

    z_size, y_size, x_size = raw.shape

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    panels = [
        (raw, f"Raw union — {int(raw.sum()):,} voxels"),
        (
            padded,
            f"Padded r={pad_radius}, holes filled — {int(padded.sum()):,} voxels",
        ),
        (only_fill, f"Voxels added by hole fill + close — {int(only_fill.sum()):,}"),
    ]

    for ax, (mask, title) in zip(axes, panels):
        ax.imshow(_mip_yx(mask), cmap="gray" if "hole" not in title.lower() else "hot", aspect="equal", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    fig.suptitle(
        f"MIP Y–X (collapse Z) — shape {z_size}×{y_size}×{x_size}; "
        "hole fill = per-slice + 3D binary_fill_holes + closing r=2",
        fontsize=10,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--pad-radius", type=int, default=DEFAULT_PAD_CELLS_RADIUS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    label_path = args.labels or discover_label_volume_path()
    labels = load_label_volume(label_path)
    out = args.out or (
        project_root() / "data" / "epi_ve" / f"cell_union_mask_padded_r{args.pad_radius}.png"
    )

    plot_union_comparison(labels, out, pad_radius=args.pad_radius, show=args.show)
    print(f"Wrote {out}")
    print(f"  Source: {label_path}")


if __name__ == "__main__":
    main()
