"""Scatter plots for EPI/VE cell feature CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_features(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.dropna(subset=["radial_alignment", "elongation"])


def plot_radial_vs_elongation(
    df: pd.DataFrame,
    out_path: Path,
    *,
    color_by: str | None = "distance_from_surface",
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))

    x = df["radial_alignment"]
    y = df["elongation"]

    if color_by and color_by in df.columns:
        sc = ax.scatter(
            x,
            y,
            c=df[color_by],
            cmap="viridis",
            alpha=0.85,
            edgecolors="k",
            linewidths=0.3,
            s=50,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.set_label(color_by.replace("_", " "))
    else:
        ax.scatter(x, y, alpha=0.85, edgecolors="k", linewidths=0.3, s=50)

    ax.set_xlabel("Radial alignment (|dot(major axis, outward normal)|)")
    ax.set_ylabel("Elongation (√λ₁/λ₂)")
    ax.set_title("Cell shape vs radial orientation (embryo1)")
    ax.set_xlim(-0.05, 1.05)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=project_root() / "data" / "epi_ve" / "cell_features.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root() / "data" / "epi_ve" / "radial_vs_elongation.png",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    df = load_features(args.csv)
    plot_radial_vs_elongation(df, args.out, show=args.show)
    print(f"Wrote {args.out} ({len(df)} cells)")


if __name__ == "__main__":
    main()
