import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from segmentation.cell_features import (
    compute_cell_features,
    padded_embryo_mask_from_cells,
)


def test_features_on_synthetic_labels() -> None:
    labels = np.zeros((8, 32, 32), dtype=np.uint32)
    # Outer shell (tangential-like placement)
    labels[3, 4:28, 4] = 1
    labels[3, 4:28, 27] = 2
    # Inner blob
    labels[3, 12:20, 12:20] = 3

    df = compute_cell_features(labels)
    assert len(df) == 3
    assert "distance_from_embryo_com" in df.columns
    assert "radial_alignment" in df.columns
    assert df["distance_from_embryo_com"].min() >= 0
    inner = df.loc[df["label"] == 3, "distance_from_embryo_com"].iloc[0]
    outer = df.loc[df["label"] == 1, "distance_from_embryo_com"].iloc[0]
    assert inner < outer

    padded = padded_embryo_mask_from_cells(labels, radius=2, fill_holes=True)
    assert padded.sum() > (labels > 0).sum()

    ring = np.zeros((8, 32, 32), dtype=np.uint32)
    ring[3, 10:22, 10:22] = 1
    ring[3, 14:18, 14:18] = 0
    filled = padded_embryo_mask_from_cells(ring, radius=3, fill_holes=True)
    assert filled[3, 15, 15]

    cup = np.zeros((8, 40, 40), dtype=np.uint32)
    cup[3, 5:35, 5:35] = 1
    cup[3, 12:28, 12:28] = 0
    filled_cup = padded_embryo_mask_from_cells(cup, radius=4, fill_holes=True)
    assert filled_cup[3, 20, 20]


print("ok")
