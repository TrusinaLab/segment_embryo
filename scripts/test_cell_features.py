import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from segmentation.cell_features import compute_cell_features


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


print("ok")
