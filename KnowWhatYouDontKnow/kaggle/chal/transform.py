"""The image transformation.

Every image of the original dataset is rotated **once**, by 90, 180 or 270
degrees -- never 0 -- before anything else happens to it. The rotated image is
the one released and the one the label model is applied to; the original
orientation is never used again.

No pixel noise is added. Matching a released image against TissueMNIST reveals
only its *original* label, and the competition's labels are drawn from the
label model given the image, independently of the original one.
"""

from __future__ import annotations

import numpy as np

QUARTER_TURNS = (1, 2, 3)          # 90, 180, 270 degrees


def rotate(X: np.ndarray, rng: np.random.Generator
           ) -> tuple[np.ndarray, np.ndarray]:
    """``(X_out uint8 (n, H, W), quarter_turns (n,))`` for ``X`` uint8 (n, H, W)."""
    X = np.asarray(X)
    assert X.dtype == np.uint8 and X.ndim == 3 and X.shape[1] == X.shape[2], \
        f"expected square uint8 images, got {X.dtype} {X.shape}"
    k = rng.choice(np.array(QUARTER_TURNS), size=len(X))
    out = np.empty_like(X)
    for q in QUARTER_TURNS:
        sel = k == q
        out[sel] = np.rot90(X[sel], q, axes=(1, 2))
    return out, k
