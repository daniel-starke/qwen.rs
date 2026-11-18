"""SmoothQuant scales for the lean builders' `--act sq` (see `w8.py
--smooth`): c_j = max|x_j|^A / max|W_j|^(1-A), max|x_j| from `actq.py
--stats` (keyed by the export's MatMul names), max|W_j| over the fused
weight; normalized to geometric mean 1. The builder quantizes x / c per
token and requantizes W * c (1 / c folds into the RMSNorm weight)."""
import numpy as np


def smooth_c(stats, name, w, alpha):
    if name not in stats:
        raise SystemExit(f"no calibration statistics for {name}")
    c = np.maximum(stats[name], 1e-5) ** alpha / np.maximum(np.abs(w).max(0), 1e-8) ** (1 - alpha)
    return (c / np.exp(np.log(c).mean())).astype(np.float32)
