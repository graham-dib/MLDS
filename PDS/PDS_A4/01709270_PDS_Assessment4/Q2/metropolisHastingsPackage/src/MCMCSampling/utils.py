# utils.py

import numpy as np


def _validate_samples(samples):
    """
    Defensive programming helper. Validate given samples for the plot_trace and plot_histogram functions. Ensures samples is a one-dimensional, non-empty array of finite floats. 
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 1:
        raise ValueError("samples must be one-dimensional.")
    if arr.size == 0:
        raise ValueError("samples must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("samples must contain only finite values.")
    return arr


def plot_trace(samples):
    """
    Print a trace plot. Using ASCII characters as non-base python packages are not allowed. pandas implicitly requires matplotlib for plotting, which is not allowed.
    """
    arr = _validate_samples(samples)

    lo, hi = arr.min(), arr.max()
    if lo == hi:
        print("Trace (constant):")
        print("*" * min(80, arr.size))
        return

    #Terminal width for trace plot
    width = 80
    idx = np.linspace(0, arr.size - 1, num=min(width, arr.size)).astype(int)
    vals = arr[idx]
    pos = ((vals - lo) / (hi - lo) * (width - 1)).astype(int)

    line = [" "] * width
    for p in pos:
        line[p] = "*"

    print("Trace:")
    print("".join(line))


def plot_histogram(samples, bins=30):
    """
    Print a normalised histogram, using ASCII characters.
    """
    arr = _validate_samples(samples)
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer.")

    hist, edges = np.histogram(arr, bins=bins, density=True)
    
    
    width = 80
    hmax = hist.max()

    print("Histogram:")
    for i in range(bins):
        bar = "" if hmax == 0 else "#" * int(hist[i] / hmax * width)
        print(f"[{edges[i]:.3g}, {edges[i+1]:.3g}) {bar}")