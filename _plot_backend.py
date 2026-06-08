"""Configure matplotlib for shell-friendly, non-interactive plotting."""

import os
import warnings

import matplotlib


matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
    category=UserWarning,
)
