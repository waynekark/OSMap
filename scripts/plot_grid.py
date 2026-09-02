"""Create printable 3D scatter plots from XYZ metre coordinates."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_xyz(
    points: np.ndarray,
    output: str | Path | None = None,
    title: str | None = None,
    axis_mode: str = "none",
    show_legend: bool = True,
    sample: int = 1,
) -> Path | None:
    """Plot XYZ points with equal metre scaling and optionally save the figure."""
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("points must be a non-empty (N, 3) numpy array")
    if axis_mode not in {"none", "xy", "all"}:
        raise ValueError("axis_mode must be 'none', 'xy', or 'all'")
    if sample < 1:
        raise ValueError("sample must be at least 1")

    points = points[::sample]
    figure = plt.figure(figsize=(16.54, 11.69))  # A3 landscape, inches.
    axes = figure.add_subplot(111, projection="3d")
    scatter = axes.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="terrain", s=0.5)
    ranges = np.ptp(points, axis=0)
    ranges[ranges == 0] = 1
    axes.set_box_aspect(ranges)

    if axis_mode == "all":
        axes.set_xlabel("Easting (m)")
        axes.set_ylabel("Northing (m)")
        axes.set_zlabel("Elevation (m)")
    elif axis_mode == "xy":
        axes.set_xlabel("Easting (m)")
        axes.set_ylabel("Northing (m)")
        axes.set_zlabel("")

    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        visible = axis_mode == "all" or (axis_mode == "xy" and axis in (axes.xaxis, axes.yaxis))
        axis.set_visible(visible)
        axis._axinfo["grid"]["visible"] = visible
    if title is not None:
        axes.set_title(title)
    if show_legend:
        figure.colorbar(scatter, ax=axes, label="Elevation (m)", shrink=0.65)

    output_path = Path(output) if output is not None else None
    if output_path is None:
        output_path = Path("plot") / f"elevation_3d_{datetime.now():%Y%m%d_%H%M%S}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help=".npy file containing an (N, 3) XYZ array")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--axis-mode", choices=("none", "xy", "all"), default="none")
    parser.add_argument("--hide-legend", action="store_true")
    parser.add_argument("--sample", type=int, default=1)
    args = parser.parse_args()
    output = plot_xyz(np.load(args.input), args.output, args.title, args.axis_mode, not args.hide_legend, args.sample)
    print(output)


if __name__ == "__main__":
    main()