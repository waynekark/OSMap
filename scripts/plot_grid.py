"""Create printable plots from XYZ metre coordinates."""

from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass
class PlotSettings:
    """Shared, editable settings used by every plot type."""

    show_axes: bool = False
    show_legend: bool = False
    scaling: str = "equal"
    elevation: float = 30
    azimuth: float = -80    # -90 results in a north-south oriented map
    cmap: str = "terrain"
    dpi: int = 600
    figsize: tuple[float, float] = (16.54, 11.69)


def _grid(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert row-major XYZ points into X, Y and Z plotting grids."""
    x_values = np.unique(points[:, 0])
    y_values = np.unique(points[:, 1])
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    z_grid = np.full((len(y_values), len(x_values)), np.nan)
    for x, y, z in points:
        z_grid[y_index[y], x_index[x]] = z
    return np.meshgrid(x_values, y_values), z_grid


def _figure(settings: PlotSettings, three_dimensional: bool = True):
    figure = plt.figure(figsize=settings.figsize)
    axes = figure.add_subplot(111, projection="3d" if three_dimensional else None)
    if three_dimensional:
        axes.view_init(elev=settings.elevation, azim=settings.azimuth)
    return figure, axes


def _style_axes(axes, settings: PlotSettings, three_dimensional: bool = True) -> None:
    if settings.scaling not in {"equal", "auto"}:
        raise ValueError("scaling must be 'equal' or 'auto'")
    if settings.scaling == "equal":
        if three_dimensional:
            limits = np.array([axes.get_xlim3d(), axes.get_ylim3d(), axes.get_zlim3d()])
            centre = limits.mean(axis=1)
            radius = (limits[:, 1] - limits[:, 0]).max() / 2
            axes.set_xlim3d(centre[0] - radius, centre[0] + radius)
            axes.set_ylim3d(centre[1] - radius, centre[1] + radius)
            axes.set_zlim3d(centre[2] - radius, centre[2] + radius)
            axes.set_box_aspect((1, 1, 1))
        else:
            axes.set_aspect("equal", adjustable="box")
    if not settings.show_axes:
        axes.set_axis_off()
        if three_dimensional:
            for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
                axis.set_visible(False)
                axis._axinfo["grid"]["visible"] = False
                axis.pane.set_visible(False)
    else:
        axes.grid(True)


def _save(figure, output: str | Path | None, settings: PlotSettings, name: str) -> Path:
    output_path = Path(output) if output is not None else Path("plot") / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=settings.dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _validate(points: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("points must be a non-empty (N, 3) numpy array")


def plot_contour_2d(points: np.ndarray, output=None, title=None, settings=None, levels=20):
    _validate(points)
    settings = settings or PlotSettings()
    (x_grid, y_grid), z_grid = _grid(points)
    figure, axes = _figure(settings, three_dimensional=False)
    contour = axes.contour(x_grid, y_grid, z_grid, levels=levels, cmap=settings.cmap)
    if title is not None:
        axes.set_title(title)
    if settings.show_legend:
        figure.colorbar(contour, ax=axes, label="Elevation (m)")
    _style_axes(axes, settings, three_dimensional=False)
    return _save(figure, output, settings, "contour_2d")


def plot_contour_3d(points: np.ndarray, output=None, title=None, settings=None, levels=20, linewidths=0.25):
    _validate(points)
    settings = settings or PlotSettings()
    (x_grid, y_grid), z_grid = _grid(points)
    figure, axes = _figure(settings)
    contour = axes.contour(x_grid, y_grid, z_grid, levels=levels, cmap=settings.cmap, linewidths=linewidths)
    if title is not None:
        axes.set_title(title)
    if settings.show_legend:
        figure.colorbar(contour, ax=axes, label="Elevation (m)", shrink=0.65)
    _style_axes(axes, settings)
    return _save(figure, output, settings, "contour_3d")


def plot_filled_contour_3d(points: np.ndarray, output=None, title=None, settings=None, levels=20):
    _validate(points)
    settings = settings or PlotSettings()
    (x_grid, y_grid), z_grid = _grid(points)
    figure, axes = _figure(settings)
    contour = axes.contourf(x_grid, y_grid, z_grid, levels=levels, cmap=settings.cmap)
    if title is not None:
        axes.set_title(title)
    if settings.show_legend:
        figure.colorbar(contour, ax=axes, label="Elevation (m)", shrink=0.65)
    _style_axes(axes, settings)
    return _save(figure, output, settings, "contour_filled_3d")


def plot_hillshade(points: np.ndarray, output=None, title=None, settings=None, azimuth=315, altitude=45):
    _validate(points)
    settings = settings or PlotSettings()
    (x_grid, y_grid), z_grid = _grid(points)
    figure, axes = _figure(settings, three_dimensional=False)
    gradient_y, gradient_x = np.gradient(z_grid)
    slope = np.pi / 2 - np.arctan(np.hypot(gradient_x, gradient_y))
    aspect = np.arctan2(-gradient_x, gradient_y)
    shaded = np.sin(np.deg2rad(altitude)) * np.sin(slope) + np.cos(np.deg2rad(altitude)) * np.cos(slope) * np.cos(np.deg2rad(azimuth) - aspect)
    axes.imshow(shaded, cmap="gray", origin="lower", extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()))
    if title is not None:
        axes.set_title(title)
    _style_axes(axes, settings, three_dimensional=False)
    return _save(figure, output, settings, "hillshade")


def plot_filled_under_3d_lines(points: np.ndarray, output=None, title=None, settings=None):
    _validate(points)
    settings = settings or PlotSettings()
    (x_grid, y_grid), z_grid = _grid(points)
    figure, axes = _figure(settings)
    base = np.nanmin(z_grid)
    for row, northing in enumerate(y_grid[:, 0]):
        elevation = z_grid[row]
        axes.plot(x_grid[row], np.full(x_grid.shape[1], northing), elevation, color="black", linewidth=0.4)
        vertices = list(zip(x_grid[row], np.full(x_grid.shape[1], northing), elevation))
        vertices.extend([(x_grid[row, -1], northing, base), (x_grid[row, 0], northing, base)])
        axes.add_collection3d(Poly3DCollection([vertices], alpha=0.35, facecolor="steelblue", edgecolor="none"))
    if title is not None:
        axes.set_title(title)
    _style_axes(axes, settings)
    return _save(figure, output, settings, "filled_3d_lines")


def plot_wireframe(points: np.ndarray, output=None, title=None, settings=None):
    _validate(points)
    settings = settings or PlotSettings()
    (x_grid, y_grid), z_grid = _grid(points)
    figure, axes = _figure(settings)
    axes.plot_wireframe(x_grid, y_grid, z_grid, color="black", linewidth=0.35)
    if title is not None:
        axes.set_title(title)
    _style_axes(axes, settings)
    return _save(figure, output, settings, "wireframe_3d")


def plot_stem(points: np.ndarray, output=None, title=None, settings=None, sample=1):
    _validate(points)
    settings = settings or PlotSettings()
    points = points[::sample]
    figure, axes = _figure(settings)
    axes.stem(points[:, 0], points[:, 1], points[:, 2], linefmt="black", markerfmt=".", basefmt=" ")
    if title is not None:
        axes.set_title(title)
    _style_axes(axes, settings)
    return _save(figure, output, settings, "stem_3d")


def plot_xyz(
    points: np.ndarray,
    output: str | Path | None = None,
    title: str | None = None,
    axis_mode: str = "none",
    show_legend: bool = False,
    sample: int = 1,
    elev: float = 30,
    azim: float = -60,
    settings: PlotSettings | None = None,
) -> Path | None:
    """Plot XYZ points with equal metre scaling and optionally save the figure."""
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("points must be a non-empty (N, 3) numpy array")
    if axis_mode not in {"none", "xy", "all"}:
        raise ValueError("axis_mode must be 'none', 'xy', or 'all'")
    if sample < 1:
        raise ValueError("sample must be at least 1")

    points = points[::sample]
    settings = settings or PlotSettings(
        show_axes=axis_mode != "none",
        show_legend=show_legend,
        elevation=elev,
        azimuth=azim,
    )
    figure = plt.figure(figsize=settings.figsize)  # A3 landscape, inches.
    axes = figure.add_subplot(111, projection="3d")
    axes.view_init(elev=settings.elevation, azim=settings.azimuth)
    scatter = axes.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=points[:, 2], cmap=settings.cmap, s=1, marker=".", linewidths=0,
    )
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

    axes.grid(False)
    if axis_mode == "none":
        axes.set_axis_off()
        for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
            axis.set_visible(False)
            axis._axinfo["grid"]["visible"] = False
            axis.pane.set_visible(False)
    else:
        for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
            visible = axis_mode == "all" or axis in (axes.xaxis, axes.yaxis)
            axis.set_visible(visible)
            axis._axinfo["grid"]["visible"] = visible
    if title is not None:
        axes.set_title(title)
    if settings.show_legend:
        figure.colorbar(scatter, ax=axes, label="Elevation (m)", shrink=0.65)

    output_path = Path(output) if output is not None else None
    if output_path is None:
        output_path = Path("plot") / f"elevation_3d_{datetime.now():%Y%m%d_%H%M%S}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=settings.dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help=".npy file containing an (N, 3) XYZ array")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--axis-mode", choices=("none", "xy", "all"), default="none")
    legend_options = parser.add_mutually_exclusive_group()
    legend_options.add_argument("--show-legend", action="store_true")
    legend_options.add_argument("--hide-legend", action="store_true")
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--elev", type=float, default=30, help="Camera elevation angle in degrees")
    parser.add_argument("--azim", type=float, default=-60, help="Camera azimuth angle in degrees")
    args = parser.parse_args()
    output = plot_xyz(
        np.load(args.input), args.output, args.title, args.axis_mode,
        args.show_legend, args.sample, args.elev, args.azim,
    )
    print(output)


if __name__ == "__main__":
    main()