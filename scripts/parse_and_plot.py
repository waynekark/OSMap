"""Parse the default OS grid directory and save a selected plot."""

from pathlib import Path

try:
    from .parse_grid import load_ascii_grids
    from .plot_grid import (
        PlotSettings, plot_contour_2d, plot_contour_3d,
        plot_filled_contour_3d, plot_filled_under_3d_lines,
        plot_hillshade, plot_stem, plot_wireframe, plot_xyz,
    )
except ImportError:
    from parse_grid import load_ascii_grids
    from plot_grid import (
        PlotSettings, plot_contour_2d, plot_contour_3d,
        plot_filled_contour_3d, plot_filled_under_3d_lines,
        plot_hillshade, plot_stem, plot_wireframe, plot_xyz,
    )


def main() -> None:
    points = load_ascii_grids(Path("data/process/grid"))
    plotters = {
        "1": ("2D contour", plot_contour_2d),
        "2": ("3D contour", plot_contour_3d),
        "3": ("3D filled contour", plot_filled_contour_3d),
        "4": ("custom hillshade", plot_hillshade),
        "5": ("filled 3D east-west lines", plot_filled_under_3d_lines),
        "6": ("3D wireframe", plot_wireframe),
        "7": ("3D stem", plot_stem),
        "8": ("3D scatter", plot_xyz),
    }
    print("Choose a plot:")
    for number, (name, _) in plotters.items():
        print(f"{number}: {name}")
    choice = input("Plot number [8]: ").strip() or "8"
    if choice not in plotters:
        raise ValueError(f"Unknown plot choice: {choice}")
    plotter = plotters[choice][1]
    settings = PlotSettings()
    output = plotter(points, settings=settings)
    print(output)


if __name__ == "__main__":
    main()