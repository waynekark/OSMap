"""Parse the default OS grid directory and save a 3D scatter plot."""

from pathlib import Path

from parse_grid import load_ascii_grids
from plot_grid import plot_xyz


def main() -> None:
    points = load_ascii_grids(Path("data/process/grid"))
    output = plot_xyz(points)
    print(output)


if __name__ == "__main__":
    main()