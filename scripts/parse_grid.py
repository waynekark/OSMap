"""Parse OS Terrain 50 ArcGIS ASCII grids into XYZ metre coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_ascii_grid(path: str | Path, ground_level: float | None = None) -> np.ndarray:
    """Return an ``(N, 3)`` array of ``(easting, northing, elevation)`` metres.

    When *ground_level* is provided, points below that elevation are discarded.
    """
    path = Path(path)
    header: dict[str, float] = {}
    values: list[float] = []

    with path.open(encoding="utf-8") as grid_file:
        for line in grid_file:
            fields = line.split()
            if not fields:
                continue
            try:
                values.extend(float(value) for value in fields)
            except ValueError:
                if len(fields) < 2:
                    raise ValueError(f"Invalid header line in {path}: {line.rstrip()}") from None
                try:
                    header[fields[0].lower()] = float(fields[1])
                except ValueError:
                    raise ValueError(f"Invalid header value in {path}: {line.rstrip()}") from None

    required = {"ncols", "nrows", "cellsize"}
    missing = required - header.keys()
    if missing:
        raise ValueError(f"Missing Arc ASCII header fields in {path}: {', '.join(sorted(missing))}")
    if not {"xllcorner", "xllcenter"} & header.keys():
        raise ValueError(f"Missing x origin in {path}")
    if not {"yllcorner", "yllcenter"} & header.keys():
        raise ValueError(f"Missing y origin in {path}")

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    if ncols <= 0 or nrows <= 0:
        raise ValueError(f"Grid dimensions must be positive in {path}")
    expected = ncols * nrows
    if len(values) != expected:
        raise ValueError(f"Grid shape mismatch in {path}: expected {expected} values, got {len(values)}")

    elevation = np.asarray(values, dtype=float).reshape(nrows, ncols)
    x_origin = header.get("xllcorner", header.get("xllcenter"))
    y_origin = header.get("yllcorner", header.get("yllcenter"))
    x_offset = 0.5 if "xllcorner" in header else 0.0
    y_offset = 0.5 if "yllcorner" in header else 0.0
    x = x_origin + (np.arange(ncols) + x_offset) * header["cellsize"]
    y = y_origin + (np.arange(nrows) + y_offset) * header["cellsize"]

    # Arc ASCII stores the first raster row at the northern edge.
    y_grid, x_grid = np.meshgrid(y[::-1], x, indexing="ij")
    points = np.column_stack((x_grid.ravel(), y_grid.ravel(), elevation.ravel()))
    nodata = header.get("nodata_value")
    if nodata is not None:
        points = points[~np.isclose(points[:, 2], nodata)]
    if ground_level is not None:
        points = points[points[:, 2] >= ground_level]
    return points


def load_ascii_grids(folder: str | Path, ground_level: float | None = None) -> np.ndarray:
    """Parse and combine every ``.asc`` file in *folder*, in filename order.

    When *ground_level* is provided, values below it are omitted from the returned data.
    """
    files = sorted(Path(folder).glob("*.asc"))
    if not files:
        raise FileNotFoundError(f"No .asc files found in {folder}")
    return np.vstack([parse_ascii_grid(path, ground_level=ground_level) for path in files])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=Path("data/process/grid"))
    parser.add_argument("-o", "--output", type=Path, help="Optional .npy output path")
    parser.add_argument(
        "--ground-level",
        type=float,
        default=None,
        help="Ignore elevation values below this ground level in metres.",
    )
    args = parser.parse_args()
    points = load_ascii_grids(args.folder, ground_level=args.ground_level)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, points)
    else:
        print(points)


if __name__ == "__main__":
    main()