from pathlib import Path

import numpy as np

from scripts.parse_grid import load_ascii_grids, parse_ascii_grid


def _write_grid(path: Path, values):
    rows = len(values)
    cols = len(values[0])
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"ncols\t{cols}\n")
        handle.write(f"nrows\t{rows}\n")
        handle.write("xllcorner\t0\n")
        handle.write("yllcorner\t0\n")
        handle.write("cellsize\t1\n")
        handle.write("nodata_value\t-9999\n")
        for row in values:
            handle.write("\t".join(str(v) for v in row) + "\n")


def test_parse_ascii_grid_filters_below_ground_level(tmp_path):
    grid = [
        [0.0, 10.0],
        [20.0, 30.0],
    ]
    path = tmp_path / "ground.asc"
    _write_grid(path, grid)

    points = parse_ascii_grid(path, ground_level=10.0)

    assert points[:, 2].min() >= 10.0
    assert points.shape[0] == 3


def test_load_ascii_grids_accepts_ground_level_filter(tmp_path):
    first = [
        [5.0, 15.0],
        [25.0, 35.0],
    ]
    second = [
        [40.0, 45.0],
        [50.0, 55.0],
    ]
    _write_grid(tmp_path / "a.asc", first)
    _write_grid(tmp_path / "b.asc", second)

    points = load_ascii_grids(tmp_path, ground_level=30.0)

    assert points[:, 2].min() >= 30.0
    assert points.shape[0] == 5
