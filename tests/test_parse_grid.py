import os

import numpy as np

from scripts.parse_and_plot_grid import main, parse_arc_ascii


def test_parse_arc_ascii_reads_real_grid_file():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "grid", "NY00.asc")
    pts = parse_arc_ascii(path)

    assert pts.shape[1] == 3
    assert pts.shape[0] == 40000
    assert np.isfinite(pts).all()
    assert pts[:, 2].min() < pts[:, 2].max()


def test_main_writes_plot_file_for_headless_runs(tmp_path):
    pts = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 3.0],
        [1.0, 1.0, 4.0],
    ])

    output_path = tmp_path / "elevation_plot.png"
    main(points=pts, save_path=str(output_path), sample=1)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
