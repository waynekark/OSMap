import os

import numpy as np

from scripts.parse_and_plot_grid import main, parse_arc_ascii, plot_3d_scatter


def test_parse_arc_ascii_reads_real_grid_file():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "process", "grid", "NY20.asc")
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


def test_plot_3d_scatter_defaults_to_hidden_axes_and_visible_gridlines(tmp_path):
    pts = np.array([
        [0.0, 0.0, 1.0],
        [50.0, 0.0, 2.0],
        [0.0, 50.0, 3.0],
        [50.0, 50.0, 4.0],
    ])

    output_path = tmp_path / "elevation_plot_defaults.png"
    ax = plot_3d_scatter(pts, save_path=str(output_path), sample=1)

    assert ax is not None
    box_aspect = ax.get_box_aspect()
    np.testing.assert_allclose(box_aspect[0] / box_aspect[2], 50.0 / 3.0)
    np.testing.assert_allclose(box_aspect[1] / box_aspect[2], 50.0 / 3.0)
    assert not ax.xaxis.line.get_visible()
    assert not ax.yaxis.line.get_visible()
    assert not ax.zaxis.line.get_visible()
    assert ax.xaxis._axinfo["grid"]["visible"] is True
    assert ax.yaxis._axinfo["grid"]["visible"] is True
    assert ax.zaxis._axinfo["grid"]["visible"] is True
