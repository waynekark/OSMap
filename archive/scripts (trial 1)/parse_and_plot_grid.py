import argparse
import glob
import logging
import os
from typing import Optional, Tuple

import matplotlib

if os.environ.get("DISPLAY", "") == "" and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_DIR = os.path.join(REPO_ROOT, "data", "process", "grid")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "plots", "elevation_3d.png")


def parse_arc_ascii(path: str) -> np.ndarray:
    """
    Parse an ArcGIS ASCII grid (.asc). Returns Nx3 array of (x, y, z).
    Supports headers: ncols, nrows, xllcorner/xllcenter, yllcorner/yllcenter, cellsize, NODATA_value
    """
    header = {}
    z_rows = []
    with open(path, "r") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if not parts:
                continue

            try:
                float(parts[0])
            except ValueError:
                if len(parts) >= 2:
                    key = parts[0].lower()
                    val = parts[1]
                    try:
                        header[key] = float(val) if '.' in val or 'e' in val.lower() else int(val)
                    except ValueError:
                        header[key] = val
                continue

            z_rows.append([float(x) for x in parts])

    if not all(k in header for k in ("ncols", "nrows", "cellsize")):
        raise ValueError(f"Arc ASCII header incomplete in {path}")

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    cell = float(header["cellsize"])
    nodata = header.get("nodata_value", None)

    # z_rows should have nrows lines (Arc ASCII often lists rows from top to bottom)
    z = np.array(z_rows)
    if z.shape[0] != nrows or z.shape[1] != ncols:
        # try transposing or reshaping if line breaks differ
        z = z.flatten()
        if z.size != nrows * ncols:
            raise ValueError(f"Grid shape mismatch in {path}: expected {nrows}x{ncols}, got {z.shape}")
        z = z.reshape((nrows, ncols))

    # determine origin
    if "xllcorner" in header:
        x0 = float(header["xllcorner"])
        x_offset_center = True
    else:
        x0 = float(header.get("xllcenter", 0.0))
        x_offset_center = False
    if "yllcorner" in header:
        y0 = float(header["yllcorner"])
        y_offset_center = True
    else:
        y0 = float(header.get("yllcenter", 0.0))
        y_offset_center = False

    # compute cell centers
    # If using corner values, cell center is x0 + (col + 0.5)*cell
    x_centers = x0 + (np.arange(ncols) + (0.5 if x_offset_center else 0.0)) * cell if x_offset_center else x0 + np.arange(ncols) * cell
    y_centers = y0 + (np.arange(nrows) + (0.5 if y_offset_center else 0.0)) * cell if y_offset_center else y0 + np.arange(nrows) * cell

    # Arc ASCII often lists rows from top (north) to bottom; flip to match increasing y
    # If y_centers ascending doesn't match row order, flip rows so that first row corresponds to max y
    # We'll assume first z row = top -> highest y
    y_centers = y_centers[::-1]

    X, Y = np.meshgrid(x_centers, y_centers)
    Z = z

    pts = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    if nodata is not None:
        pts = pts[pts[:, 2] != float(nodata)]
    return pts


def parse_table(path: str) -> np.ndarray:
    """
    Parse a table-like file: CSV (with header) or whitespace-delimited XYZ.
    Return Nx3 array of (x, y, z). Attempts delimiter detection.
    """
    try:
        # Try numpy genfromtxt with comma first
        data = np.genfromtxt(path, delimiter=",", names=True)
        if data.size and data.dtype.names:
            # find columns that look like x,y,z
            names = [n.lower() for n in data.dtype.names]
            def pick(name_options):
                for opt in name_options:
                    if opt in names:
                        return data[data.dtype.names[names.index(opt)]]
                return None
            # Attempt common names
            xcol = None
            for opt in ("x", "lon", "longitude", "easting"):
                if opt in names:
                    xcol = data[data.dtype.names[names.index(opt)]]
                    break
            ycol = None
            for opt in ("y", "lat", "latitude", "northing"):
                if opt in names:
                    ycol = data[data.dtype.names[names.index(opt)]]
                    break
            zcol = None
            for opt in ("z", "elev", "elevation", "height"):
                if opt in names:
                    zcol = data[data.dtype.names[names.index(opt)]]
                    break
            if xcol is not None and ycol is not None and zcol is not None:
                out = np.column_stack((xcol, ycol, zcol))
                return out
        # Fallback: try whitespace-delimited numeric table
        arr = np.loadtxt(path)
        if arr.ndim == 1 and arr.size == 3:
            return arr.reshape(1, 3)
        if arr.shape[1] >= 3:
            return arr[:, :3]
    except Exception:
        pass

    # last resort: try whitespace with genfromtxt
    try:
        arr = np.genfromtxt(path)
        if arr.ndim == 1 and arr.size == 3:
            return arr.reshape(1, 3)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3]
    except Exception as e:
        raise ValueError(f"Could not parse table file {path}: {e}")

    raise ValueError(f"Could not parse file {path}")


def parse_file(path: str) -> np.ndarray:
    """
    Auto-detect file format and parse to Nx3 (x,y,z).
    """
    _, ext = os.path.splitext(path.lower())
    if ext in (".asc", ".txt") or os.path.basename(path).lower().endswith(".asc"):
        try:
            return parse_arc_ascii(path)
        except Exception:
            pass
    # Try table parser
    return parse_table(path)


def load_all_grid(grid_dir: str = GRID_DIR) -> np.ndarray:
    """
    Parse all files in grid_dir and combine into single Nx3 array (x,y,z).
    """
    patterns = [os.path.join(grid_dir, "*")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(f"No grid files found in {grid_dir}")
    all_pts = []
    for f in sorted(files):
        try:
            pts = parse_file(f)
            if pts.size:
                all_pts.append(pts)
            logging.info(f"Parsed {f}: {pts.shape[0]} points")
        except Exception as e:
            logging.warning(f"Skipping {f}: {e}")
    if not all_pts:
        raise ValueError("No points parsed from any grid files")
    return np.vstack(all_pts)


def plot_3d_scatter(points: np.ndarray, sample: int = 1, figsize: Tuple[int, int] = (12, 12),
                    title: str = "Elevation 3D Scatter", cmap: str = "terrain", s: float = 1.0,
                    save_path: Optional[str] = None, output_dpi: int = 600,
                    show_axes: bool = False, show_grid: bool = True,
                    equal_axes: bool = True) -> plt.Axes:
    """
    Plot Nx3 points as a 3D scatter. 'sample' can be >1 to downsample by taking every nth point.
    The OS grid uses 50m spacing, so the default is to keep the 3D axes scaled equally in all dimensions.
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must be Nx3 array")

    pts = points[::sample]
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=z, cmap=cmap, s=s, linewidth=0, alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label="Elevation")

    if equal_axes:
        axis_ranges = (np.ptp(x), np.ptp(y), np.ptp(z))
        ax.set_box_aspect(axis_ranges)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_visible(show_axes)
        axis._axinfo["grid"]["visible"] = show_grid
        axis._axinfo["grid"]["linewidth"] = 0.5
        tick_cfg = axis._axinfo["tick"]
        tick_cfg.setdefault("linewidth", {True: 0.8, False: 0.6})

    if not show_axes:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_visible(False)
            axis.label.set_color("none")
            axis.set_tick_params(which="both", label1On=False, label2On=False,
                                 size=0, width=0, colors="none")
            tick_cfg = axis._axinfo["tick"]
            tick_cfg["inward_factor"] = 0
            tick_cfg["outward_factor"] = 0
            tick_cfg["linewidth"] = {True: 0.0, False: 0.0}

    try:
        fig.tight_layout()
    except ValueError:
        # Matplotlib 3D axes can report an empty bounding box when all axes are hidden.
        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=output_dpi)
        logging.info(f"Saved plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return ax


def main(points: Optional[np.ndarray] = None, save_path: Optional[str] = None,
         sample: Optional[int] = None, grid_dir: str = GRID_DIR, show: bool = False,
         show_axes: bool = False, show_grid: bool = True, equal_axes: bool = True) -> Optional[str]:
    if points is None:
        points = load_all_grid(grid_dir)

    if sample is None:
        max_points = 200_000
        sample = max(1, int(np.ceil(len(points) / max_points)))

    if save_path is None and not show:
        save_path = DEFAULT_OUTPUT

    plot_3d_scatter(
        points,
        sample=sample,
        save_path=save_path if not show else None,
        show_axes=show_axes,
        show_grid=show_grid,
        equal_axes=equal_axes,
    )
    if show:
        return None
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse OS grid files and plot elevation data.")
    parser.add_argument("--grid-dir", default=GRID_DIR, help="Directory containing grid .asc files")
    parser.add_argument("--save-path", default=None, help="Output image path. Defaults to data/plots/elevation_3d.png")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively instead of saving a file")
    parser.add_argument("--sample", type=int, default=None, help="Downsampling interval for large point sets")
    parser.add_argument("--show-axes", action="store_true", help="Display 3D axes and axis lines")
    parser.add_argument("--hide-grid", action="store_true", help="Hide the 3D grid lines")
    parser.add_argument("--no-equal-axes", action="store_true", help="Disable equal scaling across all 3D axes")
    args = parser.parse_args()
    main(
        grid_dir=args.grid_dir,
        save_path=args.save_path,
        sample=args.sample,
        show=args.show,
        show_axes=args.show_axes,
        show_grid=not args.hide_grid,
        equal_axes=not args.no_equal_axes,
    )
