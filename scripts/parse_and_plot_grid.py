import os
import glob
import logging
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

logging.basicConfig(level=logging.INFO)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_DIR = os.path.join(REPO_ROOT, "data", "grid")


def parse_arc_ascii(path: str) -> np.ndarray:
    """
    Parse an ArcGIS ASCII grid (.asc). Returns Nx3 array of (x, y, z).
    Supports headers: ncols, nrows, xllcorner/xllcenter, yllcorner/yllcenter, cellsize, NODATA_value
    """
    header = {}
    with open(path, "r") as f:
        # read header lines (first 6 usually)
        for _ in range(6):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if len(parts) >= 2:
                key = parts[0].lower()
                val = parts[1]
                try:
                    header[key] = float(val) if '.' in val or 'e' in val.lower() else int(val)
                except ValueError:
                    header[key] = val
        # rewind file to start reading numeric grid after header lines consumed
        # Collect remaining lines as z rows
        z_rows = []
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # split and parse floats
            z_rows.append([float(x) for x in stripped.split()])

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


def plot_3d_scatter(points: np.ndarray, sample: int = 1, figsize: Tuple[int, int] = (12, 8),
                    title: str = "Elevation 3D Scatter", cmap: str = "terrain", s: float = 1.0,
                    save_path: str = None) -> None:
    """
    Plot Nx3 points as a 3D scatter. 'sample' can be >1 to downsample by taking every nth point.
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
    plt.colorbar(sc, ax=ax, label="Elevation")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        logging.info(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    pts = load_all_grid()
    # by default downsample to at most ~200k points for plotting if very large
    max_points = 200_000
    sample = max(1, int(np.ceil(len(pts) / max_points)))
    plot_3d_scatter(pts, sample=sample)


if __name__ == "__main__":
    main()