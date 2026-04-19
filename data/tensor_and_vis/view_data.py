#!/usr/bin/env python3
"""
GeoTIFF Data Viewer
====================
Visualizes all downloaded .tif files in the data/ directory:
  • Individual bands (B2–B7) — grayscale with percentile stretch
  • True Color Composite — B4/B3/B2 as RGB
  • NDVI — RdYlGn colormap with colorbar
  • ESRI LULC — 9-class categorical map with legend

Usage:
    python view_data.py              # saves PNGs, no interactive display
    python view_data.py --show       # also opens matplotlib windows
"""

import os
import sys
import glob
import argparse

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _percentile_stretch(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Normalize an array to [0, 1] using percentile clipping."""
    p_lo = np.nanpercentile(arr[arr > 0], lo) if np.any(arr > 0) else 0
    p_hi = np.nanpercentile(arr[arr > 0], hi) if np.any(arr > 0) else 1
    return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-9), 0, 1)


def _load_band(path: str) -> np.ndarray:
    """Read a single-band GeoTIFF and mask zeros → NaN."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
    data[data == 0] = np.nan
    return data


def _print_metadata(path: str) -> None:
    """Print key GeoTIFF metadata."""
    with rasterio.open(path) as src:
        print(f"    CRS      : {src.crs}")
        print(f"    Shape    : {src.height} × {src.width}")
        print(f"    Bounds   : {src.bounds}")
        data = src.read(1).astype(np.float32)
        data[data == 0] = np.nan
        print(f"    Values   : [{np.nanmin(data):.4f}, {np.nanmax(data):.4f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization functions
# ─────────────────────────────────────────────────────────────────────────────

def _view_band_grid(band_paths: list, title: str, out_name: str,
                    show: bool = False) -> None:
    """Display a grid of spectral bands as grayscale images."""
    existing = [(b, p) for b, p in band_paths if os.path.exists(p)]

    if not existing:
        print("  No band files found")
        return

    n = len(existing)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_2d(axes)

    for idx, (band_name, path) in enumerate(existing):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        print(f"\n  {band_name}.TIF:")
        _print_metadata(path)

        data = _load_band(path)
        stretched = _percentile_stretch(data)
        ax.imshow(stretched, cmap="gray")
        ax.set_title(f"Band {band_name}", fontsize=13, fontweight="bold")
        ax.axis("off")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(config.DATA_DIR, out_name)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ Saved {out}")
    if show:
        plt.show()
    plt.close()


def view_individual_bands(show: bool = False) -> None:
    """Display each full-scene spectral band as a grayscale image."""
    paths = [(b, config.band_path(b)) for b in config.BANDS]
    _view_band_grid(paths, "Landsat Full-Scene Bands (Percentile Stretch)",
                    "preview_bands.png", show)


def view_clipped_bands(show: bool = False) -> None:
    """Display each AOI-clipped spectral band as a grayscale image."""
    paths = [(b, config.clipped_band_path(b)) for b in config.BANDS]
    if not any(os.path.exists(p) for _, p in paths):
        print("  No clipped band files found in data/clipped/")
        return
    _view_band_grid(paths, "Landsat Clipped Bands — AOI Only",
                    "preview_clipped_bands.png", show)


def _make_true_color(path_func, title: str, out_name: str,
                     show: bool = False) -> None:
    """Build and display a True Color Composite from B4/B3/B2."""
    paths = {b: path_func(b) for b in ["B4", "B3", "B2"]}
    for b, p in paths.items():
        if not os.path.exists(p):
            print(f"  ✗ {b}.TIF missing — cannot create True Color composite")
            return

    print(f"\n  Loading True Color bands (B4, B3, B2)...")
    red   = _percentile_stretch(_load_band(paths["B4"]))
    green = _percentile_stretch(_load_band(paths["B3"]))
    blue  = _percentile_stretch(_load_band(paths["B2"]))

    rgb = np.dstack((red, green, blue))
    rgb = np.nan_to_num(rgb, nan=0.0)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    out = os.path.join(config.DATA_DIR, out_name)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved {out}")
    if show:
        plt.show()
    plt.close()


def view_true_color(show: bool = False) -> None:
    """Display True Color Composite from full-scene bands."""
    _make_true_color(config.band_path,
                     "True Color Composite — Full Scene (B4/B3/B2)",
                     "preview_true_color.png", show)


def view_clipped_true_color(show: bool = False) -> None:
    """Display True Color Composite from AOI-clipped bands."""
    _make_true_color(config.clipped_band_path,
                     "True Color Composite — AOI Clipped (B4/B3/B2)",
                     "preview_clipped_true_color.png", show)


def view_ndvi(show: bool = False) -> None:
    """Display the NDVI raster with a Red-Yellow-Green colormap."""
    path = config.NDVI_PATH
    if not os.path.exists(path):
        print("  ✗ ndvi.tif not found — run compute_ndvi.py first")
        return

    print("\n  ndvi.tif:")
    _print_metadata(path)

    with rasterio.open(path) as src:
        ndvi = src.read(1).astype(np.float32)

    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDVI Value", fontsize=12)
    ax.set_title("NDVI — (NIR − Red) / (NIR + Red)", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    out = os.path.join(config.DATA_DIR, "preview_ndvi.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved {out}")
    if show:
        plt.show()
    plt.close()

    # Histogram
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    valid = ndvi[~np.isnan(ndvi)].ravel()
    ax2.hist(valid, bins=100, color="forestgreen", edgecolor="none")
    ax2.set_title("NDVI Value Distribution", fontsize=13, fontweight="bold")
    ax2.set_xlabel("NDVI")
    ax2.set_ylabel("Pixel Count")
    plt.tight_layout()
    out2 = os.path.join(config.DATA_DIR, "preview_ndvi_histogram.png")
    plt.savefig(out2, dpi=150)
    print(f"  ✓ Saved {out2}")
    if show:
        plt.show()
    plt.close()


def view_esri_lulc(show: bool = False) -> None:
    """Display the ESRI LULC map with the official 9-class legend."""
    path = config.ESRI_PATH
    if not os.path.exists(path):
        print("  ✗ esri_lulc.tif not found — run download_esri.py first")
        return

    print("\n  esri_lulc.tif:")
    _print_metadata(path)

    with rasterio.open(path) as src:
        lulc = src.read(1)

    # Build colormap that handles non-contiguous class values (0–11)
    max_val = max(config.ESRI_CLASSES.keys())
    color_list = ["#000000"] * (max_val + 1)  # default black for unmapped
    for val, (name, hex_color) in config.ESRI_CLASSES.items():
        color_list[val] = hex_color
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm(np.arange(-0.5, max_val + 1.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(lulc, cmap=cmap, norm=norm)
    ax.set_title("ESRI LULC 10m — Land Cover Classification",
                 fontsize=14, fontweight="bold")
    ax.axis("off")

    # Legend (skip "No Data" class 0)
    legend_patches = [
        mpatches.Patch(color=hex_c, label=name)
        for val, (name, hex_c) in sorted(config.ESRI_CLASSES.items())
        if val != 0
    ]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1),
              loc="upper left", borderaxespad=0., title="Land Cover",
              frameon=True)

    plt.tight_layout()
    out = os.path.join(config.DATA_DIR, "preview_esri_lulc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved {out}")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize downloaded GeoTIFF data.")
    parser.add_argument("--show", action="store_true",
                        help="Open interactive matplotlib windows")
    args = parser.parse_args()

    print("=" * 60)
    print("  GEOTIFF DATA VIEWER")
    print("=" * 60)
    print(f"  Data directory: {config.DATA_DIR}\n")

    # List all files (recursive)
    tifs = []
    for root, dirs, files in os.walk(config.DATA_DIR):
        for f in sorted(files):
            if f.lower().endswith(".tif"):
                tifs.append(os.path.join(root, f))

    if not tifs:
        print("  ✗ No .tif files found. Run the pipeline first:")
        print("    python run_pipeline.py")
        return

    print(f"  Found {len(tifs)} GeoTIFF files:")
    for t in tifs:
        size = os.path.getsize(t)
        size_str = f"{size/(1024*1024):.1f} MB" if size > 1024*1024 else f"{size/1024:.0f} KB"
        rel = os.path.relpath(t, config.DATA_DIR)
        print(f"    • {rel:30s} {size_str}")

    # Render each visualization
    print("\n" + "─" * 60)
    print("  1. Full-Scene Bands")
    print("─" * 60)
    view_individual_bands(show=args.show)

    print("\n" + "─" * 60)
    print("  2. Full-Scene True Color Composite")
    print("─" * 60)
    view_true_color(show=args.show)

    print("\n" + "─" * 60)
    print("  3. Clipped Bands (AOI)")
    print("─" * 60)
    view_clipped_bands(show=args.show)

    print("\n" + "─" * 60)
    print("  4. Clipped True Color (AOI)")
    print("─" * 60)
    view_clipped_true_color(show=args.show)

    print("\n" + "─" * 60)
    print("  5. NDVI")
    print("─" * 60)
    view_ndvi(show=args.show)

    print("\n" + "─" * 60)
    print("  6. ESRI LULC")
    print("─" * 60)
    view_esri_lulc(show=args.show)

    print("\n" + "=" * 60)
    print("  ✓ All previews saved to data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
