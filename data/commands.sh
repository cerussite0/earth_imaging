#!/bin/bash
# ============================================================================
# Landsat Pipeline — Quick Reference Commands
# ============================================================================
# Run these from: /opt/watchdog/users/cerussite/alok/work/landsat_download
# Make sure the venv is active: source ../../.venv/bin/activate
# ============================================================================

# --- FULL PIPELINE (downloads everything from scratch) ---
# Only needed if you change the AOI or want a fresh download.
# Takes ~5 min (mostly downloading 462 MB of Landsat bands).
python run_pipeline.py

# --- INDIVIDUAL STEPS (run only what you need) ---

# 1. Download Landsat bands B2-B7 (full 185km scenes → data/*.TIF)
python download_bands.py

# 2. Clip all bands to the AOI (data/*.TIF → data/clipped/*.TIF)
python clip_bands.py

# 3. Download ESRI LULC at native 10m (→ data/esri_lulc_10m.tif)
python download_esri.py

# 4. Align ESRI to the Landsat pixel grid (10m → 30m, → data/clipped/esri_lulc.tif)
python align_data.py

# 5. Compute NDVI from clipped B4+B5 (→ data/clipped/ndvi.tif)
python compute_ndvi.py

# --- VIEW / VISUALIZE ---

# Generate all preview PNGs (saved to data/) — no interactive window
python view_data.py

# Generate PNGs AND open interactive matplotlib windows
# python view_data.py --show

# --- VERIFY PIXEL ALIGNMENT ---
# Checks that all files in data/clipped/ share the same grid
python -c "from align_data import verify_alignment; verify_alignment()"

# --- LIST ALL DOWNLOADED FILES ---
find data/ -name "*.tif" -o -name "*.TIF" | sort | xargs -I{} sh -c 'echo "{}: $(du -h {} | cut -f1)"'

# --- CHECK FILE SHAPES QUICKLY ---
python -c "
import rasterio, os, config
print('=== data/clipped/ (pixel-aligned, for segmentation) ===')
for f in sorted(os.listdir(config.CLIPPED_DIR)):
    if f.lower().endswith('.tif'):
        with rasterio.open(os.path.join(config.CLIPPED_DIR, f)) as s:
            print(f'  {f:20s}  {s.height}×{s.width}  {s.crs}  {abs(s.transform.a):.0f}m')

print()
print('=== data/ (full scenes + raw ESRI) ===')
for f in sorted(os.listdir(config.DATA_DIR)):
    p = os.path.join(config.DATA_DIR, f)
    if os.path.isfile(p) and f.lower().endswith('.tif'):
        with rasterio.open(p) as s:
            print(f'  {f:25s}  {s.height}×{s.width}  {s.crs}  {abs(s.transform.a):.0f}m')
"

# --- WHERE ARE THE PREVIEW IMAGES? ---
# Full-scene bands:      data/preview_bands.png
# Full-scene true color: data/preview_true_color.png
# Clipped bands (AOI):   data/preview_clipped_bands.png
# Clipped true color:    data/preview_clipped_true_color.png
# NDVI map + histogram:  data/preview_ndvi.png, data/preview_ndvi_histogram.png
# ESRI LULC with legend: data/preview_esri_lulc.png

# --- QUICK VIEW A SPECIFIC PNG (if display is available) ---
# xdg-open data/preview_clipped_true_color.png
# Or use: eog, feh, display, etc.
