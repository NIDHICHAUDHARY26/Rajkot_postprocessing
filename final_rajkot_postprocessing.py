"""
OBIA Post-Processing — Full Rajkot District  v2
================================================
Progress bar shows: field count, %, time elapsed, ETA, speed (fields/sec)

Outputs:
  OUT_TIF : obia_rajkot_postprocessed.tif
  OUT_CSV : rajkot_field_summary.csv
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from collections import Counter
import pandas as pd
import os
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["SHAPE_RESTORE_SHX"] = "YES"

# ─── PATHS ────────────────────────────────────────────────────────────────────
SHP_PATH = r"C:\Amnex-Learning\April_task\rajkot_postprocessing\data\rajkot_UTM_final\rajkot_UTM_final.shp"
IMG_PATH = r"C:\Amnex-Learning\April_task\rajkot_postprocessing\data\rajkot_pre-final_rabi_classification_pixel_level_2025-26.img"
OUT_DIR  = r"C:\Amnex-Learning\April_task\rajkot_postprocessing\output"
OUT_TIF  = os.path.join(OUT_DIR, "obia_rajkot_postprocessed.tif")
OUT_CSV  = os.path.join(OUT_DIR, "rajkot_field_summary.csv")

# ─── THRESHOLDS ───────────────────────────────────────────────────────────────
VALID_MIN          = 1
VALID_MAX          = 36
MMU_THRESHOLD      = 0.02
NEIGHBOR_MIN_AGREE = 4
KERNEL_PASSES      = 3

# ─── CLASS NAMES ──────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0:  "Background",
    1:  "Wheat",        2:  "Jowar",        3:  "Maize",        4:  "Gram",
    5:  "Mustard",      6:  "Sugarcane",    7:  "Tobacco",      8:  "Cumin",
    9:  "Coriander",    10: "Garlic",       11: "Sawa",         12: "Isabgul",
    13: "Fennel",       14: "Onion",        15: "Potato",       16: "Vegetables",
    17: "Other Crops",  18: "Math",         19: "Mung",         20: "Bajra",
    21: "Chikori",      22: "Ajwain",       23: "Rajko",        24: "Rajgira",
    25: "Indianbean",   26: "Cowpea",       27: "Lentil",       28: "Fenugreek",
    29: "Jute",         30: "Urid",         31: "Sweet Potato", 32: "Rabi Sown",
    33: "Kalonji",      34: "Chilli",       35: "Pea"
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def apply_mmu(pixels_1d):
    valid = pixels_1d[pixels_1d > 0]
    if len(valid) == 0:
        return pixels_1d
    counts  = Counter(valid.tolist())
    total   = len(valid)
    cleaned = pixels_1d.copy()
    for cls, cnt in counts.items():
        if cnt / total < MMU_THRESHOLD:
            cleaned[cleaned == cls] = 0
    return cleaned


def kernel_fill(sub_field, sub_mask):
    result     = sub_field.copy()
    rows, cols = result.shape
    for _ in range(KERNEL_PASSES):
        zero_ys, zero_xs = np.where((result == 0) & (sub_mask == 1))
        for r, c in zip(zero_ys, zero_xs):
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and sub_mask[nr, nc] == 1
                            and result[nr, nc] > 0):
                        neighbors.append(result[nr, nc])
            if neighbors:
                top_val, top_cnt = Counter(neighbors).most_common(1)[0]
                if top_cnt >= NEIGHBOR_MIN_AGREE:
                    result[r, c] = top_val
    return result


def dominant_crop(pixels_1d):
    valid = pixels_1d[pixels_1d > 0]
    if len(valid) == 0:
        return 0, "No crop"
    val = Counter(valid.tolist()).most_common(1)[0][0]
    return val, CLASS_NAMES.get(val, "Unknown")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    # ── Load shapefile ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  OBIA Post-Processing — Full Rajkot District")
    print("="*60)
    print("\n[1/5] Loading shapefile...")
    gdf = gpd.read_file(SHP_PATH)
    total_fields = len(gdf)
    print(f"      {total_fields:,} field polygons found.")

    # ── Open raster ───────────────────────────────────────────────────────────
    print("\n[2/5] Opening raster image...")
    with rasterio.open(IMG_PATH) as src:
        profile = src.profile.copy()

        if gdf.crs != src.crs:
            print(f"      CRS mismatch — reprojecting shapefile...")
            gdf = gdf.to_crs(src.crs)
            print(f"      Done.")

        print(f"      Reading band 1 (shape: {src.shape})...")
        src_array = src.read(1).astype(np.uint8)
        src_array[(src_array < VALID_MIN) | (src_array > VALID_MAX)] = 0
        valid_px = int(np.sum(src_array > 0))
        print(f"      Valid pixels after range filter: {valid_px:,}")

        out_array    = np.zeros_like(src_array, dtype=np.uint8)
        summary_rows = []
        processed    = 0
        skipped      = 0

        # ── Field loop with tqdm ──────────────────────────────────────────────
        print(f"\n[3/5] Processing {total_fields:,} fields...\n")

        # tqdm bar format explained:
        #   {l_bar}  = left part  "Processing fields: XX%|"
        #   {bar}    = the visual bar █████░░░░
        #   {r_bar}  = right part  "| 1234/5678 [01:23<02:45, 12.34 fields/s]"
        bar_format = (
            "  {desc}: {percentage:5.1f}%|{bar:40}| "
            "{n_fmt}/{total_fmt} fields  "
            "[{elapsed}<{remaining}  {rate_fmt}]"
        )

        with tqdm(
            total      = total_fields,
            desc       = "Processing",
            unit       = " fields",
            bar_format = bar_format,
            colour     = "green",
            dynamic_ncols= True
        ) as pbar:

            for idx, row in gdf.iterrows():
                field_id = row.get("id_0", idx)

                # Rasterize polygon
                try:
                    poly_mask = rasterize(
                        [(row.geometry.__geo_interface__, 1)],
                        out_shape  = src.shape,
                        transform  = src.transform,
                        fill       = 0,
                        dtype      = np.uint8,
                        all_touched= True
                    )
                except Exception:
                    skipped += 1
                    pbar.update(1)
                    continue

                inside_pos = np.where(poly_mask == 1)
                if len(inside_pos[0]) == 0:
                    skipped += 1
                    pbar.update(1)
                    continue

                # Extract + MMU
                field_pixels = apply_mmu(src_array[inside_pos].copy())

                if np.sum(field_pixels > 0) == 0:
                    skipped += 1
                    summary_rows.append({
                        "id_0":          field_id,
                        "dominant_class":0,
                        "dominant_crop": "No crop / skipped",
                        "total_pixels":  len(field_pixels),
                        "valid_pixels":  0,
                        "zero_pixels":   len(field_pixels),
                    })
                    pbar.update(1)
                    continue

                # Build 2D sub-grid
                ys, xs  = inside_pos
                y_min   = ys.min();  y_max = ys.max() + 1
                x_min   = xs.min();  x_max = xs.max() + 1
                sub_field                        = np.zeros((y_max-y_min, x_max-x_min), dtype=np.uint8)
                sub_mask                         = poly_mask[y_min:y_max, x_min:x_max]
                sub_field[ys-y_min, xs-x_min]   = field_pixels

                # Kernel fill
                filled              = kernel_fill(sub_field, sub_mask)
                filled_pixels       = filled[sub_mask == 1]
                out_array[inside_pos] = filled_pixels

                # Summary row
                dom_cls, dom_name   = dominant_crop(filled_pixels)
                crop_counts         = Counter(filled_pixels[filled_pixels > 0].tolist())
                crop_detail         = {CLASS_NAMES.get(k, str(k)): v for k, v in crop_counts.items()}

                summary_rows.append({
                    "id_0":           field_id,
                    "dominant_class": dom_cls,
                    "dominant_crop":  dom_name,
                    "total_pixels":   len(filled_pixels),
                    "valid_pixels":   int(np.sum(filled_pixels > 0)),
                    "zero_pixels":    int(np.sum(filled_pixels == 0)),
                    **{f"px_{k}": v for k, v in crop_detail.items()}
                })

                processed += 1

                # Update bar with current dominant crop shown on the right
                pbar.set_postfix_str(
                    f"last: {dom_name:<12} | done:{processed:>6} skip:{skipped:>4}",
                    refresh=False
                )
                pbar.update(1)

    # ── Save GeoTIFF ──────────────────────────────────────────────────────────
    print(f"\n[4/5] Saving output raster...")
    print(f"      → {OUT_TIF}")
    profile.update(dtype=rasterio.uint8, count=1, compress="lzw", nodata=0)
    with rasterio.open(OUT_TIF, "w", **profile) as dst:
        dst.write(out_array, 1)
    print(f"      Saved.")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving field summary CSV...")
    print(f"      → {OUT_CSV}")
    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"      {len(df):,} rows written.")

    # ── Final report ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    mins    = int(elapsed // 60)
    secs    = int(elapsed %  60)

    print("\n" + "="*60)
    print("  COMPLETED")
    print("="*60)
    print(f"  Total time      : {mins}m {secs}s")
    print(f"  Total fields    : {total_fields:,}")
    print(f"  Processed       : {processed:,}")
    print(f"  Skipped         : {skipped:,}  (empty / geometry error)")
    print(f"  Avg speed       : {processed / elapsed:.1f} fields/sec")
    print(f"  Output raster   : {OUT_TIF}")
    print(f"  Output CSV      : {OUT_CSV}")
    print("="*60)

    # Top crops summary
    df_valid = df[df["dominant_class"] > 0]
    if not df_valid.empty:
        print("\n  Top 10 dominant crops across all Rajkot fields:")
        print(f"  {'Crop':<22} {'Fields':>8}   {'% of total':>10}")
        print(f"  {'-'*22} {'-'*8}   {'-'*10}")
        top = df_valid["dominant_crop"].value_counts().head(10)
        for crop, count in top.items():
            pct = count / total_fields * 100
            print(f"  {crop:<22} {count:>8,}   {pct:>9.1f}%")
    print()


if __name__ == "__main__":
    run()