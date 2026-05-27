"""
OBIA Post-Processing Script v6 — Rajkot Rabi Classification 2025-26
=====================================================================
FIELD INSPECTOR MODE: Processes only TARGET_ID and traces Kernel steps.
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from collections import Counter
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ─── TARGET FIELD CONFIGURATION ──────────────────────────────────────────────
TARGET_ID =     579157                       
579215 # Set the field id_0 you want to inspect

# ─── PATHS ────────────────────────────────────────────────────────────────────
SHP_PATH = r"C:\Amnex-Learning\April_task\rajkot_postprocessing\data\sample_rajkot\sample_rajkot.shp"
IMG_PATH = r"C:\Amnex-Learning\April_task\rajkot_postprocessing\data\pixel_sample_rajkot.tif"
OUT_PATH = r"C:\Amnex-Learning\April_task\rajkot_postprocessing\output\obia_postprocessed_v8.tif"

# ─── VALID CLASS RANGE ────────────────────────────────────────────────────────
VALID_MIN = 2
VALID_MAX = 36

# ─── THRESHOLDS ───────────────────────────────────────────────────────────────
COVERAGE_THRESHOLD  = 0.03
MMU_THRESHOLD       = 0.10
FILL_THRESHOLD      = 0.70
NEIGHBOR_MIN_AGREE  = 5
MAJORITY_MIN_AGREE  = 5

# ─── COLOR TABLE & NAMES ──────────────────────────────────────────────────────
COLORMAP = {
    0:  (0,   0,   0,   255),
    2:  (244, 164, 96,  255), 3:  (64,  224, 208, 255), 4:  (255, 215, 0,   255),
    5:  (147, 112, 219, 255), 6:  (255, 105, 180, 255), 7:  (207, 250, 77,  255),
    8:  (139, 10,  50,  255), 9:  (210, 180, 140, 255), 10: (0,   250, 154, 255),
    11: (255, 250, 205, 255), 12: (59,  113, 147, 255), 13: (144, 238, 144, 255),
    14: (173, 255, 47,  255), 15: (255, 99,  71,  255), 16: (139, 69,  19,  255),
    17: (37,  167, 49,  255), 18: (138, 133, 130, 255), 19: (107, 142, 35,  255),
    20: (100, 149, 237, 255), 21: (194, 152, 72,  255), 22: (38,  153, 141, 255),
    23: (220, 20,  60,  255), 24: (0,   158, 224, 255), 25: (184, 198, 152, 255),
    26: (170, 132, 170, 255), 27: (176, 196, 222, 255), 28: (29,  75,  28,  255),
    29: (214, 175, 170, 255), 30: (137, 41,  133, 255), 31: (222, 165, 225, 255),
    32: (231, 83,  125, 255), 33: (138, 133, 130, 255), 34: (255, 155, 255, 255),
    35: (37,  167, 49,  255), 36: (252, 170, 128, 255),
}

# CLASS_NAMES = {
#     2:"Wheat", 3:"Jowar", 4:"Maize", 5:"Gram", 6:"Mustard", 7:"Sugarcane", 8:"Tobacco",
#     9:"Cumin", 10:"Coriander", 11:"Garlic", 12:"Sawa", 13:"Isabgul", 14:"Fennel",
#     15:"Onion", 16:"Potato", 17:"Vegetables", 18:"Other Crops", 19:"Math", 20:"Mung",
#     21:"Bajra", 22:"Chikori", 23:"Ajwain", 24:"Rajko", 25:"Rajgira", 26:"Indianbean",
#     27:"Cowpea", 28:"Lentil", 29:"Fenugreek", 30:"Jute", 31:"Urid", 32:"Sweet Potato",
#     33:"Rabi Sown", 34:"Kalonji", 35:"Chilli", 36:"Pea",
# }

# Symbols for grid printing
SYM = {0: " . ", 9: "Cu ", 4: "Mz ", 2: "Wh ", 5: "Gm ", 10: "Co "}

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def dump_invalid_pixels(arr):
    cleaned = arr.copy()
    cleaned[(cleaned < VALID_MIN) | (cleaned > VALID_MAX)] = 0
    return cleaned

def apply_mmu_filter(pixel_1d):
    valid = pixel_1d[pixel_1d > 0]
    if len(valid) == 0: return pixel_1d
    counts = Counter(valid.tolist()); total = len(valid); cleaned = pixel_1d.copy()
    for cls, cnt in counts.items():
        if cnt / total < MMU_THRESHOLD: cleaned[cleaned == cls] = 0
    return cleaned

def print_grid(label, arr, mask):
    print(f"\n--- {label} ---")
    for r in range(arr.shape[0]):
        row_str = f"r{r:02d}|"
        for c in range(arr.shape[1]):
            if mask[r,c] == 0: row_str += " X  "
            else:
                val = arr[r,c]
                row_str += SYM.get(val, f"{val:02d} ")
        print(row_str)

def neighbor_fill_zeros(field_2d, mask_2d):
    result = field_2d.copy()
    rows, cols = field_2d.shape
    zero_ys, zero_xs = np.where((field_2d == 0) & (mask_2d == 1))
    filled = 0
    step = 0

    print_grid("GRID BEFORE KERNEL FILL", field_2d, mask_2d)

    for r, c in zip(zero_ys, zero_xs):
        step += 1
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and mask_2d[nr, nc] == 1 and field_2d[nr, nc] > 0):
                    neighbors.append(field_2d[nr, nc])
        
        if not neighbors: continue
        top_val, top_cnt = Counter(neighbors).most_common(1)[0]
        
        # Decision logic with terminal trace
        print(f"Step {step}: ({r},{c}) neighbors found: {top_cnt} of Class {top_val}", end="")
        if top_cnt >= NEIGHBOR_MIN_AGREE or (top_cnt == len(neighbors) and len(neighbors) >= 2):
            result[r, c] = top_val
            filled += 1
            print(f" -> [FILL: {CLASS_NAMES.get(top_val)}]")
        else:
            print(" -> [SKIP]")

    print_grid("GRID AFTER KERNEL FILL", result, mask_2d)
    return result, filled

def apply_majority_filter(array, boundary_mask):
    result = array.copy(); rows, cols = array.shape
    ys, xs = np.where(boundary_mask == 1)
    if len(ys) == 0: return result, 0
    y_min, y_max, x_min, x_max = max(0, ys.min()-1), min(rows, ys.max()+2), max(0, xs.min()-1), min(cols, xs.max()+2)
    sub = array[y_min:y_max, x_min:x_max].copy(); sub_mask = boundary_mask[y_min:y_max, x_min:x_max]; sub_out = sub.copy()
    changed = 0
    for r in range(1, sub.shape[0]-1):
        for c in range(1, sub.shape[1]-1):
            if sub_mask[r,c] != 1 or sub[r,c] == 0: continue
            nb = [sub[r+dr, c+dc] for dr in [-1,0,1] for dc in [-1,0,1] if not (dr==0 and dc==0) and sub_mask[r+dr,c+dc]==1 and sub[r+dr,c+dc]>0]
            if not nb: continue
            val, cnt = Counter(nb).most_common(1)[0]
            if val != sub[r,c] and cnt >= MAJORITY_MIN_AGREE:
                sub_out[r,c] = val; changed += 1
    result[y_min:y_max, x_min:x_max] = sub_out
    return result, changed

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run():
    t_start = time.time()
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf = gpd.read_file(SHP_PATH)
    
    # FILTER FOR TARGET FIELD
    gdf = gdf[gdf["id_0"] == TARGET_ID]
    if gdf.empty:
        print(f"Error: Field id_0 {TARGET_ID} not found."); return

    src = rasterio.open(IMG_PATH)
    if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)
    src_array = dump_invalid_pixels(src.read(1).astype(np.uint8))
    
    out_array = np.zeros_like(src_array, dtype=np.uint8)
    boundary_mask = np.zeros_like(src_array, dtype=np.uint8)
    stats = {"assigned":0, "zeros_filled":0}

    for idx, row in gdf.iterrows():
        poly_mask = rasterize([(row.geometry.__geo_interface__, 1)], out_shape=src.shape, transform=src.transform, fill=0, dtype=np.uint8, all_touched=True)
        inside_pos = np.where(poly_mask == 1)
        if len(inside_pos[0]) == 0: continue
        
        field_pixels = apply_mmu_filter(src_array[inside_pos].copy())
        if int(np.sum(field_pixels > 0)) == 0: continue

        ys, xs = inside_pos
        y_min, y_max, x_min, x_max = ys.min(), ys.max()+1, xs.min(), xs.max()+1
        sub_field = np.zeros((y_max-y_min, x_max-x_min), dtype=np.uint8)
        sub_field[ys-y_min, xs-x_min] = field_pixels
        sub_mask = poly_mask[y_min:y_max, x_min:x_max]

        # PROCESS FIELD WITH TRACE
        sub_filled, n_filled = neighbor_fill_zeros(sub_field, sub_mask)
        stats["zeros_filled"] += n_filled
        out_array[y_min:y_max, x_min:x_max][sub_mask == 1] = sub_filled[sub_mask == 1]
        boundary_mask[y_min:y_max, x_min:x_max][sub_mask == 1] = 1
        stats["assigned"] += 1

    out_array, changed_px = apply_majority_filter(out_array, boundary_mask)
    
    print(f"\nSummary for ID {TARGET_ID}: Filled {stats['zeros_filled']} pixels. Majority changed {changed_px}.")
    src.close()

if __name__ == "__main__":
    run()





