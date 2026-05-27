import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from collections import Counter
import os
import warnings

warnings.filterwarnings("ignore")

# ─── TARGET FIELD CONFIGURATION ──────────────────────────────────────────────
# This will focus only on your specific Wheat field
TARGET_ID = 577344

# ─── PATHS ────────────────────────────────────────────────────────────────────
SHP_PATH = r"C:\Amnex-Learning\April_task\rajkot_postprocessing\data\sample_rajkot\sample_rajkot.shp"
IMG_PATH = r"C:\Amnex-Learning\April_task\rajkot_postprocessing\data\pixel_sample_rajkot.tif"

# ─── CORRECTED THRESHOLDS ─────────────────────────────────────────────────────
VALID_MIN = 1           # CRITICAL:  Wheat
VALID_MAX = 36
MMU_THRESHOLD = 0.02    # large wheat field lessthen 0.02 cu come then remove them
NEIGHBOR_MIN_AGREE = 4  # Slightly more aggressive filling

# ─── CORRECTED DICTIONARY (Value 1 = Wheat) ──────────────────────────────────
CLASS_NAMES = {
    0: "Background",
    1: "Wheat", 2: "Jowar", 3: "Maize", 4: "Gram", 5: "Mustard",
    6: "Sugarcane", 7: "Tobacco", 8: "Cumin", 9: "Coriander", 10: "Garlic",
    11: "Sawa", 12: "Isabgul", 13: "Fennel", 14: "Onion", 15: "Potato",
    16: "Vegetables", 17: "Other Crops", 18: "Math", 19: "Mung", 20: "Bajra",
    21: "Chikori", 22: "Ajwain", 23: "Rajko", 24: "Rajgira", 25: "Indianbean",
    26: "Cowpea", 27: "Lentil", 28: "Fenugreek", 29: p"Jute", 30: "Urid",
    31: "Sweet Potato", 32: "Rabi Sown", 33: "Kalonji", 34: "Chilli", 35: "Pea"
}

# Symbols for the terminal grid print-out
SYM = {0: " . ", 1: "Wh ", 2: "Jo ", 3: "Mz ", 4: "Gr ", 5: "Mu ", 6: "Su ", 7: "To ", 8: "Cu ", 9: "Co ", 10: "Ga ", 11: "Sa ", 12: "Is ", 13: "Fe ", 14: "On ", 15: "Po ", 16: "Ve ", 17: "Oc ", 18: "Ma ", 19: "Mg ", 20: "Ba ", 21: "Ci ", 22: "Aj ", 23: "Rk ", 24: "Rg ", 25: "Ib ", 26: "Cp ", 27: "Le ", 28: "Fg ", 29: "Ju ", 30: "Ur ", 31: "Sp ", 32: "Rs ", 33: "Kl ", 34: "Ch ", 35: "Pe "}


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

def run_inspector():
    print(f"Inspecting Field ID: {TARGET_ID}...")
    
    with rasterio.open(IMG_PATH) as src:
        gdf = gpd.read_file(SHP_PATH)
        gdf = gdf[gdf["id_0"] == TARGET_ID]
        
        if gdf.empty:
            print("Error: Field ID not found in shapefile.")
            return

        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # 1. Read and Clean Raster
        src_array = src.read(1).astype(np.uint8)
        src_array[(src_array < VALID_MIN) | (src_array > VALID_MAX)] = 0

        # 2. Rasterize the specific field
        geom = gdf.iloc[0].geometry
        poly_mask = rasterize([(geom.__geo_interface__, 1)], 
                              out_shape=src.shape, 
                              transform=src.transform, 
                              all_touched=True)

        # 3. Extract and Apply MMU
        inside_pos = np.where(poly_mask == 1)
        field_pixels = src_array[inside_pos].copy()
        
        # MMU Logic
        valid = field_pixels[field_pixels > 0]
        if len(valid) > 0:
            counts = Counter(valid)
            total_valid = len(valid)
            for cls, cnt in counts.items():
                if cnt / total_valid < MMU_THRESHOLD:
                    field_pixels[field_pixels == cls] = 0

        # 4. Create local 2D Grid
        ys, xs = inside_pos
        y_min, y_max, x_min, x_max = ys.min(), ys.max()+1, xs.min(), xs.max()+1
        sub_field = np.zeros((y_max-y_min, x_max-x_min), dtype=np.uint8)
        sub_mask = poly_mask[y_min:y_max, x_min:x_max]
        sub_field[ys-y_min, xs-x_min] = field_pixels

        print_grid("GRID BEFORE FILL (Wheat should now show as Wh)", sub_field, sub_mask)

        # 5. Kernel Fill Logic (3 passes to let color flow)
        result = sub_field.copy()
        for pass_num in range(1, 4):
            zero_ys, zero_xs = np.where((result == 0) & (sub_mask == 1))
            for r, c in zip(zero_ys, zero_xs):
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < result.shape[0] and 0 <= nc < result.shape[1] 
                            and sub_mask[nr, nc] == 1 and result[nr, nc] > 0):
                            neighbors.append(result[nr, nc])
                
                if neighbors:
                    top_val, top_cnt = Counter(neighbors).most_common(1)[0]
                    if top_cnt >= NEIGHBOR_MIN_AGREE:
                        result[r, c] = top_val

        print_grid(f"GRID AFTER 3 PASSES", result, sub_mask)
        
        final_crops = Counter(result[sub_mask == 1])
        print("\n--- FINAL CROP SUMMARY ---")
        for val, count in final_crops.items():
            if val > 0:
                print(f"{CLASS_NAMES.get(val, 'Unknown')}: {count} pixels")

if __name__ == "__main__":
    run_inspector()