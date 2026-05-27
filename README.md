# OBIA Post-Processing — Field Inspector
### Rajkot Rabi Crop Classification 2025–26

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What Problem Does This Solve?](#2-what-problem-does-this-solve)
3. [How Satellite Crop Classification Works](#3-how-satellite-crop-classification-works)
4. [File Structure and Inputs](#4-file-structure-and-inputs)
5. [Configuration Parameters](#5-configuration-parameters)
6. [Crop Class Dictionary](#6-crop-class-dictionary)
7. [Full Processing Pipeline — Step by Step](#7-full-processing-pipeline--step-by-step)
   - [Step 1 — Load and Align Data](#step-1--load-and-align-data)
   - [Step 2 — Valid Range Filter](#step-2--valid-range-filter)
   - [Step 3 — Rasterize Field Polygon](#step-3--rasterize-field-polygon)
   - [Step 4 — Extract Field Pixels](#step-4--extract-field-pixels)
   - [Step 5 — MMU Filter (Noise Removal)](#step-5--mmu-filter-noise-removal)
   - [Step 6 — Build 2D Sub-Grid](#step-6--build-2d-sub-grid)
   - [Step 7 — 3-Pass Kernel Fill](#step-7--3-pass-kernel-fill)
   - [Step 8 — Final Summary](#step-8--final-summary)
8. [Noise Removal in Depth](#8-noise-removal-in-depth)
9. [The 3×3 Kernel — Deep Explanation](#9-the-33-kernel--deep-explanation)
10. [Edge Cases and Special Scenarios](#10-edge-cases-and-special-scenarios)
11. [Multi-Crop Fields](#11-multi-crop-fields)
12. [Terminal Output Guide](#12-terminal-output-guide)
13. [Dependencies](#13-dependencies)
14. [How to Run](#14-how-to-run)
15. [Known Limitations](#15-known-limitations)

---

## 1. Project Overview

This script is a **field-level inspector** for OBIA (Object-Based Image Analysis) post-processing. It takes a satellite raster image where every pixel carries a crop class label, and a shapefile containing farm field boundaries. It focuses on a **single target field**, cleans noisy or missing pixel classifications, fills gaps using neighborhood context, and produces a clean crop summary for that field.

The script is designed for **debugging and verification** — it prints the field grid before and after processing so you can visually confirm the classification quality for any field of interest.

---

## 2. What Problem Does This Solve?

When a machine learning model classifies satellite imagery, it works pixel by pixel. The result is never perfect. Common problems include:

- **Salt-and-pepper noise**: A few random pixels inside a Wheat field get incorrectly labelled as Cumin or Fennel. These are isolated errors, not real crops.
- **Unclassified pixels (zeros)**: Some pixels are not classified at all — they may fall on field boundaries, under cloud shadow, or in areas of low spectral confidence.
- **Class leakage**: Pixels near field edges sometimes borrow spectral characteristics from neighboring fields.

This script addresses all three issues using:
1. A **valid range filter** to remove out-of-range junk values
2. An **MMU (Minimum Mapping Unit) filter** to remove statistically insignificant crop classes
3. A **kernel-based majority fill** to intelligently fill unclassified zero pixels using neighborhood voting

---

## 3. How Satellite Crop Classification Works

A satellite captures reflectance values across multiple spectral bands (Red, Green, Near-Infrared, SWIR, etc.) for every pixel. A trained model reads these spectral values and assigns each pixel a crop class number.

```
Satellite pixel → [Red=0.12, NIR=0.45, SWIR=0.23] → Model → Class 1 (Wheat)
```

The output is a raster image (`.tif`) where each pixel stores an integer:

```
1 = Wheat     8 = Cumin    13 = Fennel
2 = Jowar     9 = Coriander  0 = Background / unclassified
```

This is a **discrete categorical label map** — not continuous intensity data. This distinction matters because all filtering and smoothing must work on categories (vote-based), not on numeric values (averaging is meaningless here: average of Wheat(1) and Cumin(8) = 4.5, which is not a real crop).

---

## 4. File Structure and Inputs

```
project/
├── data/
│   ├── sample_rajkot/
│   │   └── sample_rajkot.shp     ← Shapefile with field boundaries
│   └── pixel_sample_rajkot.tif   ← Classified satellite raster
├── output/
│   └── (output files go here)
└── try.py             ← This script
```

**Raster image (`.tif`):**
- Band 1 contains crop class integers (0–36)
- Resolution: typically 10m per pixel (Sentinel-2) or 3m (PlanetScope)
- CRS: must match shapefile (or is auto-reprojected)

**Shapefile (`.shp`):**
- Contains polygon geometries for each farm field
- Each polygon has an `id_0` attribute — the unique field identifier
- Used to create the field mask (which pixels belong to this field)

---

## 5. Configuration Parameters

```python
TARGET_ID = 00000      # Field id_0 from shapefile to inspect

VALID_MIN = 1            # Minimum valid crop class (1 = Wheat is included)
VALID_MAX = 36           # Maximum valid crop class
MMU_THRESHOLD = 0.02     # 2% — crop classes below this fraction are noise
NEIGHBOR_MIN_AGREE = 4   # Minimum neighbors that must agree to fill a zero pixel
```

### Why VALID_MIN = 1 (not 2)?

Class 1 is **Wheat**. If VALID_MIN were set to 2, all Wheat pixels would be zeroed out before processing — making Wheat detection impossible. 

### Why MMU_THRESHOLD = 0.02?

A threshold of 2% means: if a crop class covers fewer than 2% of the valid pixels in the field, it is treated as noise and removed. For large Wheat fields, 1–2 isolated Cumin pixels (from classifier error) are unlikely to represent real Cumin cultivation. This threshold can be tuned — lower it to preserve more minority crops; raise it to be more aggressive in noise removal.

### Why NEIGHBOR_MIN_AGREE = 4?

Out of 8 possible neighbors in the 3×3 window, requiring 4 to agree means a **majority** of the available neighbors must support the fill decision. This prevents filling pixels based on weak or ambiguous evidence. For corner pixels with fewer than 4 neighbors available, the pixel may remain unfilled unless all available neighbors agree.

---

## 6. Crop Class Dictionary

| Value | Crop | Symbol | Value | Crop | Symbol |
|-------|------|--------|-------|------|--------|
| 0 | Background | ` . ` | 18 | Math | `Ma ` |
| 1 | Wheat | `Wh ` | 19 | Mung | `Mg ` |
| 2 | Jowar | `Jo ` | 20 | Bajra | `Ba ` |
| 3 | Maize | `Mz ` | 21 | Chikori | `Ci ` |
| 4 | Gram | `Gr ` | 22 | Ajwain | `Aj ` |
| 5 | Mustard | `Mu ` | 23 | Rajko | `Rk ` |
| 6 | Sugarcane | `Su ` | 24 | Rajgira | `Rg ` |
| 7 | Tobacco | `To ` | 25 | Indianbean | `Ib ` |
| 8 | Cumin | `Cu ` | 26 | Cowpea | `Cp ` |
| 9 | Coriander | `Co ` | 27 | Lentil | `Le ` |
| 10 | Garlic | `Ga ` | 28 | Fenugreek | `Fg ` |
| 11 | Sawa | `Sa ` | 29 | Jute | `Ju ` |
| 12 | Isabgul | `Is ` | 30 | Urid | `Ur ` |
| 13 | Fennel | `Fe ` | 31 | Sweet Potato | `Sp ` |
| 14 | Onion | `On ` | 32 | Rabi Sown | `Rs ` |
| 15 | Potato | `Po ` | 33 | Kalonji | `Kl ` |
| 16 | Vegetables | `Ve ` | 34 | Chilli | `Ch ` |
| 17 | Other Crops | `Oc ` | 35 | Pea | `Pe ` |

---

## 7. Full Processing Pipeline — Step by Step

### Step 1 — Load and Align Data

```python
with rasterio.open(IMG_PATH) as src:
    gdf = gpd.read_file(SHP_PATH)
    gdf = gdf[gdf["id_0"] == TARGET_ID]
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)
```

**What happens:**
- The raster image is opened with `rasterio`. This gives access to the pixel data, coordinate reference system (CRS), and transform (the mapping from pixel coordinates to geographic coordinates).
- The shapefile is loaded with `geopandas` and immediately filtered to only the field with `id_0 == TARGET_ID`.
- The CRS of both datasets is compared. If they use different coordinate systems (e.g., shapefile is in WGS84/EPSG:4326 and raster is in UTM), the shapefile is reprojected to match the raster. This ensures the polygon lines up correctly over the pixels.

**Why CRS alignment matters:**
If the CRS does not match, the polygon may be shifted by hundreds of meters, causing the rasterized field mask to cover the wrong pixels entirely.

---

### Step 2 — Valid Range Filter

```python
src_array = src.read(1).astype(np.uint8)
src_array[(src_array < VALID_MIN) | (src_array > VALID_MAX)] = 0
```

**What happens:**
- Band 1 of the raster is read as a 2D NumPy array and cast to `uint8` (unsigned 8-bit integer, values 0–255).
- Any pixel value below `VALID_MIN` (1) or above `VALID_MAX` (36) is set to 0.

**Why this is needed:**
The classifier may produce values outside the valid class range for pixels affected by clouds, shadows, water bodies, or sensor anomalies. Values like 0 (already background), 100, or 255 are not valid crop classes and must be removed before analysis. Setting them to 0 marks them as unclassified/background.

---

### Step 3 — Rasterize Field Polygon

```python
geom = gdf.iloc[0].geometry
poly_mask = rasterize(
    [(geom.__geo_interface__, 1)],
    out_shape=src.shape,
    transform=src.transform,
    all_touched=True
)
```

**What happens:**
- The field polygon is converted into a binary raster array (`poly_mask`) of the same size as the satellite image.
- Every pixel whose center (or any part of it, due to `all_touched=True`) falls inside the polygon gets a value of 1. Everything outside is 0.

**Result:**
```
Full raster poly_mask (example):
0 0 0 0 0 0 0 0 0
0 0 1 1 1 0 0 0 0
0 0 1 1 1 1 0 0 0
0 0 0 1 1 1 0 0 0
0 0 0 0 0 0 0 0 0
```
This mask acts as a stencil — only pixels where `poly_mask == 1` belong to the target field.

**Why `all_touched=True`?**
Without it, only pixels whose center point falls inside the polygon are included. Boundary pixels would be missed, leading to underrepresentation of field edges. `all_touched=True` includes any pixel that the polygon boundary passes through, giving more complete field coverage.

---

### Step 4 — Extract Field Pixels

```python
inside_pos = np.where(poly_mask == 1)
field_pixels = src_array[inside_pos].copy()
```

**What happens:**
- `np.where(poly_mask == 1)` returns two arrays: the row indices and column indices of every pixel inside the field.
- These indices are used to extract the corresponding crop class values from the cleaned raster, giving a 1D array of all crop values for this field.

**Example:**
```
inside_pos = ([2, 2, 2, 3, 3, 3, 3], [2, 3, 4, 3, 4, 5, 6])
                 ↑ row indices          ↑ col indices

field_pixels = [1, 1, 0, 1, 8, 1, 1]
               Wh Wh  .  Wh Cu Wh Wh
```

---

### Step 5 — MMU Filter (Noise Removal)

```python
valid = field_pixels[field_pixels > 0]
if len(valid) > 0:
    counts = Counter(valid)
    total_valid = len(valid)
    for cls, cnt in counts.items():
        if cnt / total_valid < MMU_THRESHOLD:
            field_pixels[field_pixels == cls] = 0
```

**What happens:**
- Only non-zero pixels are considered (zeros are already unclassified).
- The frequency of each crop class is calculated as a fraction of total valid pixels.
- Any crop class with a fraction below `MMU_THRESHOLD` (2%) is removed — all its pixels are reset to 0.

**Example:**
```
Field has 500 valid pixels:
  Wheat (1):  482 pixels → 96.4% → KEEP  ✓
  Cumin (8):    8 pixels →  1.6% → REMOVE ✗  (below 2%)
  Fennel (13): 10 pixels →  2.0% → KEEP  ✓  (exactly at threshold)

After MMU filter:
  Wheat (1):  482 pixels ✓
  Cumin (8):    0 pixels (set to 0 — treated as noise)
  Fennel (13): 10 pixels ✓
```

**Why MMU?**
This is the key noise removal step. Isolated, statistically insignificant crop classes in a field are almost always classifier errors — not real crops. The MMU filter removes them cleanly before gap filling begins, so the kernel fill does not accidentally propagate noise values.

**Relationship to salt-and-pepper noise:**
The scattered, isolated misclassified pixels are a form of salt-and-pepper noise — named because they appear as random bright or dark dots in an otherwise uniform region, like salt and pepper scattered on a surface. The MMU filter is specifically designed to target this pattern by identifying statistically rare classes and eliminating them.

---

### Step 6 — Build 2D Sub-Grid

```python
ys, xs = inside_pos
y_min, y_max = ys.min(), ys.max() + 1
x_min, x_max = xs.min(), xs.max() + 1
sub_field = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
sub_mask = poly_mask[y_min:y_max, x_min:x_max]
sub_field[ys - y_min, xs - x_min] = field_pixels
```

**What happens:**
- The bounding box of the field (minimum and maximum row/column) is computed.
- A small sub-array (`sub_field`) is created covering only this bounding box.
- The field mask is also sliced to the same bounding box (`sub_mask`).
- The 1D `field_pixels` array is placed back into 2D positions in `sub_field`.

**Why not work on the full raster?**

| Working on full raster | Working on sub-grid |
|------------------------|---------------------|
| 5000 × 5000 = 25 million pixels | e.g., 20 × 15 = 300 pixels |
| Very slow | Very fast |
| Hard to print/debug | Easy to visualize |
| Neighbors may include other fields | Neighbors controlled by sub_mask |

The sub-grid is purely a performance and debugging optimization. The field shape is preserved exactly through `sub_mask`.

---

### Step 7 — 3-Pass Kernel Fill

```python
result = sub_field.copy()
for pass_num in range(1, 4):
    zero_ys, zero_xs = np.where((result == 0) & (sub_mask == 1))
    for r, c in zip(zero_ys, zero_xs):
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if (0 <= nr < result.shape[0] and
                    0 <= nc < result.shape[1] and
                    sub_mask[nr, nc] == 1 and
                    result[nr, nc] > 0):
                    neighbors.append(result[nr, nc])

        if neighbors:
            top_val, top_cnt = Counter(neighbors).most_common(1)[0]
            if top_cnt >= NEIGHBOR_MIN_AGREE:
                result[r, c] = top_val
```

**What happens:**
This is the core gap-filling algorithm. It runs 3 complete passes over all zero pixels inside the field mask. In each pass:

1. Find all pixels that are zero AND inside the field (mask == 1).
2. For each such pixel, examine its 8 surrounding neighbors (3×3 window, center excluded).
3. Only neighbors inside the field mask with non-zero values are collected.
4. The most common crop class among those neighbors is identified.
5. If that most common class appears at least `NEIGHBOR_MIN_AGREE` times (4), the zero pixel is filled with that class.

**Why 3 passes?**
A single pass fills pixels that already have enough valid neighbors. But some pixels — especially those near edges, corners, or triangle tips — have too few neighbors in pass 1 because the surrounding pixels are also zero. After pass 1 fills the obvious pixels, those newly filled pixels become valid neighbors for pass 2. Pass 3 similarly builds on pass 2. Values "flow" from the filled interior toward isolated corners and edges.

**Why in-place updating?**
The algorithm reads from and writes to the same `result` array within each pass. This means a pixel filled in step 10 of a pass can immediately serve as a neighbor for pixel 11. This propagation behavior accelerates filling and is intentional — it is sometimes called **iterative in-place modal filling**.

**Why majority vote and not averaging?**
Crop classes are categorical labels. Averaging Wheat(1) and Cumin(8) gives 4.5, which does not correspond to any real crop. Majority voting is the only statistically valid method for categorical data. It is equivalent to using a **mode filter** — replacing a value with the most frequent value in its neighborhood.

**3×3 window — why this size?**
- A 1×3 or 3×1 window only checks left/right or up/down neighbors — it misses diagonal context, which matters for irregular field shapes.
- A 5×5 window checks pixels up to 2 positions away, which may reach pixels from a different part of the field or even from outside the mask. It is also 4× slower.
- The 3×3 window checks all 8 immediate neighbors — the closest possible context — making it the right balance between accuracy and efficiency for this task.

**Boundary handling:**
Two boundary conditions are enforced:
- `0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]`: prevents index-out-of-bounds errors at the edges of the sub-array.
- `sub_mask[nr, nc] == 1`: ensures only pixels inside the field polygon contribute. Pixels from neighboring fields are invisible to the algorithm.

---

### Step 8 — Final Summary

```python
print_grid("GRID AFTER 3 PASSES", result, sub_mask)

final_crops = Counter(result[sub_mask == 1])
for val, count in final_crops.items():
    if val > 0:
        print(f"{CLASS_NAMES.get(val, 'Unknown')}: {count} pixels")
```

**What happens:**
- The filled grid is printed to the terminal for visual inspection.
- All pixels inside the mask are counted by class. Zero pixels (unfilled) are excluded from the summary.
- The result shows how many pixels of each crop class exist in the field after post-processing.

---

## 8. Noise Removal in Depth

### Type of noise targeted: Salt-and-Pepper

Salt-and-pepper noise refers to isolated, randomly placed pixels with incorrect values surrounded by a uniform correct region:

```
Before noise removal:

[Wh][Wh][Wh][Wh][Wh]
[Wh][Wh][Cu][Wh][Wh]   ← Cu pixel = one "pepper" dot in Wheat field
[Wh][Wh][Wh][Wh][Wh]
[Wh][Fe][Wh][Wh][Wh]   ← Fe pixel = another isolated error
[Wh][Wh][Wh][Wh][Wh]

After MMU filter (Cu=1/25=4%, Fe=1/25=4%, threshold=2% — both survive):
→ Neither removed by MMU. A lower threshold or a replace-non-zero filter would be needed.

After MMU filter (Cu=1/100=1%, threshold=2%):
→ Cu removed ✓. All Cu pixels → 0. Those zeros are candidates for kernel fill.
```

### Why not Gaussian noise removal?

Gaussian noise applies small random perturbations to **all** pixel values — it shifts continuous intensity values slightly across the entire image. This is relevant for raw satellite band data but not for classified label maps. After classification, a pixel is either correct or wrong — there is no "slightly wrong" crop label. Only salt-and-pepper noise (discrete, isolated, random) applies here.

### Why not Crimmins speckle removal?

Crimmins' algorithm operates by comparing pixel intensities along 4 directional scans and pulling outlier values toward their neighbors. This requires continuous numeric intensity values. Crop class labels (1, 8, 13) are discrete categories with no meaningful numeric ordering — Crimmins would treat Wheat(1) as numerically less than Cumin(8) and produce meaningless results.

### Noise removal category used

| Category | Technique Used | Purpose |
|----------|---------------|---------|
| Noise removal | MMU filter | Remove statistically rare / isolated crop classes |
| Gap filling | Kernel majority vote (3 passes) | Fill unclassified zero pixels |
| Smoothing | Implicit — majority fill produces uniform regions | Reduces fragmentation |
| Edge enhancement | Not used | Not needed — boundaries come from the shapefile |
| Edge extraction | Not used | Not needed — field shape already known |

---

## 9. The 3×3 Kernel — Deep Explanation

### Window layout

```
Position in 3×3 window:

[(-1,-1)][(-1, 0)][(-1,+1)]
[( 0,-1)][ CENTER ][( 0,+1)]
[(+1,-1)][(+1, 0)][(+1,+1)]

8 neighbors checked (center excluded via: if dr == 0 and dc == 0: continue)
```

### How it handles irregular field shapes

The field polygon is typically not a rectangle. It may be triangular, L-shaped, or have curved edges. The `sub_mask` array handles this — only pixels where `sub_mask == 1` participate as valid neighbors.

For a pixel near the tip of a triangular field:

```
Triangle field — tip pixel at Row 0:

         [Wh]        ← Row 0, the tip
        [Wh][Wh]     ← Row 1
       [Wh][Wh][Wh]  ← Row 2

3×3 window at tip pixel:
[X ][X ][X ]    X = outside field (mask=0) — ignored
[X ][TIP][X ]
[X ][Wh][Wh]    Only 2 valid neighbors

2 < NEIGHBOR_MIN_AGREE(4) → pixel stays 0 in pass 1.
After pass 1 fills Row 1 and Row 2 fully:
→ pass 2: tip now has 2 filled neighbors (still 2 < 4) → stays 0
→ pass 3: still 2 < 4 → remains unfilled.

Result: tip pixel with only 1–2 reachable neighbors may remain zero.
This is correct behavior — insufficient evidence to fill.
```

### Tie-breaking

When two crop classes receive equal vote counts:

```
Neighbors: [Wh, Wh, Wh, Wh, Cu, Cu, Cu, Cu]
Wheat = 4 votes, Cumin = 4 votes — perfect tie.

Python's Counter.most_common(1) returns the first-seen class in case of tie.
Wheat wins (appeared first in the array iteration).
top_cnt = 4 >= 4 → FILL with Wheat.

Note: Tie resolution is arbitrary. In a genuinely 50/50 mixed field, 
the result depends on pixel scan order, not on any meaningful spatial logic.
```

### When 3 crops compete

```
Neighbors: [Wh, Wh, Wh, Wh, Wh, Cu, Cu, Fe]
Wheat = 5, Cumin = 2, Fennel = 1

most_common(1) = Wheat (5 votes)
5 >= 4 → FILL with Wheat ✓

If no majority (e.g., Wh=3, Cu=3, Fe=2):
most_common = Wheat or Cumin (tie, top_cnt=3)
3 < 4 → SKIP. Pixel stays 0. Correct behavior — ambiguous boundary.
```

### Is this convolution?

True convolution multiplies pixel values by a kernel weight matrix and sums the result. This script does something different — it performs a **modal filter** (replacing a value with the neighborhood mode). It is convolution-inspired in structure (sliding window, neighborhood aggregation) but operates on categorical data using voting rather than arithmetic. Convolution on categorical labels would produce meaningless non-integer results.

---

## 10. Edge Cases and Special Scenarios

### Case 1 — Triangle tip with no source data

```
START:
Row 0: [ 0 ]        ← No Wheat data at all in top rows
Row 1: [ 0 ][ 0 ]
Row 2: [Wh][ 0 ][Wh]
Row 3: [Wh][Wh][ 0 ][Wh]
Row 4: [Wh][Wh][Wh][Wh][Wh]

AFTER 3 PASSES:
Row 0: [ . ]        ← Remains unfilled — no source value to propagate from
Row 1: [ . ][ . ]   ← Remains unfilled — never reaches 4 valid neighbors
Row 2: [Wh][Wh][Wh] ← Filled from Row 3/4 context
Row 3: [Wh][Wh][Wh][Wh]
Row 4: [Wh][Wh][Wh][Wh][Wh]

Summary: Wheat: 12 pixels. Unfilled: 3 pixels.
```

**Explanation:** The 3-pass fill cannot create crop values from nothing. If there are no labeled pixels nearby to serve as source, the tip remains unfilled. This is the correct and safe behavior — inventing a classification without evidence would be wrong.

### Case 2 — Last 1 or 2 pixels unfilled

Unfilled pixels are zero values in the final `result` array. The summary loop (`if val > 0`) skips them. They appear as `" . "` in the printed grid but do not affect the crop summary. 2 unfilled pixels in a 200-pixel field represent a 1% gap — acceptable for most agricultural mapping purposes.

### Case 3 — Entire field has no valid pixels after MMU

If MMU removes all crop classes (all were below 2%), `field_pixels` becomes all zeros. The kernel fill has nothing to propagate, and the final summary prints nothing. This indicates the classifier produced no reliable classification for this field and it requires manual review.

---

## 11. Multi-Crop Fields

### Two crops in separate zones (Wheat top half, Cumin bottom half)

```
Field layout:
[Wh][Wh][Wh][Wh]   → Wheat zone
[Wh][Wh][Wh][Wh]
[Cu][Cu][Cu][Cu]   → Cumin zone
[Cu][Cu][Cu][Cu]

MMU check (50% Wheat, 50% Cumin):
Both well above 2% → both kept ✓

Kernel fill behavior:
Zero pixels in Wheat zone → surrounded by Wheat → filled with Wheat ✓
Zero pixels in Cumin zone → surrounded by Cumin → filled with Cumin ✓
Zero pixels at the boundary → mixed neighbors (Wh and Cu) → may stay 0 if no majority

Final summary:
Wheat: ~N pixels
Cumin: ~N pixels
→ Both crops correctly reported.
```

### Three or more crops

With 3+ crops, the MMU filter keeps all crops above 2%. The kernel fill fills zero pixels with whichever crop dominates the local neighborhood. At boundaries between crop zones, pixels with balanced mixed neighborhoods remain unfilled (zero). This is correct — the script does not guess at ambiguous boundary pixels.

### One crop with a single-row band of another

```
[Wh][Wh][Wh][Wh][Wh]
[Cu][Cu][Cu][Cu][Cu]   ← 1 row of Cumin (minority)
[Wh][Wh][Wh][Wh][Wh]

If total field = 15 pixels (5 Wh + 5 Cu + 5 Wh):
Cumin = 5/15 = 33% → well above 2% → KEEP

Zero pixels in the Cumin row get filled with Cumin.
Zero pixels in the Wheat rows get filled with Wheat.
Boundary zeros (between rows) may be contested and remain 0.

Final summary:
Wheat: ~10 pixels
Cumin: ~5 pixels
```

**Important limitation:** If Cumin were only 1–2 scattered pixels (not a cohesive band), it would fall below MMU threshold and be removed — even if those pixels represent a real thin strip of Cumin. The MMU threshold is set for the expected minimum meaningful crop patch size and should be calibrated to field conditions.

---

## 12. Terminal Output Guide

### Grid symbols

```
" . "  →  Zero pixel (background or unfilled)
" X "  →  Outside field (mask = 0)
"Wh "  →  Wheat (class 1)
"Jo "  →  Jowar (class 2)
"Cu "  →  Cumin (class 8)
(see full SYM dictionary in code for all classes)
```

### Example output

```
--- GRID BEFORE FILL (Wheat should now show as Wh) ---
r00| X   X  Wh  X   X 
r01| X  Wh  Wh  Wh  X 
r02|Wh  Wh   .  Wh  Wh
r03|Wh   .  Wh  Wh  Wh
r04|Wh  Wh  Wh  Wh  Wh

--- GRID AFTER 3 PASSES ---
r00| X   X  Wh  X   X 
r01| X  Wh  Wh  Wh  X 
r02|Wh  Wh  Wh  Wh  Wh
r03|Wh  Wh  Wh  Wh  Wh
r04|Wh  Wh  Wh  Wh  Wh

--- FINAL CROP SUMMARY ---
Wheat: 19 pixels
```

**Reading the grid:**
- Compare BEFORE and AFTER to see which ` . ` pixels were filled and with what crop
- Pixels that remain ` . ` after 3 passes did not have enough neighboring agreement
- ` X ` pixels are outside the field boundary — they are never modified

---

## 13. Dependencies

```
Python 3.8+

numpy          — array operations and pixel indexing
geopandas      — shapefile loading and CRS operations
rasterio       — satellite raster reading and polygon rasterization
shapely        — geometry operations (used via geopandas)
collections    — Counter for pixel class frequency counting
```

Install with:

```bash
pip install numpy geopandas rasterio shapely
```

---

## 14. How to Run

1. Set `TARGET_ID` to the `id_0` value of the field you want to inspect.
2. Set `SHP_PATH` to the path of your shapefile.
3. Set `IMG_PATH` to the path of your classified raster `.tif`.
4. Adjust thresholds if needed (`MMU_THRESHOLD`, `NEIGHBOR_MIN_AGREE`).
5. Run the script:

```bash
python field_inspector.py
```

The script prints the before/after field grid and final crop summary to the terminal.

---

## 15. Known Limitations

| Limitation | Description | Workaround |
|------------|-------------|------------|
| Triangle/corner pixels may stay unfilled | Pixels with fewer than 4 valid neighbors never receive a fill | Lower `NEIGHBOR_MIN_AGREE` to 2 for a final cleanup pass, or use nearest-neighbor fill as fallback |
| MMU removes real minority crops | A real crop patch covering < 2% of the field is removed as noise | Lower `MMU_THRESHOLD` for fields known to have intentional multi-crop patterns |
| Tie resolution is arbitrary | When two classes tie in neighbor votes, Python's Counter resolves by insertion order | No clean fix — ambiguous boundary pixels are better left at 0 |
| Only fills zero pixels | Existing wrong (non-zero) labels are not replaced by the kernel fill | Run MMU first to convert wrong labels to 0 before fill; or add a majority-replace pass for non-zero pixels |
| Single-pass fill leaves isolated corner zeros | 3 passes helps but cannot always reach extreme corners | Add a 4th pass or a fallback nearest-neighbor fill for remaining zeros |
| No output raster written | This script is inspection-only — it does not write a post-processed `.tif` | See `v6` / `v8` of the full pipeline script which includes raster output |
