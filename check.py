import rasterio

file_path = r"C:\Amnex-Learning\April_task\rajkot_postprocessing\data\pixel_sample_rajkot.tif"  # Put your filename here

with rasterio.open(file_path) as src:
    print(f"--- File Report for: {file_path} ---")
    print(f"Number of Bands: {src.count}")
    print(f"Data Type: {src.dtypes[0]}")
    print(f"Width x Height: {src.width} x {src.height}")
    
    if src.count == 1:
        print("\nRESULT: This is likely a 1-band Classification map.")
    elif src.count >= 6:
        print("\nRESULT: This is likely a Multi-band RAW image (Ready for Prithvi!)")