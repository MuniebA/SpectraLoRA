import rasterio
from rasterio.windows import Window
import os

def create_chips(input_tiff, output_dir, prefix, chip_size=224, max_chips=10):
    print(f"🔪 Chopping {input_tiff} into {chip_size}x{chip_size} chips...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(input_tiff) as src:
        meta = src.meta.copy()
        width, height = src.width, src.height
        
        chip_count = 0
        
        # Slide a window across the large image
        for row in range(0, height - chip_size, chip_size):
            for col in range(0, width - chip_size, chip_size):
                if chip_count >= max_chips:
                    return
                
                window = Window(col, row, chip_size, chip_size)
                transform = src.window_transform(window)
                
                # Update metadata for the small chip
                meta.update({
                    'height': chip_size,
                    'width': chip_size,
                    'transform': transform
                })
                
                # Read the 6 bands for this window
                data = src.read(window=window)
                
                # Skip chips that are mostly blank (edges of satellite passes)
                if data.max() == 0:
                    continue
                    
                # We add the prefix here so filenames are unique!
                output_path = os.path.join(output_dir, f"{prefix}_{chip_count:03d}.tif")
                
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(data)
                
                print(f"✅ Saved {output_path}")
                chip_count += 1

if __name__ == "__main__":
    # Our two new master files and a prefix for their chip names
    master_files = [
        ("master_test_doha.tif", "doha"),
        ("master_test_southkhartoum.tif", "skh")
    ]
    
    # NEW FOLDER: Keep test data separate from training data!
    output_folder = "test_dataset_224x224"
    
    # Generate 10 chips from EACH image (20 test chips total)
    for tiff_file, prefix in master_files:
        if os.path.exists(tiff_file):
            create_chips(tiff_file, output_folder, prefix=prefix, max_chips=10)
        else:
            print(f"⚠️ Warning: Could not find {tiff_file}. Did the download finish?")
            
    print("🎉 Test dataset generation complete!")