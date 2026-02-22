import numpy as np
import rasterio
from rasterio.transform import from_origin

print("🌍 Creating a 6-Band Multispectral Calibration GeoTIFF...")

# 224x224 is the exact size Prithvi expects
width, height = 224, 224
num_bands = 6 

# Create an empty array (uint16 is the standard format for Sentinel-2 surface reflectance)
data = np.zeros((num_bands, height, width), dtype=np.uint16)

# We will divide the image into 4 quadrants with REAL physical reflectance values (scaled by 10000)
# Band Order: [Blue, Green, Red, NIR, SWIR1, SWIR2]

# 1. Dense Forest (High NIR, Low Red)
forest_sig = [200, 400, 200, 3500, 1500, 600]
# 2. Deep Water (High Green, Low NIR/SWIR)
water_sig  = [500, 600, 400, 100, 50, 50]
# 3. Urban/Concrete (High SWIR, High Red)
urban_sig  = [1200, 1400, 1600, 2000, 2500, 2200]
# 4. Bare Soil/Desert (Very High SWIR and Red)
desert_sig = [1500, 2000, 2800, 3200, 4500, 3500]

# Paint the quadrants
for b in range(num_bands):
    data[b, :112, :112] = forest_sig[b]   # Top-Left
    data[b, :112, 112:] = water_sig[b]    # Top-Right
    data[b, 112:, :112] = urban_sig[b]    # Bottom-Left
    data[b, 112:, 112:] = desert_sig[b]   # Bottom-Right

# Add slight natural variance (noise) so it's not a perfectly flat color
noise = np.random.randint(-50, 50, size=(num_bands, height, width), dtype=np.int16)
data = np.clip(data.astype(np.int32) + noise, 0, 10000).astype(np.uint16)

# Create geographic coordinates for the GeoTIFF metadata
transform = from_origin(500000, 4000000, 30, 30) 

with rasterio.open(
    'prithvi_test_sample.tif', 'w',
    driver='GTiff', height=height, width=width,
    count=num_bands, dtype=data.dtype,
    crs='EPSG:32633', transform=transform
) as dst:
    dst.write(data)

print("✅ Success! 'prithvi_test_sample.tif' has been created in your folder.")