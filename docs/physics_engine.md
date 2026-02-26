# The Physics Engine (`spectral_ops`)

The Physics Engine is the heart of SpectraLoRA. It operates directly on raw, multi-spectral PyTorch tensors to extract physical meaning before the deep learning model sees the data. 

The engine currently tracks 5 distinct indices, optimized for both dense vegetation and arid, desert environments.

## The Spectral Fingerprint
The function `get_spectral_fingerprint(x, band_indices)` takes a satellite image tensor and returns a global context vector $z$ of shape `(Batch, 5)`. This vector acts as the input to the Gating Network.

### 1. NDVI (Normalized Difference Vegetation Index)
The standard metric for identifying live green vegetation.

$$NDVI = \frac{NIR - Red}{NIR + Red + \epsilon}$$

### 2. SAVI (Soil-Adjusted Vegetation Index)
Crucial for arid regions (like Sudan or the Middle East) where bright sand reflects light and drowns out the chlorophyll signal of sparse shrubs.

$$SAVI = \frac{(NIR - Red) \cdot (1 + L)}{NIR + Red + L + \epsilon}$$

*(Default $L = 0.5$)*

### 3. NDWI (Normalized Difference Water Index)
Used to identify open water bodies.

$$NDWI = \frac{Green - NIR}{Green + NIR + \epsilon}$$

### 4. NDBI (Normalized Difference Built-up Index)
Used to map urban areas, concrete, and asphalt.

$$NDBI = \frac{SWIR - NIR}{SWIR + NIR + \epsilon}$$

### 5. BSI (Bare Soil Index)
Distinguishes "Bare Earth" (Sand/Dirt) from "Built Structures". This is essential to stop the AI from confusing desert dunes with concrete buildings.

$$BSI = \frac{(SWIR + Red) - (NIR + Blue)}{(SWIR + Red) + (NIR + Blue) + \epsilon}$$