# The Geospatial Data Catalog

In standard machine learning, the PyTorch `DataLoader` blindly reads files from a local directory. It has no concept of *where* those images belong on Earth. 

With v0.2.0, SpectraLoRA introduces a hybrid, spatially-aware `DataLoader` backed by **PostgreSQL and PostGIS**.

## The Architecture
![PostGIS Integration](assets/postgis_integration.png)

When PostGIS is enabled, your training loop stops reading local folders and instead executes spatial SQL queries to fetch highly specific training data.

### 1. Enable PostGIS
Ensure your database has the spatial extension enabled:
```sql
CREATE EXTENSION postgis;
```

### 2. Configure Your Environment

Set your database URL in a `.env` file at the root of your project:

```env
SPECTRALORA_DB_URL="postgresql://username:password@localhost:5432/spectralora_db"

```

### 3. Ingest Your Data

Use the `ingest_satellite_folder` utility to scan your raw satellite imagery. SpectraLoRA will read the CRS tags, translate the boundaries to EPSG:4326 (Latitude/Longitude), and save the exact Polygon geometries into the database.

```python
from spectra_lora import ingest_satellite_folder
ingest_satellite_folder("path/to/imagery")

```

### 4. Dynamic Training Filters

Pass a `spatial_filter` directly into your PyTorch dataset. If the database connection drops, or if a user downloads your code without PostGIS installed, SpectraLoRA will **gracefully fall back** to standard local folder reading.

```python
from experiments.train import RealSatelliteDataset
from torch.utils.data import DataLoader

# Query PostGIS for high-quality, low-cloud images
dataset = RealSatelliteDataset(
    folder_path="dataset_224x224", 
    spatial_filter={"max_cloud_cover": 5.0} 
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

```
