import os
import uuid
from datetime import datetime
import rasterio
from rasterio.warp import transform_bounds
import numpy as np

# Import our Database and Configuration
from spectra_lora.db import SessionLocal, SatelliteChip
from spectra_lora.config import SpectraConfig

def ingest_satellite_folder(folder_path: str):
    """
    Scans a directory of .tif files, extracts their geographic bounding boxes,
    reprojects them to EPSG:4326 (Lat/Lon), and saves them to PostGIS.
    """
    if not os.path.exists(folder_path):
        raise RuntimeError(f"Folder '{folder_path}' not found!")

    config = SpectraConfig()
    db = SessionLocal()
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    print(f"🌍 Starting Ingestion: Found {len(files)} satellite chips in '{folder_path}'.")

    success_count = 0

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        
        # Check if file is already in the database to prevent duplicates
        exists = db.query(SatelliteChip).filter_by(file_path=file_path).first()
        if exists:
            continue

        try:
            with rasterio.open(file_path) as src:
                # 1. Transform Bounds to Standard GPS (EPSG:4326)
                # src.bounds is often in UTM (meters). We must convert to Lat/Lon.
                min_lon, min_lat, max_lon, max_lat = transform_bounds(
                    src.crs, 'EPSG:4326', *src.bounds
                )
                
                # 2. Create the PostGIS WKT (Well-Known Text) Polygon string
                # Format: POLYGON((minx miny, maxx miny, maxx maxy, minx maxy, minx miny))
                wkt_polygon = f"SRID=4326;POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"

                # 3. Calculate Average NDVI (Optional but great for dynamic filtering)
                # rasterio bands are 1-indexed, so we add 1 to our BAND_MAP indices
                red = src.read(config.BAND_MAP['RED'] + 1).astype(np.float32)
                nir = src.read(config.BAND_MAP['NIR'] + 1).astype(np.float32)
                
                ndvi = (nir - red) / (nir + red + 1e-8)
                avg_ndvi = float(np.nanmean(ndvi))

                # 4. Extract Metadata (Mocking Date/Cloud Cover if not present in tags)
                tags = src.tags()
                cloud_cover = float(tags.get('CLOUD_COVER', 0.0))
                
                # 5. Save to Database
                chip = SatelliteChip(
                    chip_id=str(uuid.uuid4()),
                    file_path=file_path,
                    geom=wkt_polygon,
                    avg_ndvi=avg_ndvi,
                    cloud_cover=cloud_cover,
                    acquisition_date=datetime.utcnow() # Fallback date
                )
                
                db.add(chip)
                success_count += 1
                
        except Exception as e:
            print(f"⚠️ Error reading {file_name}: {e}")

    db.commit()
    db.close()
    print(f"✅ Ingestion Complete! Added {success_count} new chips to the PostGIS Spatial Catalog.")