import osmnx as ox
import ee
import geemap
import os

# 1. Initialize Earth Engine with your new Cloud Project
PROJECT_ID = 'norse-blade-473221-f3'
try:
    ee.Initialize(project=PROJECT_ID)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

def automated_data_pipeline(city_name="Bengaluru, Karnataka, India", output_dir="./data/raw/training_data"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting zero-manual-effort pipeline for {city_name}...")

    # ==========================================
    # STEP A: Extract OSM Ground-Truth Vector Masks
    # ==========================================
    print("Pulling OpenStreetMap vector layers...")
    # Pull the driveable mathematical road graph automatically
    graph = ox.graph_from_place(city_name, network_type='drive')
    nodes, edges = ox.graph_to_gdfs(graph)
    
    # Save the ground truth as a GeoJSON reference annotation
    osm_path = os.path.join(output_dir, "osm_ground_truth.geojson")
    edges.to_file(osm_path, driver="GeoJSON")
    print(f"OSM Vector Data saved to: {osm_path}")

    # ==========================================
    # STEP B: Extract Matching Sentinel-2 Feed
    # ==========================================
    print("Aligning bounding box and pulling Sentinel-2 satellite feed...")
    # Get the bounding box of the city from the OSM graph
    bbox = ox.utils_geo.bbox_from_point(ox.geocode(city_name), dist=5000)
    north, south, east, west = bbox
    
    # Create an Earth Engine geometry representing the city's bounding box
    region = ee.Geometry.Rectangle([west, south, east, north])

    # Query the openly available Sentinel-2 dataset, filter by the city region and low cloud cover
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate('2023-01-01', '2023-12-31')
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                  .median()) # Take the median to remove transient clouds/shadows

    # Select RGB bands for spatial visualization
    image = collection.select(['B4', 'B3', 'B2'])

    # Export the aligned satellite imagery to your local machine
    sentinel_path = os.path.join(output_dir, "sentinel2_aligned.tif")
    geemap.ee_export_image(image, filename=sentinel_path, scale=10, region=region)
    print(f"Multi-resolution satellite feed saved to: {sentinel_path}")
    print("Pipeline Complete. Data readiness secured.")

if __name__ == "__main__":
    automated_data_pipeline()
