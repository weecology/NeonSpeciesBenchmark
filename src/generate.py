#Convert NEON field sample points into bounding boxes of cropped image data for model training
import glob
import os
import pandas as pd
from src.neon_paths import find_sensor_path, lookup_and_convert, bounds_to_geoindex
from src import patches
from distributed import wait   
import traceback
import warnings

warnings.filterwarnings('ignore')

def write_crop(row, img_path, savedir, replace=True):
    """Wrapper to write a crop based on size and savedir"""
    if replace == False:
        filename = "{}/{}.tif".format(savedir, row["individual"])
        file_exists = os.path.exists(filename)
        if file_exists:
            annotation = pd.DataFrame({"image_path":[filename], "taxonID":[row["taxonID"]], "plotID":[row["plotID"]], "individualID":[row["individual"]], "RGB_tile":[row["RGB_tile"]], "siteID":[row["siteID"]],"box_id":[row["box_id"]]})
            return annotation            
        else:
            filename = patches.crop(bounds=row["geometry"].bounds, sensor_path=img_path, savedir=savedir, basename=row["individual"])  
    else:
        filename = patches.crop(bounds=row["geometry"].bounds, sensor_path=img_path, savedir=savedir, basename=row["individual"])
        annotation = pd.DataFrame({"image_path":[filename], "taxonID":[row["taxonID"]], "plotID":[row["plotID"]], "individualID":[row["individual"]], "RGB_tile":[row["RGB_tile"]], "siteID":[row["siteID"]],"box_id":[row["box_id"]]})
        return annotation

def generate_crops(gdf, sensor_glob, savedir, rgb_glob, client=None, convert_h5=False, HSI_tif_dir=None, replace=True):
    """
    Given a shapefile of crowns in a plot, create pixel crops and a dataframe of unique names and labels"
    Args:
        shapefile: a .shp with geometry objects and an taxonID column
        savedir: path to save image crops
        img_pool: glob to search remote sensing files. This can be either RGB of .tif hyperspectral data, as long as it can be read by rasterio
        client: optional dask client
        convert_h5: If HSI data is passed, make sure .tif conversion is complete
        rgb_glob: glob to search images to match when converting h5s -> tif.
        HSI_tif_dir: if converting H5 -> tif, where to save .tif files. Only needed if convert_h5 is True
    Returns:
       annotations: pandas dataframe of filenames and individual IDs to link with data
    """
    annotations = []
    
    img_pool = glob.glob(sensor_glob, recursive=True)
    rgb_pool = glob.glob(rgb_glob, recursive=True)
    
    #There were erroneous point cloud .tif
    img_pool = [x for x in img_pool if not "point_cloud" in x]
    rgb_pool = [x for x in rgb_pool if not "point_cloud" in x]
     
    #Looking up the rgb -> HSI tile naming is expensive and repetitive. Create a dictionary first.
    gdf["geo_index"] = gdf.geometry.apply(lambda x: bounds_to_geoindex(x.bounds))
    tiles = gdf["geo_index"].unique()
    
    tile_to_path = {}
    for geo_index in tiles:
        try:
            #Check if h5 -> tif conversion is complete
            if convert_h5:
                if rgb_glob is None:
                    raise ValueError("rgb_glob is None, but convert_h5 is True, please supply glob to search for rgb images")
                else:
                    img_path = lookup_and_convert(rgb_pool=rgb_pool, hyperspectral_pool=img_pool, savedir=HSI_tif_dir,  geo_index = geo_index)
            else:
                img_path = find_sensor_path(lookup_pool = img_pool, geo_index = geo_index)  
        except:
            print("{} failed to find sensor path with traceback {}".format(geo_index, traceback.print_exc()))
            continue
        tile_to_path[geo_index] = img_path
            
    if client:
        futures = []
        for index, row in gdf.iterrows():
            try:
                img_path = tile_to_path[row["geo_index"]]
            except:
                continue
            future = client.submit(write_crop,row=row,img_path=img_path, savedir=savedir, replace=replace)
            futures.append(future)
            
        wait(futures)
        for x in futures:
            try:
                annotation = x.result()
                annotations.append(annotation)                
            except:
                print("Future failed with {}".format(traceback.print_exc()))
    else:
        for index, row in gdf.iterrows():
            try:
                img_path = tile_to_path[row["geo_index"]]
            except:
                continue
            try:
                annotation = write_crop(row=row, img_path=img_path, savedir=savedir, replace=replace)
            except Exception as e:
                print("index {} failed with {}".format(index,e))
                continue
    
            annotations.append(annotation)
    
    annotations = pd.concat(annotations)
        
    return annotations
        