#Create benchmark
import geopandas as gpd
import pandas as pd
from src import generate
from src import utils
from src import start_cluster
import os
import shutil

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

config = utils.read_config("config.yml")
gdf = gpd.read_file("data/processed/canopy_points.shp")
gdf = gdf[~gdf.individual.str.contains("contrib")]
gdf["RGB_tile"] = None

#HSI crops
client = start_cluster.start(cpus=100)
#client = None

#Fixed boxes 5m from point 
gdf["points"] = gdf.geometry
gdf["geometry"] = gdf.buffer(5)

#dummy schema
gdf["box_id"] = None

annotations = generate.generate_crops(gdf, sensor_glob=config["HSI_sensor_pool"], savedir="/blue/ewhite/b.weinstein/species_benchmark/HSI/", rgb_glob=config["rgb_sensor_pool"], client=client, convert_h5=True, HSI_tif_dir=config["HSI_tif_dir"])
generate.generate_crops(gdf, sensor_glob=config["rgb_sensor_pool"], savedir="/blue/ewhite/b.weinstein/species_benchmark/RGB/", rgb_glob=config["rgb_sensor_pool"], client=client)
generate.generate_crops(gdf, sensor_glob=config["CHM_pool"], savedir="/blue/ewhite/b.weinstein/species_benchmark/CHM/", rgb_glob=config["rgb_sensor_pool"], client=client)

gdf["geometry"] = gdf["points"]
gdf = gdf[["taxonID","individual","siteID","eventID","stemDiamet","elevation","canopyPosi","utmZone","itcEasting","itcNorthin","CHM_height","geometry"]]
gdf["label"] = gdf.taxonID.astype("category").cat.codes
train_annotations = gdf[gdf.individual.isin(train.individualID)]
test_annotations = gdf[gdf.individual.isin(test.individualID)]

#Image coordinates of the point

for i in train_annotations.individual:
    shutil.copy("/blue/ewhite/b.weinstein/species_benchmark/CHM/{}.tif".format(i),"/blue/ewhite/b.weinstein/species_benchmark/zenodo/train/CHM/{}.tif".format(i,i))
    shutil.copy("/blue/ewhite/b.weinstein/species_benchmark/RGB/{}.tif".format(i),"/blue/ewhite/b.weinstein/species_benchmark/zenodo/train/RGB/{}.tif".format(i,i))
    shutil.copy("/blue/ewhite/b.weinstein/species_benchmark/HSI/{}.tif".format(i),"/blue/ewhite/b.weinstein/species_benchmark/zenodo/train/HSI/{}.tif".format(i,i))

train_annotations.to_file("/blue/ewhite/b.weinstein/species_benchmark/zenodo/train/label.shp")

for i in test_annotations.individual:
    shutil.copy("/blue/ewhite/b.weinstein/species_benchmark/CHM/{}.tif".format(i),"/blue/ewhite/b.weinstein/species_benchmark/zenodo/test/CHM/{}.tif".format(i,i))
    shutil.copy("/blue/ewhite/b.weinstein/species_benchmark/RGB/{}.tif".format(i),"/blue/ewhite/b.weinstein/species_benchmark/zenodo/test/RGB/{}.tif".format(i,i))
    shutil.copy("/blue/ewhite/b.weinstein/species_benchmark/HSI/{}.tif".format(i),"/blue/ewhite/b.weinstein/species_benchmark/zenodo/test/HSI/{}.tif".format(i,i))

test_annotations.to_file("/blue/ewhite/b.weinstein/species_benchmark/zenodo/test/label.shp")
