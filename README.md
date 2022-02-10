# NeonSpeciesBenchmark

A tree species classification benchmark for the National Ecological Observatory Network.

# Goal

# Contents


# Usage


# Scores

# Citation

# Dev Guide

This is intended for users within the Weecology Lab. The raw neon sensor data is >30TB and can be downloaded from the [NEON data portal](https://data.neonscience.org/). The config.yml in this repo points at resources at the University of Florida.

To convert the raw data in data/raw/neon_vst_data_2022.csv into geospatial points. 

```
python prepare_raw_data.py
```

which creates 
* data/processed/canopy_points.shp
* data/processed/train.csv
* data/processed/test.csv

To crop the RGB, HSI and CHM sensor data for each of these data

```
python create_benchmark.py
```
