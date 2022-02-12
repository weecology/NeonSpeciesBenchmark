#Create train test split
from src.data import TreeData
from src import start_cluster

client = start_cluster.start(cpus=100)
data_module = TreeData("data/raw/neon_vst_data_2022.csv", client = client)
data_module.setup()