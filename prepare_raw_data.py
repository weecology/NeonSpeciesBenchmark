#Create train test split
from src.data import TreeData
data_module = TreeData("data/raw/neon_vst_data_2022.csv")
data_module.setup()