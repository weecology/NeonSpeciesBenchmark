#Test TreeData
from src import utils
from src import data
import pandas as pd
import os
import pytest

@pytest.fixture() 
def ROOT():
    ROOT = os.path.dirname(os.path.dirname(data.__file__))
    return ROOT

@pytest.fixture()
def config(ROOT, tmpdir):
    print("Creating global config")
    #Turn of CHM filtering for the moment
    config = utils.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    config["rgb_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["min_train_samples"] = 1
    config["min_test_samples"] = 1
    config["crop_dir"] = tmpdir
    config["classes"] = 3
    config["convert_h5"] = False
    config["min_CHM_diff"] = None    
    
    return config

def test_TreeData_setup(config, ROOT):
    #One site's worth of data
    config["regenerate"] = True 
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)               
    dm = data.TreeData(config=config, csv_file=csv_file, data_dir="{}/tests/data".format(ROOT), debug=True) 
    dm.setup()  
    
    test = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))
    train = pd.read_csv("{}/tests/data/processed/train.csv".format(ROOT))
    
    assert not test.empty
    assert not train.empty
