#Ligthning data module
from . import __file__
from distributed import wait
import glob
import geopandas as gpd
import numpy as np
import os
import pandas as pd
from pytorch_lightning import LightningDataModule
from shapely.geometry import Point
from src import CHM
from src import utils
from torch.utils.data import Dataset
        
def filter_data(path, config):
    """Transform raw NEON data into clean shapefile   
    Args:
        config: DeepTreeAttention config dict, see config.yml
    """
    field = pd.read_csv(path)
    field = field[~field.itcEasting.isnull()]
    field = field[~field.growthForm.isin(["liana","small shrub"])]
    field = field[~field.growthForm.isnull()]
    field = field[~field.plantStatus.isnull()]        
    field = field[field.plantStatus.str.contains("Live")]    
    
    groups = field.groupby("individualID")
    shaded_ids = []
    for name, group in groups:
        shaded = any([x in ["Full shade", "Mostly shaded"] for x in group.canopyPosition.values])
        if shaded:
            if any([x in ["Open grown", "Full sun"] for x in group.canopyPosition.values]):
                continue
            else:
                shaded_ids.append(group.individualID.unique()[0])
        
    field = field[~(field.individualID.isin(shaded_ids))]
    field = field[(field.height > 3) | (field.height.isnull())]
    field = field[field.stemDiameter > config["min_stem_diameter"]]
    field.loc[field.taxonID=="PSMEM","taxonID"] = "PSME"
    
    field = field[~field.taxonID.isin(["BETUL", "FRAXI", "HALES", "PICEA", "PINUS", "QUERC", "ULMUS", "2PLANT"])]
    field = field[~(field.eventID.str.contains("2014"))]
    with_heights = field[~field.height.isnull()]
    with_heights = with_heights.loc[with_heights.groupby('individualID')['height'].idxmax()]
    
    missing_heights = field[field.height.isnull()]
    missing_heights = missing_heights[~missing_heights.individualID.isin(with_heights.individualID)]
    missing_heights = missing_heights.groupby("individualID").apply(lambda x: x.sort_values(["eventID"],ascending=False).head(1)).reset_index(drop=True)
  
    field = pd.concat([with_heights,missing_heights])
    
    #remove multibole
    field = field[~(field.individualID.str.contains('[A-Z]$',regex=True))]

    #List of hand cleaned errors
    known_errors = ["NEON.PLA.D03.OSBS.03422","NEON.PLA.D03.OSBS.03422","NEON.PLA.D03.OSBS.03382", "NEON.PLA.D17.TEAK.01883"]
    field = field[~(field.individualID.isin(known_errors))]
    field = field[~(field.plotID == "SOAP_054")]
    
    #Create shapefile
    field["geometry"] = [Point(x,y) for x,y in zip(field["itcEasting"], field["itcNorthing"])]
    shp = gpd.GeoDataFrame(field)
    
    #HOTFIX, BLAN has some data in 18N UTM, reproject to 17N update columns
    BLAN_errors = shp[(shp.siteID == "BLAN") & (shp.utmZone == "18N")]
    BLAN_errors.set_crs(epsg=32618, inplace=True)
    BLAN_errors.to_crs(32617,inplace=True)
    BLAN_errors["utmZone"] = "17N"
    BLAN_errors["itcEasting"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][0])
    BLAN_errors["itcNorthing"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][1])
    
    #reupdate
    shp.loc[BLAN_errors.index] = BLAN_errors
    
    #Oak Right Lab has no AOP data
    shp = shp[~(shp.siteID.isin(["PUUM","ORNL"]))]

    #There are a couple NEON plots within the OSBS megaplot, make sure they are removed
    shp = shp[~shp.plotID.isin(["OSBS_026","OSBS_029","OSBS_039","OSBS_027","OSBS_036"])]

    #subset columns
    shp = shp[["geometry","individualID","taxonID","eventID","pointID","plotID","siteID","height","utmZone","itcEasting","itcNorthing","canopyPosition","stemDiameter"]]
    
    return shp

def sample_plots(shp, min_train_samples=5, min_test_samples=3, iteration = 1):
    """Sample and split a pandas dataframe based on plotID
    Args:
        shp: pandas dataframe of filtered tree locations
        test_fraction: proportion of plots in test datasets
        min_samples: minimum number of samples per class
        iteration: a dummy parameter to make dask submission unique
    """
    #split by plot level
    plotIDs = shp.plotID.unique()
    if len(plotIDs) == 0:
        test = shp[shp.plotID == shp.plotID.unique()[0]]
        train = shp[shp.plotID == shp.plotID.unique()[1]]
        
        return train, test
                
    np.random.shuffle(plotIDs)
    test = shp[shp.plotID == plotIDs[0]]
    
    for plotID in plotIDs[1:]:
        include = False
        selected_plot = shp[shp.plotID == plotID]
        # If any species is missing from min samples, include plot
        for x in selected_plot.taxonID.unique():
            if sum(test.taxonID == x) < min_test_samples:
                include = True
        if include:
            test = pd.concat([test,selected_plot])
            
    train = shp[~shp.plotID.isin(test.plotID.unique())]
    
    #remove fixed boxes from test
    test = test.groupby("taxonID").filter(lambda x: x.shape[0] >= min_test_samples)
    train = train[train.taxonID.isin(test.taxonID)]    
    test = test[test.taxonID.isin(train.taxonID)]
    
    return train, test
    
def train_test_split(shp, config, client = None):
    """Create the train test split
    Args:
        shp: a filter pandas dataframe (or geodataframe)  
        client: optional dask client
    Returns:
        None: train.shp and test.shp are written as side effect
        """    
    min_sampled = config["min_train_samples"] + config["min_test_samples"]
    keep = shp.taxonID.value_counts() > (min_sampled)
    species_to_keep = keep[keep].index
    shp = shp[shp.taxonID.isin(species_to_keep)]
    print("splitting data into train test. Initial data has {} points from {} species with a min of {} samples".format(shp.shape[0],shp.taxonID.nunique(),min_sampled))
    test_species = 0
    ties = []
    if client:
        futures = [ ]
        for x in np.arange(config["iterations"]):
            future = client.submit(sample_plots, shp=shp, min_train_samples=config["min_train_samples"], iteration=x, min_test_samples=config["min_test_samples"])
            futures.append(future)
        
        wait(futures)
        for x in futures:
            train, test = x.result()
            if test.taxonID.nunique() > test_species:
                print("Selected test has {} points and {} species".format(test.shape[0], test.taxonID.nunique()))
                saved_train = train
                saved_test = test
                test_species = test.taxonID.nunique()
                ties = []
                ties.append([train, test])
            elif test.taxonID.nunique() == test_species:
                ties.append([train, test])          
    else:
        for x in np.arange(config["iterations"]):
            train, test = sample_plots(shp, min_train_samples=config["min_train_samples"], min_test_samples=config["min_test_samples"])
            if test.taxonID.nunique() > test_species:
                print("Selected test has {} points and {} species".format(test.shape[0], test.taxonID.nunique()))
                saved_train = train
                saved_test = test
                test_species = test.taxonID.nunique()
                #reset ties
                ties = []
                ties.append([train, test])
            elif test.taxonID.nunique() == test_species:
                ties.append([train, test])
    
    # The size of the datasets
    if len(ties) > 1:
        print("The size of tied datasets with {} species is {}".format(test_species, [x[1].shape[0] for x in ties]))        
        saved_train, saved_test = ties[np.argmax([x[1].shape[0] for x in ties])]
        
    train = saved_train
    test = saved_test    
    
    #Give tests a unique index to match against
    test["point_id"] = test.index.values
    train["point_id"] = train.index.values
    
    return train, test
    
class TreeData(LightningDataModule):
    """
    Lightning data module to convert raw NEON data into HSI pixel crops based on the config.yml file. 
    The module checkpoints the different phases of setup, if one stage failed it will restart from that stage. 
    Use regenerate=True to override this behavior in setup()
    """
    def __init__(self, csv_file, client = None, config=None, data_dir=None, debug=False):
        """
        Args:
            config: optional config file to override
            data_dir: override data location, defaults to ROOT   
            regenerate: Whether to recreate raw data
            debug: a test mode for small samples
        """
        super().__init__()
        self.ROOT = os.path.dirname(os.path.dirname(__file__))
        self.csv_file = csv_file
        self.debug = debug 
        
        #default training location
        self.client = client
        if data_dir is None:
            self.data_dir = "{}/data/".format(self.ROOT)
        else:
            self.data_dir = data_dir            
        
        self.train_file = "{}/processed/train.csv".format(self.data_dir)
        
        if config is None:
            self.config = utils.read_config("{}/config.yml".format(self.ROOT))   
        else:
            self.config = config
                
    def setup(self,stage=None):
        #Clean data from raw csv, regenerate from scratch or check for progress and complete
        if self.config["regenerate"]:
            if self.config["replace"]:#remove any previous runs
                try:
                    os.remove("{}/processed/canopy_points.shp".format(self.data_dir))
                    os.remove(" ".format(self.data_dir))
                    os.remove("{}/processed/crowns.shp".format(self.data_dir))
                    for x in glob.glob(self.config["crop_dir"]):
                        os.remove(x)
                except:
                    pass
                    
                #Convert raw neon data to x,y tree locatins
                df = filter_data(self.csv_file, config=self.config)
                    

                #Filter points based on LiDAR height
                annotations = CHM.filter_CHM(df, CHM_pool=self.config["CHM_pool"],
                                    min_CHM_height=self.config["min_CHM_height"], 
                                    max_CHM_diff=self.config["max_CHM_diff"], 
                                    CHM_height_limit=self.config["CHM_height_limit"])  
                
                annotations.to_file("{}/processed/canopy_points.shp".format(self.data_dir))
            
            if self.config["new_train_test_split"]:
                train_annotations, test_annotations = train_test_split(df,config=self.config, client=self.client)   
            else:
                previous_train = pd.read_csv("{}/processed/train.csv".format(self.data_dir))
                previous_test = pd.read_csv("{}/processed/test.csv".format(self.data_dir))
                
                train_annotations = annotations[annotations.individualID.isin(previous_train.individualID)]
                test_annotations = annotations[annotations.individualID.isin(previous_test.individualID)]
                
            #capture discarded species
            individualIDs = np.concatenate([train_annotations.individualID.unique(), test_annotations.individualID.unique()])
            novel = annotations[~annotations.individualID.isin(individualIDs)]
            novel = novel[~novel.taxonID.isin(np.concatenate([train_annotations.taxonID.unique(), test_annotations.taxonID.unique()]))]
            novel.to_csv("{}/processed/novel_species.csv".format(self.data_dir))
            
            #Store class labels
            unique_species_labels = np.concatenate([train_annotations.taxonID.unique(), test_annotations.taxonID.unique()])
            unique_species_labels = np.unique(unique_species_labels)
            unique_species_labels = np.sort(unique_species_labels)            
            self.num_classes = len(unique_species_labels)
            
            #Taxon to ID dict and the reverse    
            self.species_label_dict = {}
            for index, taxonID in enumerate(unique_species_labels):
                self.species_label_dict[taxonID] = index
                
            #Store site labels
            unique_site_labels = np.concatenate([train_annotations.siteID.unique(), test_annotations.siteID.unique()])
            unique_site_labels = np.unique(unique_site_labels)
            
            self.site_label_dict = {}
            for index, label in enumerate(unique_site_labels):
                self.site_label_dict[label] = index
            self.num_sites = len(self.site_label_dict)                   
            
            self.label_to_taxonID = {v: k  for k, v in self.species_label_dict.items()}
            
            #Encode the numeric site and class data
            train_annotations["label"] = train_annotations.taxonID.apply(lambda x: self.species_label_dict[x])
            train_annotations["site"] = train_annotations.siteID.apply(lambda x: self.site_label_dict[x])
            
            test_annotations["label"] = test_annotations.taxonID.apply(lambda x: self.species_label_dict[x])
            test_annotations["site"] = test_annotations.siteID.apply(lambda x: self.site_label_dict[x])
            
            train_annotations.to_csv("{}/processed/train.csv".format(self.data_dir), index=False)            
            test_annotations.to_csv("{}/processed/test.csv".format(self.data_dir), index=False)
            
            print("There are {} records for {} species for {} sites in filtered train".format(
                train_annotations.shape[0],
                len(train_annotations.label.unique()),
                len(train_annotations.site.unique())
            ))
            
            print("There are {} records for {} species for {} sites in test".format(
                test_annotations.shape[0],
                len(test_annotations.label.unique()),
                len(test_annotations.site.unique()))
            )