
import json
from dataclasses import asdict, dataclass
from os.path import splitext

class SavableConfig:
    """A generic Config dataclass that can be saved to a json format"""

    def save_to_json(self, path: str) -> None:
        """Save this instance as json

        Parameters
        ----------
        path : str
            The .json filepath, e.g. 'directory/config.json'
        """
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)
    
    def load_from_json(self, path: str) -> None:
        """Load this instance from a json file

        Parameters
        ----------
        path : str
            The .json filepath, e.g. 'directory/config.json'
        """
        with open(path, "r") as f:
            data = json.load(f)
            for key, value in data.items():
                setattr(self, key, value)

@dataclass
class PrepocessConfig(SavableConfig):
    """All parameters for the preprocess step
    
    Parameters
    ----------
    Version : str
        The version of the MIMIC-IV dataset to use, e.g. "2.2"
    RawDataPath : str
        The path to the raw MIMIC-IV data, e.g. "data/mimic-IV/2.2"
    PreprocessedDataPath : str
        The path to save the preprocessed data, e.g. "data/output"
    Task : str
        The prediction task to preprocess for, e.g. "mortality", "LengthOfStay", "readmission", "phenotype"
    Include_ICU : bool
        Whether to include ICU stays in the dataset
    Include_Diagnosis : bool
        Whether to include diagnosis codes in the dataset
    Include_Procedures : bool
        Whether to include procedure codes in the dataset
    Include_Medications : bool
        Whether to include medication codes in the dataset
    Include_chart_event : bool
        Whether to include chart events in the dataset
    Include_output_event : bool
        Whether to include output events in the dataset
    Include_HF_patients : bool
        Whether to include patients with heart failure in the dataset
    Include_COPD_patients : bool
        Whether to include patients with chronic obstructive pulmonary disease in the dataset
    Include_CKD_patients : bool
        Whether to include patients with chronic kidney disease in the dataset
    Include_CAD_patients : bool
        Whether to include patients with coronary artery disease in the dataset
    Outliers_management : str
        The method to handle outliers in the dataset, e.g. "remove", "impute_mean", "impute_median", "impute_mode", "impute_random", "none"
    Outliers_threshold : float
        The Z-score threshold for outlier detection, e.g. 3.0
    Time_window_reference : str
        The reference point for the time window, e.g. "admission", "discharge"
    Time_window_size : int
        The size of the time window in hours, e.g. 24 for the first 24 hours of the ICU stay
    Time_window_bucket_size : int
        The size of the time buckets in hours, e.g. 1 for hourly buckets
    Time_prediction_horizon : int
        The prediction horizon in hours, e.g. 2 for predicting within the next 2 hours
    Missing_values_management : str
        The method to handle missing values in the dataset, e.g. "impute_mean", "impute_median", "impute_mode", "impute_random", "none"

    """


    Version: str = "2.2"
    # Paths
    RawDataPath: str = "data/mimic-IV/2.2"
    PreprocessedDataPath: str = "data/output"
    # Preprocessing parameters
    Task: str = "mortality"  # mortality, LengthOfStay, readmission , phenotype 
    # task specific parameters 
    Mortality_prediction_horizon: int = 2  # in hours, e.g. 2 for predicting mortality within next 24 hours
    LengthOfStay_greater_or_equal_threshold: int = 3  # in days, e.g. 3 for predicting if Length of Stay is greater or equal to 3 days 
    Readmission_number_of_days_threshold: int = 30  # in days, e.g. 30 for predicting readmission within 30 days 
    Phenotype: str = "HF"  # HF, COPD, CKD, CAD 
    Phenotype_prediction_horizon: int = 24  # in hours, e.g. 24 for predicting phenotype within the first 24 hours of the ICU stay 

    
    Include_ICU: bool = True
    Include_Diagnosis: bool = True
    Include_Procedures: bool = True
    Include_Medications: bool = True
    Include_chart_event: bool = True 
    Include_output_event: bool = True 
    
    Disease_Filter: str = "None"  # None, HF, COPD, CKD, CAD 
    Outliers_management: str = "remove"  # remove, impute_mean , impute_median, impute_mode, impute_random, none 
    Outliers_threshold_right: float = 98.0  # right treshold for outlier detection, e.g. 98.0 for the 98th percentile  
    Outliers_threshold_left: float = 0.0  # left treshold for outlier detection, e.g. 0.0 for the 0th percentile (no left outliers)

    Time_window_size: int = 24  # in hours, e.g. 24 for first 24 hours 
    Time_window_bucket_size: int = 1  # in hours, e.g. 1 for hourly buckets 

    Missing_values_management: str = "mean"  # mean, median, mode, random, none 

    def save_to_json(self, path: str | None = None):
        """Saves this config at `path` if provided, else in the same place as `self.out_path`"""
        if path is None:
            path, _ = splitext(self.out_path)

        super().save_to_json(path + ".json")

    def load_from_json(self, path: str | None = None):
        """Loads this config from `path` if provided, else from the same place as `self.out_path`"""
        if path is None:
            path, _ = splitext(self.out_path)

        super().load_from_json(path)

    # def print 
    def __str__(self) -> str:
        """String representation of this config"""
        config_dict = asdict(self)
        config_str = "PreprocessConfig:\n"
        for key, value in config_dict.items():
            config_str += f"  {key}: {value}\n"
        return config_str

    def __repr__(self) -> str:
        """String representation of this config"""
        return self.__str__()



    
