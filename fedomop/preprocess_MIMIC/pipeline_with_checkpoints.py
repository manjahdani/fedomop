from utils.config import PrepocessConfig
import os
from steps import extraction
from steps import feature_selection
from steps import data_generation
from steps import data_generation_icu
from steps import build_dataset
import sys

base_config = PrepocessConfig(
    Version="2.2",
    RawDataPath="mimic-IV/2.2/",
    PreprocessedDataPath="data/output2/",
    Task="Readmission",
    Include_ICU=True,
    Include_Diagnosis=True,
    Include_Procedures=True,
    Include_Medications=True,
    Include_chart_event=True,
    Include_output_event=True,
    Disease_Filter = "HF",
    Outliers_management = "remove",
    Outliers_threshold_right = 98.0,
    Outliers_threshold_left = 0,
    Time_window_size = 24,
    Time_window_bucket_size = 1,
    Missing_values_management = "mean",
    Oversampling = True , 
    Concatenate = False, 
    Output_format = "csv"

)

# -----------------------
# Checkpoint Utilities
# -----------------------

class CheckpointManager:

    def __init__(self, config):
        self.config = config
        self.cohort_output = self._build_cohort_name()
        self.checkpoint_dir = self._init_checkpoint_dir()

    def _build_cohort_name(self):
        use_ICU = str(self.config.Include_ICU)
        label = self.config.Task
        disease_label = self.config.Disease_Filter
        time = self.config.Time_window_size
        bucket = self.config.Time_window_bucket_size

        cohort_output = (
            "cohort_"
            + use_ICU.lower()
            + "_"
            + label.lower().replace(" ", "_")
            + "_"
            + str(time)
            + "_"
            + str(bucket)
            + "_"
            + disease_label
        )

        return cohort_output

    def _init_checkpoint_dir(self):
        path = os.path.join("data/output", self.cohort_output, ".checkpoints")
        os.makedirs(path, exist_ok=True)
        return path

    def is_done(self, step_name: str) -> bool:
        return os.path.exists(
            os.path.join(self.checkpoint_dir, f"{step_name}.done")
        )

    def mark_done(self, step_name: str):
        open(
            os.path.join(self.checkpoint_dir, f"{step_name}.done"),
            "w"
        ).close()

    def reset_step(self, step_name: str):
        file_path = os.path.join(self.checkpoint_dir, f"{step_name}.done")
        if os.path.exists(file_path):
            os.remove(file_path)

# -----------------------
# Functions
# -----------------------

def Extraction(config: PrepocessConfig , root_dir:str , checkpoint:CheckpointManager):
    print("\n")
    print("Extraction step")
    print("\n")
    disease_label=""
    time=0
    label=config.Task 

    if label=='Readmission':
        time = config.Readmission_number_of_days_threshold 
    elif label=='Length of Stay':
        time = config.LengthOfStay_greater_or_equal_threshold 

    if label=='Phenotype':    
        print(("Phenotype selected: ", config.Phenotype))
        if config.Phenotype == 'HF':
            print("HF selected")
            label='Readmission'
            time=config.Phenotype_prediction_horizon
            disease_label='I50'
        elif config.Phenotype == 'CAD':
            label='Readmission'
            time=30
            disease_label='I25'
        elif config.Phenotype == 'CKD': 
            label='Readmission'
            time=30
            disease_label='N18'
        elif config.Phenotype == 'COPD':
            label='Readmission'
            time=30
            disease_label='J44'
        
    data_icu= config.Include_ICU
    data_mort=label=="Mortality"
    data_admn=label=='Readmission'
    data_los=label=='Length of Stay'
            

    if (config.Disease_Filter=="HF"):
        print("HF selected")
        icd_code='I50'
    elif (config.Disease_Filter=="CKD"):
        icd_code='N18'
    elif (config.Disease_Filter=="COPD"):
        icd_code='J44'
    elif (config.Disease_Filter=="CAD"):
        icd_code='I25'
    else:
        icd_code='No Disease Filter'

    if config.Version=='2.2':
        version_path=config.RawDataPath
        icu_text = "ICU" if data_icu else "Non_ICU"
        cohort_output = extraction.extract_data(icu_text,label,time,icd_code, root_dir, version_path ,disease_label)
    else:
        raise ValueError("Invalid version selected. This pipline only supports 2.2 for now")
    
    return cohort_output    

def FeatureSelection(config: PrepocessConfig , cohort_output , checkpoint:CheckpointManager):
    
    if checkpoint.is_done( "FeatureSelection"):
        print("\n")
        print("Feature selection already completed. Skipping.")
        print("\n")
        return 
    print("\n")
    print("Feature selection step")
    print("\n")
    version_path=config.RawDataPath
    diag_flag=config.Include_Diagnosis
    out_flag=config.Include_output_event
    chart_flag=config.Include_chart_event
    proc_flag=config.Include_Procedures
    med_flag=config.Include_Medications
    if config.Include_ICU:
        feature_selection.feature_icu(cohort_output, version_path,diag_flag,out_flag,chart_flag,proc_flag,med_flag)
        checkpoint.mark_done( "FeatureSelection")
    else:
        feature_selection.feature_nonicu(cohort_output, version_path,diag_flag,chart_flag,proc_flag,med_flag)
        checkpoint.mark_done( "FeatureSelection")

def FeatureProcessing(config: PrepocessConfig , cohort_output , checkpoint:CheckpointManager):
    if checkpoint.is_done( "FeatureProcessing"):
        print("\n")
        print("Feature processing already completed. Skipping.")
        print("\n")
        return 
    print("\n")
    print("Feature processing step")
    print("\n")
    group_diag=False
    group_med=False
    group_proc=False
    if config.Include_ICU:
        if config.Include_Diagnosis:
            group_diag='Convert ICD-9 to ICD-10 and group ICD-10 codes'
        clean_chart=config.Outliers_management=='remove'
        impute_outlier_chart= config.Outliers_management == 'remove'
        thresh_right=config.Outliers_threshold_right 
        thresh_left=config.Outliers_threshold_left
        diag_flag=config.Include_Diagnosis 
        chart_flag=config.Include_chart_event 

        feature_selection.preprocess_features_icu(cohort_output, diag_flag, group_diag,chart_flag,clean_chart,impute_outlier_chart,thresh_right,thresh_left)
        checkpoint.mark_done( "FeatureProcessing")
    else:
        group_diag=False
        group_med=False
        group_proc=False
        diag_flag=config.Include_Diagnosis 
        chart_flag=config.Include_chart_event 
        proc_flag=config.Include_Procedures
        clean_chart=config.Outliers_management=='remove'
        impute_outlier_chart= config.Outliers_management == 'remove'
        thresh_right=config.Outliers_threshold_right 
        thresh_left=config.Outliers_threshold_left
        med_flag=config.Include_Medications 
        if config.Include_Diagnosis:
            group_diag='Convert ICD-9 to ICD-10 and group ICD-10 codes'
        if config.Include_Medications:
            group_med="Yes"
        if config.Include_Procedures:
            group_proc="ICD-10"
        feature_selection.preprocess_features_hosp(cohort_output, diag_flag,proc_flag,med_flag,chart_flag,group_diag,group_med,group_proc,clean_chart,impute_outlier_chart,thresh_right,thresh_left)
        checkpoint.mark_done( "FeatureProcessing")

def Generation(config: PrepocessConfig , cohort_output , checkpoint:CheckpointManager):

    if checkpoint.is_done("Generation"):
        print("\n")
        print("Generation already completed. Skipping.")
        print("\n")
        return 
    print("\n")
    print("Generation step")
    print("\n")

    diag_flag=config.Include_Diagnosis 
    out_flag=config.Include_output_event 
    chart_flag=config.Include_chart_event 
    proc_flag=config.Include_Procedures 
    med_flag=config.Include_Medications 
    impute=config.Missing_values_management 
    include=config.Time_window_size
    bucket=config.Time_window_bucket_size 
    predW=config.Mortality_prediction_horizon 
    data_mort=config.Task=="Mortality"
    data_admn= config.Task=='Readmission' or config.Task=='Phenotype'
    data_los=config.Task=='Length of Stay'

    if config.Include_ICU:
        gen=data_generation_icu.Generator(cohort_output,data_mort,data_admn,data_los,diag_flag,proc_flag,out_flag,chart_flag,med_flag,impute,include,bucket,predW)
        checkpoint.mark_done("Generation")
    else:
        gen=data_generation.Generator(cohort_output,data_mort,data_admn,data_los,diag_flag,chart_flag,proc_flag,med_flag,impute,include,bucket,predW)
        checkpoint.mark_done("Generation")

def Create_final_csv(config: PrepocessConfig):
    fmt = config.Output_format
    print("\n")
    print("Create final csv file")
    print("\n")
    use_ICU = "ICU" if config.Include_ICU else "Non_ICU"
    label = config.Task
    disease_label = config.Disease_Filter
    time = config.Time_window_size
    bucket = config.Time_window_bucket_size
    oversampling = config.Oversampling 
    concat = config.Concatenate
    build_dataset.build_dataset(use_ICU , label , disease_label , bucket , time , oversampling , concat, fmt )


# -----------------------
# MAIN
# -----------------------

if __name__ == "__main__":

    root_dir = os.getcwd() + "/"

    if len(sys.argv) == 2:
        config_path = sys.argv[1]
        config = PrepocessConfig()
        config.load_from_json(config_path)
    elif len(sys.argv) == 1:
        config = base_config
    else:
        raise ValueError("Provide 0 or 1 argument for config file.")

    checkpoint = CheckpointManager(config)

    print("Configuration loaded:")
    print(config)

    cohort_output = Extraction(config, root_dir , checkpoint)
    
    FeatureSelection(config, cohort_output , checkpoint)
    FeatureProcessing(config, cohort_output , checkpoint)
    Generation(config, cohort_output , checkpoint)

    Create_final_csv(config)
