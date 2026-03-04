from utils.config import PrepocessConfig
import os 
from steps import extraction 
from steps import feature_selection
from steps import data_generation
from steps import data_generation_icu
import sys


base_config = PrepocessConfig(
    Version="2.2",
    RawDataPath="mimic-IV/2.2/",
    PreprocessedDataPath="data/output/",
    Task="Length of Stay",
    LengthOfStay_greater_or_equal_threshold=3, 
    Include_ICU=False,
    Include_Diagnosis=True,
    Include_Procedures=True,
    Include_Medications=True,
    Include_chart_event=True,
    Include_output_event=True,
    Disease_Filter = "None",
    Outliers_management = "remove",
    Outliers_threshold_right = 98.0,
    Outliers_threshold_left = 0,
    Time_window_size = 24,
    Time_window_bucket_size = 1,
    Missing_values_management = "mean",

)


def Extraction(config: PrepocessConfig , root_dir:str):
    print("Extraction step")
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
            

    if (config.Disease_Filter=="Heart Failure"):
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
    # elif config.Version=='3.1':
    #     version_path=config.RawDataPath
    #     cohort_output = day_intervals_cohort_v3.extract_data(radio_input1.value,label,time,icd_code, root_dir, version_path ,disease_label)
    else:
        raise ValueError("Invalid version selected. Please select either 2.2 or 3.1.")

    return cohort_output
        
def FeatureSelection(config: PrepocessConfig , root_dir:str , cohort_output):

    version_path=config.RawDataPath
    diag_flag=config.Include_Diagnosis
    out_flag=config.Include_output_event
    chart_flag=config.Include_chart_event
    proc_flag=config.Include_Procedures
    med_flag=config.Include_Medications
    if config.Include_ICU:
        feature_selection.feature_icu(cohort_output, version_path,diag_flag,out_flag,chart_flag,proc_flag,med_flag)
        
    else:
        feature_selection.feature_nonicu(cohort_output, version_path,diag_flag,chart_flag,proc_flag,med_flag)

def Generation(config: PrepocessConfig , root_dir:str , cohort_output):
    version_path=config.RawDataPath 
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
        #gen=data_generation_icu.Generator(cohort_output,data_mort,diag_flag,False,False,chart_flag,False,impute,include,bucket,predW)
        #if chart_flag:
        #    gen=data_generation_icu.Generator(cohort_output,data_mort,False,False,False,chart_flag,False,impute,include,bucket,predW)
    else:
        gen=data_generation.Generator(cohort_output,data_mort,data_admn,data_los,diag_flag,chart_flag,proc_flag,med_flag,impute,include,bucket,predW)

if __name__ == "__main__":

    root_dir = os.getcwd() + "/"
    
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
        config = PrepocessConfig()
        config.load_from_json(config_path)
    elif len(sys.argv) == 1:
        config = base_config
    else :
        raise ValueError("Invalid number of arguments. Please provide either 0 or 1 argument for the config file path.")
    print("Configuration loaded:")
    print(config)
    cohort_output = Extraction(config, root_dir)
    FeatureSelection(config, root_dir, cohort_output)

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
    
    Generation(config, root_dir, cohort_output)

