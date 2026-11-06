# LOAD PACKAGES
# 0.1 - DATA PREP AND MATH LIBRARIES
import pandas       as pd
import numpy        as np

# PLOTTING
import seaborn      as sns
from   matplotlib                 import pyplot as plt
#%matplotlib inline 


# DECISION TREES
from   sklearn.feature_extraction import DictVectorizer        # CONVERT DATAFRAME TO MATRIX AND APPLY ONE-HOT ENCODING
from   sklearn.model_selection    import train_test_split      # SPLIT THE DATASET
from   sklearn.metrics            import (
                                    roc_auc_score,              # CLASSIFICATION ACCURACY METRIC
                                    mean_squared_error         # REGRESSION ACCURACY METRIC
                                )
from   sklearn.tree               import (
                                    DecisionTreeClassifier,     # CLASSIFICATION PREDICTIONS 
                                    DecisionTreeRegressor,      # REGRESSION PREDICTIONS
                                    export_text                # 
                                )
from   sklearn.ensemble           import RandomForestRegressor # ENEMBLE LEANING

###########################################################################################
## GET DATA
###########################################################################################
data_path  = "C:/Users/duffl/anaconda3/_CT161_GitHub_Repos/ml-zoomcamp-homework/07_midterm_project_plants/data/Australian_plant_translocation_database_PUBLIC_2018(ALL_TRANSLOCATIONS).csv"
df         = pd.read_csv(data_path)


###########################################################################################
# BEGIN PREPARING DATA
###########################################################################################
# STANDARDIZE ALL COLUMN NAMES
df.columns = (
                df.columns.str.lower()
                  .str.replace('%',           '_pct_', regex=False)  # REPLACE WHITESPACES AND HYPHENS WITH UNDERSCORES
                  .str.replace(r'[ -()//]', '_',     regex=True)  # REPLACE WHITESPACES AND HYPHENS WITH UNDERSCORES
                  .str.replace(r"""['"]""",   '',      regex=True)  # REMOVE SINGLE QUOTES AND DOUBLE QUOTES
                  .str.replace(r'[,;.-?]',     '',      regex=True)  # REMOVE COMMAS, PERIODS, 
             )

# CREATE A LIST OF CATEGORICAL COLUMNS
categorical_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()  
#categorical_columns

# NORMALIZE CATEGORICAL STRING VALUES - TRIM WHITESPACE, REMOVE QUOTES, LOWER CASE
df[categorical_columns]   = df[categorical_columns].apply(
                                lambda col: col.astype('string')
                                               .str.replace(r"""['"]""", '', regex=True)
                                               .str.strip()
                                               .str.lower()
                            )
## PART 1 - BEGIN PREPARING DATA AND FEATURE ENGINEERING
# - DROP FEATURES UNLIKELY TO BE MEAINGFUL
# - STANDARDIZE FEATURE VALUES
# - SIMPLIFY FEATURE NAMES

exclude_status_summary = r'incomplete|unable_to_monitor|finalised '

df                     = df.loc[
                          ~df['status_summary'].str.contains(
                            exclude_status_summary, 
                            na=False
                          )
                        ].copy()

###########################################################################################
# DROP FEATURES WHOSE CONTENT IS UNIQUE ACROSS ALL RECORDS OR OF LOW FREQUENCY 
###########################################################################################
exclude_features_list  =    [
                                'status_summary',
                                'treatment_description', 
                                'species', 
                                'family',
                                'bioregion',
                                'conservation_status__state_',
                                'latitude__decimal_degrees_',
                                'longitude__decimal_degrees_',
                                'habitat_description',
                                'start_month',
                                'start_year',
                                'number_of_seeds__direct_seeding_only_',
                                'number_of_individuals_planted_or_germinated_if_direct_seeding__total_',
                                'number_planted_notes',
                                'watering_in_only', 
                                'watering___months_', 
                                'watering___months_or_in_dry_season_', 
                                'last_monitoring_date',
                                'number_of_plants_surviving',
                                'number_of_subsequent_plantings',
                                'no_second_generation_plants',
                                'short-term_success___pct__of_plants_which_survived_first_year_', 
                                'medium-term_success__population_persisting_at_secure_site_',
                                'long-term_success__self-sustaining_population_established_with_dynamics_comparable_to_natural_populations_',
                                'reasons_for_failure_or_sub-optimal_performance_or_success__if_known_',
                                'failure_factors_summary',
                                'success_factors_summary',
                                'notes',
                                'project_managed_by',
                                'contact_person_s', 
                                'references' 
                            ]

missing = [col for col in exclude_features_list if col not in df.columns]
print("Not found:\r\n", missing)

df = df.drop(columns=exclude_features_list, errors='ignore')


###########################################################################################
# EXTRACT PERCENTAGE OF SURVIVING TRANSLOCATED PLANTS
###########################################################################################
source = df['plants_surviving___pct__'].astype('string').str.strip()

replacements = {
    
    '<NA>': '0.00',
    '??': '0.00',
    '#DIV/0!': '0.00',
    '#REF!': '0.00',
    '#VALUE!' : '0.00',
    'Apr-00': '0.00',
    'n/a (annual)': '0.00',
    'n/a for annual species': '0.00',
    'n/a (annual plant)' : '0.00',
    'n/a (direct seeding)': '0.00',
    'n/a (short-lived)': '0.00',
    'Not monitored yet' : '0.00',
    'Unknown (probably 0)' : '0.00',
    '0 (but substantial recruitment)': '0.00',
    '<5'   : '5.0',
    '<10'  : '10.00',
    '>10'  : '10.00',
    '>10%' : '10.00',
    '<20'  : '20.00',
    'c.20' : '20.00',
    '>25'  : '25.00',
    '>25%' : '25.00',
    'c.25' : '25.00',
    '20-40%': '30.00',
    '20-40': '30.00',
    'c.25' : '25.00',
    '~30'  : '30.00',
    '>30'  : '30.00',
    '>33'  : '33.00',
    '~40'  : '40.00',
    '>40'  : '40.00',
    '>50%' : '50.00',
    '<50%' : '50.00',
    '>50%' : '50.00',
    '>50'  : '50.00',
    '<50?' : '50.00',
    '~50'  : '50.00',
    '<50%?': '50.00',
    '<50?' : '50.00',
    'Unknown, but probably >75%': '75.00',
    '80 (across all sites and treatments)': '80.00',
    '80% (across all sites and treatments)': '80.00',
    '>90'  : '90.00',
    '>95'  : '95.00',
    '>95%' : '95.00',
    '>100 (with recruitment)' : '0.00',
    'Good' : '0.00',
    'unknown' : '0.00',
    'Unknown': '0.00',
    'Unknown (number germinating not documented)' : '0.00',
    'Unknown how many were planted in total': '0.00',
    'Unknown (number of germinants not counted)': '0.00',
    'Too soon': '0.00',
    'Too early': '0.00',
    'Plants in the ground for < 12 months': '0.00',
    '<12 months since planting': '0.00',
    '< 12 mnths since translocation' : '0.00',
    '<12 months elapsed': '0.00',
    '%' : '',
}

cleaned = source.replace(replacements)
cleaned = cleaned.str.replace('%', '', regex=False).str.strip()
cleaned = cleaned.fillna('0.00')



parts = cleaned.str.split(':', n=1)

df['col_plants_surviving_pct'] = parts.str[0].astype('string').str.strip()
# df['col_plants_surviving_ct'] = parts.str[1].astype('string').str.strip()



###########################################################################################
# EXTRACT PERCENTAGE OF SURVIVING TRANSLOCATED PLANTS
###########################################################################################
source = df['plants_surviving___pct__'].astype('string').str.strip()

replacements = {
    
    '<NA>': '0.00',
    '??': '0.00',
    '#DIV/0!': '0.00',
    '#REF!': '0.00',
    '#VALUE!' : '0.00',
    'Apr-00': '0.00',
    'n/a (annual)': '0.00',
    'n/a for annual species': '0.00',
    'n/a (annual plant)' : '0.00',
    'n/a (direct seeding)': '0.00',
    'n/a (short-lived)': '0.00',
    'Not monitored yet' : '0.00',
    'Unknown (probably 0)' : '0.00',
    '0 (but substantial recruitment)': '0.00',
    '<5'   : '5.0',
    '<10'  : '10.00',
    '>10'  : '10.00',
    '>10%' : '10.00',
    '<20'  : '20.00',
    'c.20' : '20.00',
    '>25'  : '25.00',
    '>25%' : '25.00',
    'c.25' : '25.00',
    '20-40%': '30.00',
    '20-40': '30.00',
    'c.25' : '25.00',
    '~30'  : '30.00',
    '>30'  : '30.00',
    '>33'  : '33.00',
    '~40'  : '40.00',
    '>40'  : '40.00',
    '>50%' : '50.00',
    '<50%' : '50.00',
    '>50%' : '50.00',
    '>50'  : '50.00',
    '<50?' : '50.00',
    '~50'  : '50.00',
    '<50%?': '50.00',
    '<50?' : '50.00',
    'Unknown, but probably >75%': '75.00',
    '80 (across all sites and treatments)': '80.00',
    '80% (across all sites and treatments)': '80.00',
    '>90'  : '90.00',
    '>95'  : '95.00',
    '>95%' : '95.00',
    '>100 (with recruitment)' : '0.00',
    'Good' : '0.00',
    'unknown' : '0.00',
    'Unknown': '0.00',
    'Unknown (number germinating not documented)' : '0.00',
    'Unknown how many were planted in total': '0.00',
    'Unknown (number of germinants not counted)': '0.00',
    'Too soon': '0.00',
    'Too early': '0.00',
    'Plants in the ground for < 12 months': '0.00',
    '<12 months since planting': '0.00',
    '< 12 mnths since translocation' : '0.00',
    '<12 months elapsed': '0.00',
    '%' : '',
}

cleaned = source.replace(replacements)
cleaned = cleaned.str.replace('%', '', regex=False).str.strip()
cleaned = cleaned.fillna('0.00')



parts = cleaned.str.split(':', n=1)

df['col_plants_surviving_pct'] = parts.str[0].astype('string').str.strip()
# df['col_plants_surviving_ct'] = parts.str[1].astype('string').str.strip()
