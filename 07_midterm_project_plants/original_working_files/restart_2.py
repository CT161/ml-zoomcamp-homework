#!/usr/bin/env python
# coding: utf-8

# # 1.0 - LOAD PACKAGES

# ## 1.1 - LOAD PACKAGES AND GET DATA

# In[1]:


# 0.1 - DATA PREP AND MATH LIBRARIES
import pandas       as pd
import numpy        as np
import re

# PLOTTING
import seaborn      as sns
from   matplotlib                 import pyplot as plt
#%matplotlib inline 


# LOGISTIC REGRESSION
from sklearn.linear_model         import LogisticRegression

# DECISION TREES
from   sklearn.feature_extraction import DictVectorizer        # CONVERT DATAFRAME TO MATRIX AND APPLY ONE-HOT ENCODING
from   sklearn.model_selection    import train_test_split      # SPLIT THE DATASET
from   sklearn.metrics            import (
                                    roc_auc_score,             # CLASSIFICATION ACCURACY METRIC
                                    mean_squared_error         # REGRESSION ACCURACY METRIC
                                )
from   sklearn.tree               import (
                                    DecisionTreeClassifier,    # CLASSIFICATION PREDICTIONS 
                                    DecisionTreeRegressor,     # REGRESSION PREDICTIONS
                                    export_text                # 
                                )
from   sklearn.ensemble           import RandomForestRegressor # ENEMBLE LEANING


# # 2.0 - HELPER FUNCTIONS
column_types_df = df.dtypes.reset_index()
column_types_df.columns = ["column", "dtype"]

print(column_types_df)
column_types_df = df.dtypes.reset_index()
column_types_df.columns = ["column", "dtype"]

print(column_types_df)
# ## 2.1 - DATA PREPARATION

# In[2]:


# CLEAN STRINGS 
def fn_clean_text(value):
    # ============================================================
    # DESCRIPTION: RECEIVE A STRING AND REMOVE UNWANTED CHARACTERS 
    # ============================================================
    # PACKAGE REQUIREMENTS:
    #  pandas
    # ============================================================
    # PARAMETER    DESCRIPTION
    # ------------------------------------------------------------
    #  value        STRING TO CLEAN 
    # ============================================================
    # ============================================================
    # SYNTAX 
    # ------------------------------------------------------------
    #   fn_clean_text(column)
    # ============================================================
    if not isinstance(value, str):
        return value

    value = value.lower()
    value = value.replace('%', '_pct_')
    value = re.sub(r'[\s()/\-]+', '_', value)  # spaces, slashes, parentheses, hyphens
    value = re.sub(r"""['"]""", '', value)     # remove quotes
    value = re.sub(r'[,;?]', '', value)        # remove punctuation
    return value

def drop_column(in_dataframe, in_columns):
    # ============================================================
    # DESCRIPTION: RETURN A COPY OF THE DATAFRAME WITH ONE OR MORE  
    #              FEATURES (SOLUMNS) REMOVED 
    # ============================================================
    # PACKAGE REQUIREMENTS:
    #  pandas
    # ============================================================
    # PARAMETER    DESCRIPTION
    # ------------------------------------------------------------
    #  df          'pandas' DATAFRAME 
    #  columns     EITHER A FEATURE NAME OR A LIST CONTAINING  
    #                FEATURE NAMES 
    # ============================================================
    # SYNTAX 
    # ------------------------------------------------------------
    #   df = drop_column(df, "lifeform")
    #   df = drop_column(df, ["lifeform", "second_gen"])
    # ============================================================
    if not isinstance(in_dataframe, pd.DataFrame):
        raise TypeError("'in_dataframe' must be a pandas DataFrame")

    if isinstance(in_columns, str):
        in_columns = [in_columns]
    elif not isinstance(in_columns, (list, tuple)):
        raise TypeError("'in_columns' must be a column name or a list/tuple of names")

    missing = [
        column for column in in_columns
        if column not in in_dataframe.columns
    ]
    if missing:
        raise KeyError(f"Feature(s) (Column(s)) not found:\r\n{missing}")

    return in_dataframe.drop(columns=list(in_columns))


# ## 2.2 - DATA REVIEW

# In[3]:


def fn_view_col_value_status_table(in_data, in_df=None):
    # ============================================================
    # DESCRIPTION: CREATE A TABLE SHOWING HOW MANY ROWS IN A 
    #              FEATURE ARE MISSING VALUES
    # ============================================================
    # PACKAGE REQUIREMENTS:
    #  pandas
    # ============================================================
    # PARAMETER    DESCRIPTION
    # ------------------------------------------------------------
    #  data         'pandas' DATAFRAME OR FEATURE NAME
    # ============================================================
    # SYNTAX 
    # ------------------------------------------------------------
    #   fn_view_col_value_status_table(df)
    #   fn_view_col_value_status_table(df["second_gen"])
    #   fn_view_col_value_status_table("second_gen", in_df=df)
    # ============================================================
    if isinstance(in_data, str):
        if in_df is None:
            raise ValueError("Pass 'in_df' when in_data is a column name")
        if in_data not in in_df.columns:
            raise KeyError(f"Column not found: {in_data}")
        columns = [(in_data, in_df[in_data])]

    elif isinstance(in_data, pd.Series):
        columns = [(in_data.name or "value", in_data)]

    elif isinstance(in_data, pd.DataFrame):
        columns = list(in_data.items())

    else:
        raise TypeError(
            "'in_data' must be a feature (column) name, pandas Series, or pandas DataFrame"
        )

    print("=" * 80)
    print("NON_MISSING\tMISSING\t\t%_MISSING\tCOLUMN")
    print("=" * 80)

    for column_name, series in columns:
        missing = int(series.isna().sum())
        non_missing = int(series.notna().sum())
        percent_missing = missing / len(series) * 100 if len(series) else 0

        print(
            f"{non_missing}\t\t{missing}\t\t"
            f"{percent_missing:.2f}%\t\t{column_name}"
        )

    print("=" * 80)


def fn_view_column_value_counts(in_data, in_df=None):
    # ============================================================
    # DESCRIPTION: PRINT VALUE COUNTS AND PERCENTAGES FOR EACH 
    #              VALUE IN A 'pandas' SERIES OR DATAFRAME
    # ============================================================
    # PACKAGE REQUIREMENTS:
    #  pandas      import pandas as pd
    # ============================================================
    # PARAMETER    DESCRIPTION
    # ------------------------------------------------------------
    #  in_data     'pandas' DATAFRAME OR FEATURE NAME
    #  in_df       'pandas' DATAFRAME
    # ============================================================
    # SYNTAX 
    # ------------------------------------------------------------
    #   fn_view_column_value_counts("second_gen", in_df=df)
    #   fn_view_column_value_counts(df["second_gen"])
    #   fn_view_column_value_counts(df)
    # ============================================================
    if isinstance(in_data, str):
        if in_df is None:
            raise ValueError("Pass 'in_df' when 'in_data' is a Feature (column) name")
        if in_data not in in_df.columns:
            raise KeyError(f"Column not found: {in_data}")
        series_list = [(in_data, in_df[in_data])]

    elif isinstance(in_data, pd.Series):
        series_list = [(in_data.name or "value", in_data)]

    elif isinstance(in_data, pd.DataFrame):
        series_list = list(in_data.items())

    else:
        raise TypeError(
            "'in_data' must be a Feature (column) name, pandas Series, or pandas DataFrame"
        )

    for column_name, series in series_list:
        counts = series.value_counts(dropna=False)
        total = len(series)

        print(f"\n{column_name}")
        print("COUNT\t\tFREQUENCY\tVALUE")
        print("=" * 80)

        for value, count in counts.items():
            percent = count / total * 100 if total else 0
            print(f"{count}\t\t{percent:.2f}%\t\t{value}")


def fn_view_feature_data_quality(in_data, in_df=None):
    # ============================================================
    # DESCRIPTION: SHOW MISSING VALUE STATUS AND VALUE COUNTS FOR 
    #                A 'pandas' FEATURE (COLUMN), SERIES, OR 
    #                DATAFRAME 
    #              COMBINES THE OUTPUT FROMFUNCTIONS: 
    #              - fn_view_col_value_status_table(in_df, in_columns)
    #              - fn_view_column_value_counts(in_data, in_df=None)
    # ============================================================
    # PACKAGE REQUIREMENTS:
    #  pandas
    # ============================================================
    # PARAMETER    DESCRIPTION
    # ------------------------------------------------------------
    #  data         'pandas' DATAFRAME OR FEATURE NAME
    # ============================================================
    # ============================================================
    # SYNTAX 
    # ------------------------------------------------------------
    #   fn_view_feature_data_quality("second_gen", in_df=df)
    #   fn_view_feature_data_quality(df["second_gen"])
    #   fn_view_feature_data_quality(df)
    # ============================================================

    if isinstance(in_data, str):
        if in_df is None:
            raise ValueError("Pass 'in_df' when 'in_data' is a feature (column name)")

        fn_view_col_value_status_table(in_data, in_df=in_df)
        fn_view_column_value_counts(in_data, in_df=in_df)

    elif isinstance(in_data, (pd.Series, pd.DataFrame)):
        fn_view_col_value_status_table(in_data)
        fn_view_column_value_counts(in_data)

    else:
        raise TypeError(
            "'in_data' must be a feature (column) name, pandas Series, or pandas DataFrame"
        )


# ## 2.2 - EVALUATION

# ### 2.2.1 - REGRESSION EVALUATION

# In[4]:


# RMSE IS IN 'sklearn'
#   from sklearn.metrics import root_mean_squared_errorp    # IMPORT FUNCTION
#
#   rmse = root_mean_squared_error(y_val, predictions)      # CALCULATE ACCURACY
#   print("Validation RMSE:", rmse)                         # PRINT ACCURACY


# ### 2.2.2 - CLASSIFICATION EVALUATION

# In[5]:


# ROC-AUC
#   from sklearn.metrics import                             # IMPORT FUNCTION
#
#   y_prob = model.predict_proba(X_val)[:, 1]               # MAKE PREDICTION
#   auc = roc_auc_score(y_val, y_prob)                      # CALCULATE 'auc'
#   print("ROC-AUC:", auc)                                  # PRINT ACCURACY


# # 3.0 - PREPARE DATA

# ## 3.1 - GET AND PREVIEW DATA

# In[6]:


###########################################################################################
#   3.1.1 - GET DATA
###########################################################################################
data_path  = "C:/Users/duffl/anaconda3/_CT161_GitHub_Repos/ml-zoomcamp-homework/07_midterm_project_plants/data/Australian_plant_translocation_database_PUBLIC_2018(ALL_TRANSLOCATIONS).csv"
df         = pd.read_csv(data_path)

###########################################################################################
#   3.1.2 - CLEAN COLUMN NAMES AND RECORD VALUES
###########################################################################################
df.columns             = [fn_clean_text(column) for column in df.columns]
df                     = df.map(fn_clean_text)

# MANUALLY FIX COLUMN WITH PERIOD IN COLUMN NAME
df['habitat_condition'] = df['habitat_condition_e.g._highly_disturbed_small_remnant_large_remnant_']
df.drop(columns=['habitat_condition_e.g._highly_disturbed_small_remnant_large_remnant_'], inplace=True)

###########################################################################################
#   3.1.3 - PREVIEW NEW COLUMN NAMES 
###########################################################################################
fn_view_col_value_status_table(df)
#   fn_view_col_value_status_table(df)
    #   fn_view_col_value_status_table(df["second_gen"])
    #   fn_view_col_value_status_table("second_gen", in_df=df)

#   fn_view_column_value_counts("second_gen", in_df=df)
    #   fn_view_column_value_counts(df["second_gen"])
    #   fn_view_column_value_counts(df)

#   fn_view_feature_data_quality("second_gen", in_df=df)
    #   fn_view_feature_data_quality(df["second_gen"])
    #   fn_view_feature_data_quality(df)

###########################################################################################
#   3.1.4 - PREVIEW DATA VALUES 
###########################################################################################
df.head(20)


# ## 3.2 - FIRST PASS CLEANING

# ### 3.2.1 - DROP RECORDS MARKED AS UNSUITABLE FOR ANALYSIS

# In[7]:


# datapoint = 
df.loc[10].to_dict()


# In[8]:


###########################################################################################
#   3.2.1 - DROP RECORDS DATASET CREATORS HAVE MARKED AS UNSUITABLE FOR ANALYSIS
###########################################################################################
exclude_status_summary = r'incomplete|unable_to_monitor|finalised '

df                     = df.loc[
                          ~df['status_summary'].str.contains(
                            exclude_status_summary, 
                            na=False
                          )
                        ].copy()
print('records dropped')


# ### 3.2.2 - DROP FEATURES UNSUITABLE FOR ANALYSIS

# In[9]:


###########################################################################################
#   1. - CRITERIA TO DROP FEATURES  
#        - CONTENT IS UNIQUE ACROSS ALL RECORDS 
#        - FEW RECORDS HAVE VALUES
#        - TOO MANY CATEGORIES AND CATEGORY LABELS CANNOT BE COLLAPSED INTO A NEW LABEL 
###########################################################################################
exclude_features_list  =    [
                                'status_summary',
                                'treatment_number',
                                'treatment_description', 
                                'species', 
                                'family',
                                'bioregion',
                                'conservation_status_state_',
                                'latitude_decimal_degrees_',
                                'longitude_decimal_degrees_',
                                'habitat_description',
                                'start_month',
                                'start_year',
                                'translocation_number',
                                'number_of_seeds_direct_seeding_only_',
                                'number_of_individuals_planted_or_germinated_if_direct_seeding_total_',
                                'number_planted_notes',
                                'watering_in_only', 
                                'watering_<6_months_', 
                                'watering_>6_months_or_in_dry_season_', 
                                'last_monitoring_date',
                                'number_of_subsequent_plantings',
                                'no_second_generation_plants',
                                'short_term_success__pct__of_plants_which_survived_first_year_', 
                                'medium_term_success_population_persisting_at_secure_site_',
                                'long_term_success_self_sustaining_population_established_with_dynamics_comparable_to_natural_populations_',
                                'reasons_for_failure_or_sub_optimal_performance_or_success_if_known_',
                                'failure_factors_summary',
                                'success_factors_summary',
                                'notes',
                                'project_managed_by',
                                'contact_person_s', 
                                'references'
                            ]

###########################################################################################
#   2. - REPORT ANY FEATURES THAT CAN NOT BE FOUND
###########################################################################################
missing                = [col for col in exclude_features_list if col not in df.columns]
print("Features to remove that were not found:\r\n", missing)
print()

###########################################################################################
#   3. - DROP FEATURES IN LIST
###########################################################################################
df                     = df.drop(columns=exclude_features_list, errors='ignore')

###########################################################################################
#   4. - REVIEW REMAININNG FEATURES
###########################################################################################
print("=" * 80)
print("===== REMAINING FEATURES =======================================================")
fn_view_col_value_status_table(df)
print("=" * 80)


# ## 3.3 - DATA CLEANING
# PREVIEW DATA CONTENT 
df['plants_surviving__pct__'].head(50)
# ### 3.3.01 - FEATURE ENRICHMENT: 'treatment_number' ==> 'treatment'
# PREVIEW RAW DATA VALUES 
source_column          = 'treatment_number'
destination_column     = 'treatment'

fn_view_feature_data_quality(source_column)

# CLEAN DATA  
dict_map               = {
                            '0' : 'no',
                            '1' : 'yes',
                            'u' : 'unk'
                         }
df[destination_column]     = df[source_column].astype('string').str.strip().replace(dict_map).fillna('missing')

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(destination_column)

# ### 3.3.02 - FEATURE ENRICHMENT: 'lifeform'
df['lifeform'].head()
# In[10]:


# PREVIEW RAW DATA VALUES 
source_column          = 'lifeform'
destination_column     = 'lifeform'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
dict_map               = {
                            'shrub': 'shrub',
                            'tree': 'tree',
                            'perennial_forb': 'perennial_forb',
                            'ground_orchid': 'ground_orchid',
                            'perennial_vine': 'perennial_vine',
                            'annual_forb': 'annual_forb',
                            'perennial_grass': 'perennial_grass',
                            'perennial_lily': 'perennial_lily',
                            'annual_grass': 'annual_grass',
                            'cycad': 'cycad',
                            'perennial_forb_fern_': 'perennial_forb_fern',
                            'perennial_sedge': 'perennial_sedge',
                            'perennnial_forb': 'perennnial_forb',
                            'perennial_sedge': 'perennial_sedge',
                            'perennnial_forb': 'perennnial_forb',
                            'orchid_epiphytic_': 'orchid_epiphytic',
                            'palm': 'palm',
                            'grass_tree': 'grass_tree',
                            'tree_mallee_': 'tree_mallee',
                            'fern': 'fern',
                            'epiphyte': 'epiphyte',
                            'annual_sedge': 'annual_sedge',
                            'tree_': 'tree',
                            'perennial_aquatic_forb': 'perennial_aquatic_forb',
                            'shrub_' : 'shrub'
                         }
df[destination_column]     = df[source_column].astype('string').str.strip().replace(dict_map).fillna('missing')

if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.03 - FEATURE ENRICHMENT: 'state'

# In[11]:


# PREVIEW RAW DATA VALUES 
source_column           = 'state' 
destination_column      = 'state' 

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
# df[destination_column]  = df[source_column].copy()
df[destination_column]= df[source_column].str.strip().replace('qld_', 'qld')

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.04 - FEATURE ENRICHMENT: 'conservation_status_epbc_'

# In[12]:


# PREVIEW RAW DATA VALUES 
source_column           = 'conservation_status_epbc_' 
destination_column      = 'conservation_status' 

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA                 
dict_map               = {
                            'en' : 'endangered',
                            'vu' : 'vulnerable',
                            'cr' : 'critically_endangered',
                            'vu' : 'vulnerable',
                            'vu_' : 'vulnerable',
                            'rotap_2rc_' : 'rare',
                            'not_listed_rotap_3rca_' : 'rare',
                            'not_listed_rotap_' : 'not_listed',
                            'ev' : 'extinct_in_wild',
                         }
df[destination_column] = df[source_column].astype('string').str.strip().replace(dict_map).fillna('missing') 

# DROP ORIGINAL COLUMN 
# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.05 - FEATURE ENRICHMENT: 'broad_habitat_group'

# In[13]:


# PREVIEW RAW DATA VALUES 
source_column          = 'broad_habitat_group'
destination_column     = 'habitat'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA 
dict_map               = {
                            'rainforest': 'rainforest',
                            'southern_shrublands_heathlands_woodlands': 'southern_shrublands_heathlands_woodlands',
                            'southern_shrublands_heathlands_woodlands_': 'southern_shrublands_heathlands_woodlands',
                            'grassland': 'grassland',
                            'grassy_woodland': 'grassy_woodland',
                            'wetland': 'wetland',
                            'dry_sclerophyll': 'dry_sclerophyll',
                            'mallee': 'mallee',
                            'grassland_with_chenopods': 'grassland_with_chenopods',
                            'coastal_heath': 'coastal_heath',
                            'wet_sclerophyll': 'wet_sclerophyll',
                            'banded_ironstone': 'banded_ironstone',
                            'grassland_sub_tropical_': 'grassland_sub_tropical',
                            'montane': 'montane',
                            'other': 'other',
                            'coastal_shrubland': 'coastal_shrubland',
                            'semi_arid_woodland': 'semi_arid_woodland',
                            'wetland_floodplain_': 'wetland_floodplain',
                            'coastal_headland': 'coastal_headland',
                            'coastal_forest': 'coastal_forest',
                            'coastal_dunes': 'coastal_dunes',
                            'grassland_': 'grassland',
                            'grassland_lomandra_': 'grassland_lomandra',
                            'wetland_riparian_': 'wetland_riparian',
                       }
df[destination_column] = df[source_column].astype('string').str.strip().replace(dict_map).fillna('unk')

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.06 - FEATURE ENRICHMENT: 'research_to_support_translocation_'

# In[14]:


# PREVIEW RAW DATA VALUES 
source_column          = 'research_to_support_translocation_'
destination_column     = 'research'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA 
is_empty               = df[source_column].isna() | df[source_column].astype('string').str.strip().eq('')
is_underscore          = df[source_column].astype('string').str.strip().eq('_')

df[destination_column] = 'yes'
df.loc[is_empty | is_underscore, destination_column] = 'no'

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.07 - FEATURE ENRICHMENT: 'mitigation_translocation'

# In[15]:


# PREVIEW RAW DATA VALUES 
source_column          = 'mitigation_translocation'
destination_column     = 'mitigation'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA 
df[destination_column] = df[source_column].astype('string').str[0]
dict_map               = {
                            'n' : 'no',
                            'y' : 'yes',
                            'u' : 'unk'
                          }
df[destination_column] = df[destination_column].astype('string').str.strip().replace(dict_map).fillna('missing')

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.08 - FEATURE REDUCTION: 'habitat_condition'

# In[16]:


# PREVIEW RAW DATA VALUES 
source_column           = 'habitat_condition'
destination_column      = '-N/A-'

fn_view_feature_data_quality(df[source_column])

print("\r\n\r\nTOO MANY CATEGORIES AND NOT ENOUGH RECORDS WITHN EACH CATEGORY")

# DROP FEATURE
df = drop_column(in_dataframe=df, in_columns=source_column)

print(f"\r\n--- FEATURE ['{source_column}'] DROPPED ---")


# ### 3.3.09 - FEATURE ENRICHMENT: 'translocation_type'

# In[17]:


# PREVIEW RAW DATA VALUES 
source_column          = 'translocation_type'
destination_column     = 'translocation'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA 
dict_map               = {
                            'introduction'               : 'introduction',
                            'augmentation'               : 'augmentation',
                            'augmentation_experimental_' : 'augmentation',
                            'introduction_'              : 'introduction',
                            'introduction_salvage_'      : 'introduction',
                            'introduction_experimental_' : 'introduction',
                            'introduction_seed_orchard_' : 'introduction',
                            'augmentation_salvage_'      : 'augmentation',
                            'restocking_of_original_introduction_above_' : 'reintroduction',
                            'reintroduction'             : 'reintroduction',
                            'introduction_reintroduction_to_riverland_' : 'introduction_reintroduction',
                            'restocking_of_above_'       : 'reintroduction',
                            'assisted_migration'         : 'introduction',
                            'introduction_augmentation_' : 'introduction_augmentation_',
                            'introduction_different_site_but_same_remnant_as_natural_population_' : 'introduction',
                          }
df[destination_column] = df[source_column].astype('string').str.strip().replace(dict_map).fillna('missing')

# DROP ORIGINAL COLUMN 
# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.10 - FEATURE REDUCTION: 'propagule_type_seed_seedling_cutting_tissue_culture_whole_plants_etc_'

# In[18]:


# PREVIEW RAW DATA VALUES 
source_column           = 'propagule_type_seed_seedling_cutting_tissue_culture_whole_plants_etc_'
destination_column      = '-N/A-'

fn_view_feature_data_quality(df[source_column])

print("\r\n\r\nTOO MANY CATEGORIES AND NOT ENOUGH RECORDS WITHN EACH CATEGORY")

# DROP FEATURE
df = drop_column(in_dataframe=df, in_columns=source_column)

print(f"\r\n--- FEATURE ['{source_column}'] DROPPED ---")




# ### 3.3.11 - FEATURE ENRICHMENT: 'soil_preparation_y_n_'

# In[19]:


# PREVIEW RAW DATA VALUES 
source_column           = 'soil_preparation_y_n_'
destination_column      = 'soil_prep'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA 
df[destination_column]  = df[source_column].astype('string').str[0]
dict_map                = {
                            'n' : 'no',
                            'y' : 'yes',
                            'u' : 'unk'
                          }
df[destination_column]  = df[destination_column].astype('string').str.strip().replace(dict_map).fillna('missing')

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.12 - FEATURE ENRICHMENT: 'pre_planting_burn_y_n_'

# In[20]:


# PREVIEW RAW DATA VALUES 
source_column          = 'pre_planting_burn_y_n_'
destination_column     = 'pre_planting_burn'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
df[destination_column] = df[source_column].astype('string').str[0]
dict_map                = {
                            'n' : 'no',
                            'y' : 'yes', 
                            'u' : 'unk'
                          }
df[destination_column]  = df[destination_column].astype('string').str.strip().replace(dict_map).fillna('missing')

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.13 - FEATURE ENRICHMENT: 'pre_planting_slashing_or_weeding_y_n_'

# In[21]:


# PREVIEW RAW DATA VALUES 
source_column          = 'pre_planting_slashing_or_weeding_y_n_'
destination_column     = 'pre_planting_slashing_or_weeding'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
df[destination_column] = df[source_column].astype('string').str[0]
dict_map               = {
                            'n' : 'no',
                            'y' : 'yes',
                            'u' : 'unk'
                         }
df[destination_column] = df[destination_column].astype('string').str.strip().replace(dict_map).fillna('missing')

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.14 - FEATURE ENRICHMENT: 'pre_site_preparation_all_'

# In[22]:


# PREVIEW RAW DATA VALUES 
source_column          = 'pre_site_preparation_all_'
destination_column     = 'pre_site_prep'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
dict_map               = {
                            '0.0' : 'no',
                            '1.0' : 'yes',
                            'u' : 'unk'
                          }
df[destination_column] = df[source_column].astype('string').str.strip().replace(dict_map).fillna('missing')

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.15 - FEATURE ENRICHMENT: 'herbivore_exclosure_fence_y_n_'

# In[23]:


# PREVIEW RAW DATA VALUES 
source_column          = 'herbivore_exclosure_fence_y_n_'
destination_column     = 'exclosure_fence'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
df[destination_column] = df[source_column].astype('string').str[0]
dict_map               = {
                            'n' : 'no',
                            'y' : 'yes',
                            'p' : 'yes'
                         }
df[destination_column] = df[destination_column].astype('string').str.strip().replace(dict_map).fillna('missing') 

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA  
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.16 - FEATURE ENRICHMENT: 'herbivore_exclosure_cages_y_n_'

# In[24]:


# PREVIEW RAW DATA VALUES 
source_column          = 'herbivore_exclosure_cages_y_n_'
destination_column     = 'exclosure_cage'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
df[destination_column] = df[source_column].astype('string').str[0]
dict_map               = {
                            'n' : 'no',
                            'y' : 'yes',
                            'u' : 'unk',
                         }
df[destination_column] = df[destination_column].astype('string').str.strip().replace(dict_map).fillna('unk') 

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.17 - FEATURE ENRICHMENT: 'herbivore_protection_y_n_'

# In[ ]:





# ### 3.3.17 - FEATURE ENRICHMENT: 'tree_guards_y_n_'

# In[25]:


# PREVIEW RAW DATA VALUES 
source_column          = 'tree_guards_y_n_'
destination_column     = 'tree_guard'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
df[destination_column] = df[source_column].astype('string').str[0]
dict_map               = {
                            'n' : 'no',
                            'y' : 'yes',
                            'u' : 'unk',
                         }
df[destination_column] = df[destination_column].astype('string').str.strip().replace(dict_map).fillna('missing') 

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.18 - FEATURE ENRICHMENT: 'grazing_protection_all_'

# In[26]:


# PREVIEW RAW DATA VALUES 
source_column          = 'grazing_protection_all_'
destination_column     = 'grazing_protection'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
df[destination_column] = df[source_column].astype('string').str[0]
dict_map               = {
                            '0' : 'no',
                            '1' : 'yes',
                         }
df[destination_column] = df[destination_column].astype('string').str.strip().replace(dict_map).fillna('missing') 

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.19 - FEATURE ENRICHMENT: 'mulching_y_n_'

# In[27]:


# PREVIEW RAW DATA VALUES 
source_column          = 'mulching_y_n_'
destination_column     = 'mulch'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
df[destination_column] = df[source_column].astype('string').str[0]
dict_map               = {
                            'n' : 'no',
                            'y' : 'yes',
                         }
df[destination_column] = df[destination_column].astype('string').str.strip().replace(dict_map).fillna('missing') 

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.20 - FEATURE ENRICHMENT: 'watering_all_'

# In[28]:


# PREVIEW RAW DATA VALUES 
source_column          = 'watering_all_'
destination_column     = 'watering'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
dict_map               = {
                            '0.0' : 'no',
                            '1.0' : 'yes',
                            'u'   : 'unk'
                          }
df[destination_column] = df[source_column].astype('string').str.strip().replace(dict_map).fillna('missing')

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.21 - FEATURE ENRICHMENT: 'burnt_0_unburnt_1_planned_2_wildfire_'

# In[29]:


# PREVIEW RAW DATA VALUES 
source_column          = 'burnt_0_unburnt_1_planned_2_wildfire_'
destination_column     = 'burnt'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
dict_map               = {
                            '0'   : 'unburnt',
                            '1'   : 'planned',
                            '2'   : 'wildfire',
                            '1_2' : 'planned_wildfire',
                         }
df[destination_column] = df[source_column].astype('string').str.strip().replace(dict_map).fillna('missing')

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.22 - FEATURE ENRICHMENT: 'weeding_includes_exotic_and_native_competitors_number_of_times_or_y_n_'
fn_view_col_value_status_table(df)
# In[30]:


# PREVIEW RAW DATA VALUES 
source_column          = 'weeding_includes_exotic_and_native_competitors_number_of_times_or_y_n_'
destination_column     = 'weeding'

print('#####  RAW  #####')
fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
df[destination_column] = df[source_column].astype('string').str.strip().str[0]
dict_map               = {
                            'n'   : 'no',
                            'y'   : 'yes',
                            '0'   : 'yes',
                            '1'   : 'yes',
                            '2'   : 'yes',
                            '3'   : 'yes',
                            '4'   : 'yes',
                            '5'   : 'yes',
                            '6'   : 'yes',
                            '7'   : 'yes',
                            '8'   : 'yes',
                            '9'   : 'yes',
                            'a'   : 'yes',
                            'c'   : 'yes',
                            'o'   : 'yes',
                            'r'   : 'yes',
                            't'   : 'yes',
                            'w'   : 'yes',
                            'u'   : 'unk'
                         }
df[destination_column] = df[destination_column].astype('string').replace(dict_map).fillna('missing')

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
print('##### CLEANED #####')
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.23 - FEATURE ENRICHMENT: 'plants_surviving__pct__'

# In[31]:


source_column          = 'plants_surviving__pct__'
destination_column     = 'plants_surviving_pct'
# destination_column_cnt = 'plants_surviving_cnt'

fn_view_feature_data_quality(df[source_column])

##########################################################################################
# NORMALIZE PERCENTAGE OF PLANTS THAT SURVIVED TRANSPLANTING
###########################################################################################
source = df[source_column].astype('string').str.strip()

dict_map = {
    '<NA>'                          : '0.00',
    '<na>'                          : '0.00',
    '??'                            : '0.00',
    '#div/0!'                       : '0.00',
    '#ref!'                         : '0.00',
    '#value!'                       : '0.00',
    'apr_00'                        : '0.00',
    'n_a_annual_'                   : '0.00',
    'n_a_for_annual_species'        : '0.00',
    'n_a_annual_plant_'             : '0.00',
    'n_a_direct_seeding_'           : '0.00',
    'n_a_short_lived_'              : '0.00',
    'Not monitored yet'             : '0.00',
    'unknown_probably_0_'           : '0.00',
    '0_but_substantial_recruitment_': '0.00',
    '<5'                            : '5.00',
    '<10'                           : '10.00',
    '>10'                           : '10.00',
    '>10_pct_'                      : '10.00',
    '16_pct_'                       : '16.00',
    '4_pct_'                        : '4.00', 
    '11_pct_'                       : '11.00',
    'c20'                           : '20.00',
    'c25_pct_'                      : '25.00',
    '17_pct_'                       : '17.00',
    '<20'                           : '20.00',
    'c.20'                          : '20.00',
    '>25'                           : '25.00',
    '>25_pct_'                      : '25.00',
    'c.25'                          : '25.00',
    '20-40_pct_'                    : '30.00',
    '20-40'                         : '30.00',
    'c.25'                          : '25.00',
    '29_pct_'                       : '29.00',
    '~30'                           : '30.00',
    '>30'                           : '30.00',
    '>33'                           : '33.00',
    '~40'                           : '40.00',
    '>40'                           : '40.00',
    '<50?'                          : '50.00',
    '<50_pct_'                      : '50.00',
    '<50_pct_?'                     : '50.00',
    '<50?'                          : '50.00',
    '50_pct_'                       : '50.00',
    '~50'                           : '50.00',
    '>50_pct_'                      : '50.00',
    '>50'                           : '50.00',
    '>50_pct_'                      : '50.00',
    '52_pct_'                       : '52.00',
    '60_pct_'                       : '60.00',
    '69_pct_'                       : '69.00',
    'unknown_but_probably_>75_pct_' : '75.00',
    '77_pct_'                       : '77.00',
    '80_pct_'                       : '80.0',
    '80_pct__across_all_sites_and_treatments_'  : '80.00',
    '80_pct_ (across all sites and treatments)' : '80.00',
    '83_pct_'                       : '83.00',
    '85_pct_'                       : '85.0',
    '>90'                           : '90.00',
    '>95'                           : '95.00',
    '>95_pct_'                      : '95.00',
    '100_pct_'                      : '100.00',
    '>100_with_recruitment_'        : '0.00',
    'good'                          : '0.00',
    'unknown'                       : '0.00',
    'Unknown'                       : '0.00',
    'unknown_number_germinating_not_documented_' : '0.00',
    'unknown_how_many_were_planted_in_total'     : '0.00',
    'unknown_number_of_germinants_not_counted_'  : '0.00',
    'too_early'                     : '0.00',
    'plants_in_the_ground_for_<_12_months'       : '0.00',
    '<12_months_since_planting'     : '0.00',
    '<_12_mnths_since_translocation': '0.00',
    '<12_months_elapsed'            : '0.00',
    '_pct_'                         : '0.00',
}

cleaned = (
    source.replace(dict_map)
          .str.replace('%', '', regex=False)
          .str.strip()
)

parts = cleaned.str.split(':',n=1)

# CREATE NEW COLUMN
# df[destination_column] = (
#    pd.to_numeric(parts.str[0], errors='coerce')
#      .fillna(0.0)
# )

surviving                  = pd.to_numeric(parts.str[0].str.strip(), errors="coerce")
surviving_mean             = surviving.mean()
surviving                  = surviving.fillna(surviving_mean) 

count                      = pd.to_numeric(parts.str[1].str.strip(), errors="coerce") # .fillna(0.0) 
count_mean                 = count.mean()
count                      = count.fillna(count_mean) 

# STORE AS NUMERIC
df[destination_column] = surviving.astype("Float64")
# df["plants_surviving_cnt"] = count.astype("Int32")

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_column_value_counts(df[destination_column])


# ### 3.3.24 - FEATURE ENRICHMENT: 'reproductive_flowering_fruiting_y_n_'

# In[32]:


# PREVIEW RAW DATA VALUES 
source_column          = 'reproductive_flowering_fruiting_y_n_'
destination_column     = 'flowering_fruiting'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
df[destination_column] = df[source_column].astype('string').str[0]
dict_map               = {
                            'y'   : 'yes',
                            'n'   : 'no',
                            't'   : 'unk',
                            'p'   : 'unk',
                            '<'   : 'unk',
                            'u'   : 'unk',
                            'unk' : 'unk'
                         }
df[destination_column] = df[destination_column].astype('string').str.strip().replace(dict_map).fillna('missing')

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.25 - FEATURE ENRICHMENT: 'second_generation_y_n_'

# In[33]:


# PREVIEW RAW DATA VALUES 
source_column          = 'second_generation_y_n_'
destination_column     = 'second_gen'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
df[destination_column] = df[source_column].astype('string').str[0:3]
dict_map               = {
                            'y'   : 'yes',
                            'y_'  : 'yes',
                            'yes' : 'yes',
                            'y_4' : 'yes',
                            'y_f' : 'yes',
                            'y_i' : 'yes',
                            'y_r' : 'yes',
                            'y_v' : 'yes',
                            'n'   : 'no', 
                            'n_'  : 'no',  
                            '0'   : 'no', 
                            '0.0' : 'no', 
                            'too' : 'too_soon',
                            'pla' : 'too_soon',
                            '<12' : 'too_soon',
                            ''    : 'missing',
                            'not' : 'missing',
                            'u'   : 'missing',
                            'unk' : 'missing'
                         }
df[destination_column] = df[destination_column].astype('string').str.strip().replace(dict_map).fillna('missing')

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# ### 3.3.27 - FEATURE ENRICHMENT: 'no.plants_surviving_y_n'

# In[34]:


# PREVIEW RAW DATA VALUES 
source_column           = 'number_of_plants_surviving' # 'no.plants_surviving_y_n' 
destination_column      = 'plants_surviving'

fn_view_feature_data_quality(df[source_column])

# CLEAN DATA  
dict_map                = {
                            "na": "",
                            "+": "999",
                            "<12_months_elapsed": "999",
                            ">100": "999",
                            "n_a_short_lived_": "999",
                            "34_30_just_planted_july2017_": "999",
                            "good_numbers_not_counted_": "999",
                            "<12_months_since_planting": "999",
                            "<_12_mnths_since_translocation": "999",
                            "present": "999",
                            "40_40_just_planted_july2017_": "999",
                            ">20": "999",
                            "not_monitored_yet": "",
                            "not_detailed": "999",
                            ">50": "999",
                            "19_+_12_planted_in_july_2016": "999",
                            "105_+_suckers": "999",
                            "17_+_suckers": "999",
                            "2_+_suckers": "999",
                            "3_+_suckers": "999",
                            "plants_in_the_ground_for_<_12_months": "999",
                            ">200": "999",
                            "unknown_probably_0_": "",
                            "0_non_viable_seed_was_used_": "0.00",
                            "c.500": "999",
                            "5_patches": "999",
                            ">600": "999",
                            ">400": "999",
                            "c.1000": "999",
                            "c.225": "999",
                            "c.15": "999",
                            "4000+": "999",
                            "20_30": "999",
                            ">25": "999",
                            ">2000": "999",
                            "all_plants_surviving": "999",
                            "pending": "",
                            "40_50": "999",
                            "unknown": "",
                            "c.100": "999",
                            "<200": "999",
                            "300_250_just_planted_july2017_": "999",
                            "hundreds": "999",
                            "37_30_just_planted_july2017_": "999",
                            "48_40_just_planted_july2017_": "999",
                          }

cleaned                 = (
                            df[source_column]
                            .astype("string")
                            .str.strip()
                            .replace(dict_map)
                          )

numeric_values          = pd.to_numeric(cleaned, errors="coerce").fillna(0.0)

df[destination_column]  = numeric_values.gt(0).astype(int)

# DROP ORIGINAL COLUMN 
if (source_column != destination_column): 
    df.drop(columns=[source_column], inplace=True)

# PREVIEW CLEANED DATA 
fn_view_feature_data_quality(df[destination_column])


# In[35]:


#DEBUG 
fn_view_col_value_status_table(df)


# ### 3.3.26 - FEATURE ENRICHMENT: 'no._second_generation_plants'

# In[36]:


# PREVIEW RAW DATA VALUES 
source_column           = 'no._second_generation_plants'
destination_column      = '-N/A-'

fn_view_feature_data_quality(df[source_column])

print("\r\n\r\nTOO MANY CATEGORIES AND NOT ENOUGH RECORDS WITHN EACH CATEGORY")

# DROP FEATURE
# df = drop_column(in_dataframe=df, in_columns=source_column)
# df = df.drop(columns=list(source_column))

df                     = df.drop(columns=source_column, errors='ignore')

print(f"\r\n--- FEATURE ['{source_column}'] DROPPED ---")


# ### 3.3.26 - MULTI-CLASS FEATURES
MULTI-CLASS FEATURES ARE ADDRESSED WHEN PREPARING TO SPLIT 

============================================================
**burnt**
COUNT		FREQUENCY	VALUE
============================================================
608		72.29%		unburnt
155		18.43%		missing
66		7.85%		planned
8		0.95%		wildfire
4		0.48%		planned_wildfire
============================================================
  NOTE: the 'planned_wildfire' classification should be broken into both 'planned' and 'wildfire'

============================================================
**translocation**
COUNT	FREQUENCY	VALUE
============================================================
669	79.55%	introduction
154	18.31%	augmentation
14	1.66%	reintroduction
2	0.24%	introduction_reintroduction
2	0.24%	introduction_augmentation_
============================================================
  NOTE: the `introduction_reintroduction` classification should be broken into both `introduction` and `reintroduction`
  NOTE: the `introduction_augmentation_` classification should be broken into both `introduction` and `augmentation` 

# # 5.0 - EDA

# ## 5.1 - REVIEW FEATURES

# In[37]:


fn_view_col_value_status_table(df)


# ## 5.2 - PLOT: 'plants_surviving_pct'

# In[38]:


target_column = "plants_surviving_pct"

df[target_column].hist(bins=20, edgecolor="black")
plt.xlabel(f"{target_column}")
plt.ylabel("Frequency")
plt.title("Distribution of Plant Survival Rates")
plt.show()


# In[39]:


import pandas as pd
import matplotlib.pyplot as plt

# Replace "second_gen" with your column name
target_column = "second_gen"

dict_map = {
    "no": 0.0,
    "yes": 1.0
}

# Normalize strings, map known values, and turn unmatched values into NaN
values = df[target_column].astype("string").str.strip().str.lower()
numeric_values = values.map(dict_map)

# Plot the mapped values
numeric_values.dropna().hist(bins=10, edgecolor="black")
plt.xlabel("Value (0 = no, 1 = yes)")
plt.ylabel("Frequency")
plt.title(f"Distribution of {target_column}")
plt.show()


# ## 5.3 - PLOT: 'second_gen'

# In[40]:


import pandas as pd
import matplotlib.pyplot as plt

target_column = "second_gen"
df_temp = df.copy()

dict_map = {"no": 0.0, "yes": 1.0}

# Normalize text; unrecognized values (including "missing" and "too_soon") become NaN
values = df_temp[target_column].astype("string").str.strip().str.lower()
df_temp[target_column] = values.map(dict_map)

df_temp[target_column].value_counts(dropna=False).sort_index().plot(
    kind="bar", edgecolor="black"
)
plt.xlabel("second_gen (0 = no, 1 = yes; NaN = unrecognized/missing)")
plt.ylabel("Frequency")
plt.title("Counts for second_gen")
plt.xticks(rotation=0)
plt.show()


# ## 5.4 - BONUS: EXPERIMENTAL DISTRIBUTION CHECK
# 
# Checks to review how closely a data series aligns with a statistical distribution

# In[41]:


# TRY TO FIND A REASONABLE DISTRIBUTION TO DESCRIBE THE DATA
import numpy as np
from scipy import stats

x = df["plants_surviving_pct"].dropna().astype(float).to_numpy()

candidates       =  {
                        "normal"     : stats.norm,
                        "lognormal"  : stats.lognorm,
                        "gamma"      : stats.gamma,
                        "weibull"    : stats.weibull_min,
                        "logistic"   : stats.logistic,
                        "student_t"  : stats.t,
                        "uniform"    : stats.uniform,
                        "triangular" : stats.triang,
                        "exponential": stats.expon,
                        "invgamma"   : stats.invgamma,
                        "fisk"       : stats.fisk,
                    }

results          = []

for name, dist in candidates.items():
    try:
        params         = dist.fit(x)
        log_likelihood = np.sum(dist.logpdf(x, *params))
        n_params       = len(params)
        aic            = 2 * n_params - 2 * log_likelihood
        results.append((name, aic, params))
    except (ValueError, FloatingPointError):
        pass

results.sort(key=lambda row: row[1])

print("=" * 80)
print("DISTRIBUTION\tAIC\t\t\tPARAMETERS")
print("=" * 80)
for name, aic, params in results:
    print(f"{name:10s}\tAIC= {aic:.2f}\t\t{params}")

print("=" * 80)
print("Lowest-AIC candidate:\t", results[0][0])
print("=" * 80)


# In[42]:


sns.histplot(df['plants_surviving_pct'], bins=50)


# In[43]:


# PLOT 'second_generation_y_n_'

# CONVERT CATEGORICAL INFORMATION TO NUMERIC USING 'second_generation_0_1_'
dict_map = {
    'yes': 1,
    'no' : 0,
    'unk': 0,
}
source = df['second_gen'].astype('string').str.strip().str.lower()

df['second_generation_0_1'] = (
    source.map(dict_map)
          .fillna(0)
          .astype('int32')
)

# PLOT
sns.histplot(df['second_generation_0_1'], bins=50)

# DROP TEMP COLUMN USED TO CREATE THE PLOT
df.drop(columns=['second_generation_0_1'], inplace=True)


# # 6.0 - PREPARE SPLIT DATASETS

# In[44]:


# REVIEW COLUMNS
fn_view_col_value_status_table(df)


# ## 6.1 -  ADJUST FOR MULTI-CLASS FEATURES

# ### 6.1.1 - INTRO

# MULTI-CLASS records can either be fixed or deleted.  Given that we have not discussed how to applt one-hot encoding to a multi-class record, I propose an approach to address this.  I will not use this approach as it complicates the usability of the  prediction model as it forces the client to one-hot encode their data before using the web API.

# ### 6.1.1 - FIX MULTI-CLASS RECORDS
# CREATE A TEMP DICTIONARY AND ONE-HOT-ENCODE ENTIRE DATASET
train_dicts            = df.fillna(0).to_dict(orient='records')

dv                     = DictVectorizer(sparse=True)
X_train                = dv.fit_transform(train_dicts)

# CONVERT ONE-HOT ENCODED TEMP DICTIONARY BACK INTO PANDAS TO HANDLE WITH MULTI-CLASS CLASSIFICATIONS
X_train_df             = pd.DataFrame.sparse.from_spmatrix(
                            X_train,
                            columns=dv.get_feature_names_out()
                         ).sparse.to_dense()
X_train_df.index       = df.index

# PART 1 - MULTI-CLASS: 'translocation=introduction_augmentation_'
mask = X_train_df['translocation=introduction_augmentation_'].eq(1)

X_train_df.loc[mask, [
    'translocation=introduction',
    'translocation=augmentation'
]] = 1
# DROP MULTI-CLASS FEATURE
X_train_df.drop(columns=["translocation=introduction_augmentation_"], inplace=True)

# ================================================================================
# PART 2 - MULTI-CLASS: 'translocation=introduction_reintroduction'
mask = X_train_df['translocation=introduction_reintroduction'].eq(1)

X_train_df.loc[mask, [
    'translocation=introduction',
    'translocation=reintroduction'
]] = 1
# DROP MULTI-CLASS FEATURE
X_train_df.drop(columns=['translocation=introduction_reintroduction'], inplace=True)

# ================================================================================
# PART 3 - MULTI-CLASS: 'burnt=planned_wildfire''
mask = X_train_df['burnt=planned_wildfire'].eq(1)

X_train_df.loc[mask, [
    'burnt=planned',
    'burnt=wildfire'
]] = 1

# DROP MULTI-CLASS FEATURE
X_train_df.drop(columns=['burnt=planned_wildfire'], inplace=True)

# ================================================================================
# REPLACE ORIGINAL FEATURE DATAFRAME WITH A DATAFRAME ADDRESSING MULTI-CLASS RECORDS 
df = X_train_df.copy()

# ### 6.1.3 -  REMOVE MULTI-CLASS RECORDS

# In[45]:


exclude_features_list   = ['translocation','burnt']

exclude_values          = r'introduction_augmentation_|introduction_reintroduction|planned_wildfire'

for feature in exclude_features_list:
    for status in exclude_values:
        df              = df.loc[
                                ~df[feature].str.contains(
                                    exclude_values, 
                                    na = False
                                )
                          ].copy()
    print(f"'{feature}' cleaned")


# ## 6.2 - MODEL TRAINING

# ### 6.2.1 - LOGISTIC REGRESSION

# #### 6.2.1.1 - SPLIT THE DATASET

# In[47]:


# SPLIT DATASET - LOGISTIC REGRESSION
from sklearn.model_selection import train_test_split

# SET TARGET COLUMN
target_column          = 'plants_surviving'
model_seed             =  123456789
# df[target_column]      = (df["plants_surviving_pct"] > 0.0).astype(int)

df_full_train, df_test = train_test_split(
                            df,
                            test_size    = 0.2,  
                            random_state = model_seed,
                            stratify     = df[target_column]
                        )
df_train, df_val       = train_test_split(
                            df_full_train, 
                            test_size    = 0.25, 
                            random_state = model_seed,
                            stratify     = df_full_train[target_column]
                        )

# RESET INDEX 
df_train               = df_train.reset_index(drop=True)
df_val                 = df_val.reset_index(drop=True)
df_test                = df_test.reset_index(drop=True)

# CREATE TARGET VECTOR
y_train                = (df_train[target_column]).astype('int').values
y_val                  = (df_val[target_column]).astype('int').values
y_test                 = (df_test[target_column]).astype('int').values

# DROP TARGET (plants_surviving) FROM THE FEATURE MATRIX AND DROP PROXY FOR TARGET (plants_surviving_pct)
for in_df in (df_train, df_val, df_test):
    in_df.drop(columns=[target_column, 'plants_surviving_pct'], inplace=True)

# CREATE TRAINING, VALIDATION, AND TEST DICTIONARIES
train_dicts            = df_train.fillna(0).to_dict(orient='records')
val_dicts              = df_val.fillna(0).to_dict(orient='records')
test_dicts             = df_test.fillna(0).to_dict(orient='records')

# CREATE VECTORIZER
dv                     = DictVectorizer(sparse=True)

# CREATE FEATURE MATRIX
X_train                = dv.fit_transform(train_dicts)
X_val                  = dv.transform(val_dicts)
X_test                 = dv.transform(test_dicts)


# DEBUG  - 'y_test' DOES NOT STORE BOTH CASES
fn_view_column_value_counts(pd.DataFrame(y_train))
fn_view_column_value_counts(pd.DataFrame(y_val))
fn_view_column_value_counts(pd.DataFrame(y_test))


# In[48]:


# DEBUG
train_names = set(dv.get_feature_names_out())  # vectorizer used for training


# #### 6.2.1.2 - MAKE THE MODEL

# In[49]:


# from sklearn.linear_model import LogisticRegression

model            =  LogisticRegression(
                        solver   = "lbfgs", 
                        max_iter = 1000
                    )
model.fit(X_train, y_train)


# #### 6.2.1.3 - MAKE A PREDICTION

# In[50]:


# CLASS 1: PLANTS SURVIVIE
y_pred           = model.predict_proba(X_val)[:, 1]
print(y_pred[:5])


# #### 6.2.1.4 - RMSE VALIDATION

# In[51]:


from sklearn.metrics import root_mean_squared_error

predictions      = model.predict(X_val)
rmse             = root_mean_squared_error(y_val, predictions)

print("Validation RMSE:", rmse)


# #### 6.2.1.5 - MODEL TUNING

# In[52]:


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

cv               =  StratifiedKFold(
                        n_splits     = 5, 
                        shuffle      = True, 
                        random_state = 42
                    )

lr_pipeline      = Pipeline(
                        [
                            ("scale",   StandardScaler(
                                            with_mean=False
                                        )
                            ),
                            ("model",   LogisticRegression(
                                            solver       = "lbfgs",
                                            max_iter     = 5000,
                                            random_state = 42,
                                        )
                            ),
                        ]
                    )

param_grid       =  {
                        "model__C": [0.01, 0.1, 1, 10, 100],
                    }

lr_search        =  GridSearchCV(
                        estimator  = lr_pipeline,
                        param_grid = param_grid,
                        scoring    = "roc_auc",
                        cv         = cv,
                        n_jobs     = -1,
                        refit      = True,
                        verbose    = 1,
                    )

lr_search.fit(X_train, y_train)

# Fit the grid search
lr_search.fit(X_train, y_train)

# Save and print the best C value
best_C = lr_search.best_params_["model__C"]
print(f"Best C value: {best_C}")



print("Best logistic regression parameters:", lr_search.best_params_)
print(f"Best CV ROC AUC: {lr_search.best_score_:.3f}")

# ============================================================
# PLOT THE MEAN CV ROC AUC FOR EACH VALUE OF EACH TUNED PARAMETER
# ============================================================
results = pd.DataFrame(lr_search.cv_results_)

summary = (
    results.groupby("param_model__C")["mean_test_score"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(6, 4))
plt.plot(summary["param_model__C"], summary["mean_test_score"], marker="o")
plt.xscale("log")
plt.title("Logistic regression: tuning C (default penalty)")
plt.xlabel("C")
plt.ylabel("Mean CV ROC AUC")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# #### 6.2.1.6 - REVIEW PARAMETER IMPORTANCE

# #### 6.2.1.6 - FINALIZE THE MODEL 

# In[53]:


model_final_lr =  Pipeline(
                [
                    ("scale",   StandardScaler(
                                    with_mean    = False
                                )
                    ),
                    ("model",   LogisticRegression(
                                    C            = 1,
                                    solver       = "lbfgs",
                                    max_iter     = 5000,
                                    random_state = 42,
                                )
                    ),
                ]
            )

# MAKE THE FINALIZED, TUNED MODEL 
model_final_lr.fit(X_train, y_train)


# #### 6.2.1.6 - CONSIDER A SIMPLIFIED MODEL

# In[54]:


import numpy as np
import pandas as pd 

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Permutation importance requires dense input in this setup.
X_val_for_importance = (
    X_val.toarray() if hasattr(X_val, "toarray") else X_val
)

# Calculate permutation importance
result           =  permutation_importance(
                        estimator=model_final_lr,
                        X=X_val_for_importance,
                        y=y_val,
                        scoring="roc_auc",
                        n_repeats=20,
                        random_state=42,
                        n_jobs=-1,
                    )

# Build the feature-importance table
feature_names    = (
                        list(X_val.columns)
                        if hasattr(X_val, "columns")
                        else [f"feature_{i}" for i in range(X_val.shape[1])]
                    )

importance       =  pd.DataFrame(
                        {
                            "feature_index": np.arange(X_val.shape[1]),
                            "feature": feature_names,
                            "importance_mean": result.importances_mean,
                            "importance_std": result.importances_std,
                        }
                    ).sort_values("importance_mean")

# Select candidate features to drop
low_mean_cutoff  = importance["importance_mean"].quantile(0.10)
low_std_cutoff   = importance["importance_std"].median()

drop_mask        =  (
                        (importance["importance_mean"] <= low_mean_cutoff)
                        & (importance["importance_std"] <= low_std_cutoff)
                    )

features_to_drop = importance.loc[drop_mask, "feature"].tolist()
drop_indices     = importance.loc[drop_mask, "feature_index"].to_numpy()

print("Candidate features to drop:", features_to_drop)

# Keep all columns except the candidates
keep_mask        = np.ones(X_train.shape[1], dtype=bool)
keep_mask[drop_indices] = False

def select_columns(X, mask):
    """Select columns from a DataFrame, sparse matrix, or NumPy array."""
    if hasattr(X, "iloc"):
        return X.iloc[:, mask]
    return X[:, mask]

X_train_reduced  = select_columns(X_train, keep_mask)
X_val_reduced    = select_columns(X_val, keep_mask)
X_test_reduced   = select_columns(X_test, keep_mask)

# Fit a new model on the reduced feature set
model_simplified_lr =  Pipeline(
                        [
                            ("scale", StandardScaler(with_mean=False)),
                            ("model", LogisticRegression(
                                C            = 1,  # replace with your selected C value
                                solver       = "lbfgs",
                                max_iter     = 5000,
                                random_state = 42,
                            )),
                        ]
                    )

model_simplified_lr.fit(X_train_reduced, y_train)



print(importance.head(200))

# #### 6.2.1.6 - EVALUATE THE SIMPLIFIED MODEL

# In[55]:


from sklearn.metrics import root_mean_squared_error

predictions      = model_simplified_lr.predict(X_val_reduced)
rmse             = root_mean_squared_error(y_val, predictions)

print("SIMPLIFIED MODEL VALIDATION RMSE:", rmse)

predictions      = model_simplified_lr.predict(X_test_reduced)
rmse             = root_mean_squared_error(y_test, predictions)

print("SIMPLIFIED MODEL TEST RMSE:", rmse)


# ### 6.2.1 - DECISION TREE

# #### 6.3.1.1 - MAKE THE BASE MODEL

# In[56]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Create and fit the decision tree
tree_model       =  DecisionTreeClassifier(
                        max_depth         = 5,              # adjust or tune this
                        min_samples_split = 2,
                        min_samples_leaf  = 1,
                        random_state      = model_seed,
                    )
tree_model.fit(X_train, y_train)

# Predict probabilities and class labels
tree_probs       = tree_model.predict_proba(X_test)[:, 1]
tree_preds       = tree_model.predict(X_test)

# Compare with your fitted logistic-regression model: lr
# lr_probs = lr.predict_proba(X_test)[:, 1]

print(f"Decision tree test ROC AUC:        {roc_auc_score(y_test, tree_probs):.3f}")
# print(f"Logistic regression test ROC AUC:  {roc_auc_score(y_test, lr_probs):.3f}")

print("\nDecision tree classification report:")
print(classification_report(y_test, tree_preds))


# #### 6.3.1.2 - MODEL TUNING

# In[57]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, classification_report

# ==============================================================================
# NEW MODEL TUNING CODE BLOCK #1 STARTS HERE 
# ==============================================================================
cv               =  StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# ==============================================================================
# NEW MODEL TUNING CODE BLOCK #1 ENDS HERE 
# ==============================================================================

tree             =  DecisionTreeClassifier(random_state=42)

param_grid       =  {
                        "max_depth"         : [2, 3, 5, 7, 10, None],
                        "min_samples_split" : [2, 5, 10, 20],
                        "min_samples_leaf"  : [1, 2, 5, 10],
                    }

tree_search      =  GridSearchCV(
                        estimator  = tree,
                        param_grid = param_grid,
                        scoring    = "roc_auc",
                        cv         = cv,
                        n_jobs     = -1,
                        refit      = True,
                        verbose    = 1,
                    )

tree_search.fit(X_train, y_train)

# ==============================================================================
# NEW MODEL TUNING CODE BLOCK #2 STARTS HERE 
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt

# CREATE PLOTS TO EVALUATE HYPER-PARAMETER TUNING
results          =  pd.DataFrame(tree_search.cv_results_)

params           =  ["max_depth", "min_samples_split", "min_samples_leaf"]
fig, axes        =  plt.subplots(1, 3, figsize=(15, 4))

for ax, param in zip(axes, params):
    summary      =  (
                        results.groupby(f"param_{param}", dropna=False)["mean_test_score"]
                        .mean()
                        .reset_index()
                    )

    # Treat values as categories so that None is displayed correctly
    x_labels     =  summary[f"param_{param}"].map(
                        lambda value: "None" if pd.isna(value) else str(value)
                    )

    ax.plot(x_labels, summary["mean_test_score"], marker="o")
    ax.set_title(param)
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Mean CV ROC AUC")
    ax.grid(True, alpha=0.3)

plt.suptitle("Decision tree hyperparameter tuning")
plt.tight_layout()
plt.show()
# ==============================================================================
# NEW MODEL TUNING CODE BLOCK #2 ENDS HERE
# ==============================================================================

print("Best parameters:", tree_search.best_params_)
print(f"Best CV ROC AUC: {tree_search.best_score_:.3f}")

# Evaluate the tuned tree on the held-out test set
best_tree        = tree_search.best_estimator_
tree_probs       = best_tree.predict_proba(X_test)[:, 1]
tree_preds       = best_tree.predict(X_test)

print(f"Test ROC AUC: {roc_auc_score(y_test, tree_probs):.3f}")
print(classification_report(y_test, tree_preds))


# #### 6.3.1.3 - FINALIZE THE TUNED MODEL

# In[58]:


from sklearn.tree import DecisionTreeClassifier

final_tree       =  DecisionTreeClassifier(
                        max_depth         = 5,            # replace with your tuned value
                        min_samples_split = 10,   # replace with your tuned value
                        min_samples_leaf  = 2,     # replace with your tuned value
                        random_state      = model_seed,
                    )

final_tree.fit(X_train, y_train)


# #### 6.3.1.4 - CONSIDER A SIMPLIFIED MODEL

# In[59]:


import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction import DictVectorizer

# ============================================================
# STEP 1 — CREATE A LIST OF ORIGINAL FEATURE NAMES
#          USE THIS LIST TO CREATE A NEW VECTORIZER IF 
#          FEATURES ARE EVENTUALLY DROPPED
# ============================================================
dv_feature_names = list(dv.get_feature_names_out())

# ============================================================
# STEP 2 — FIT THE BASELINE TREE AND MEASURE FEATURE IMPORTANCE
# ============================================================

# Refit the best tree found during tuning.
# If you don't have tree_search, replace tree_search.best_params_
# with your chosen parameters, for example:
# tree_params = {
#     "max_depth": 5,
#     "min_samples_split": 10,
#     "min_samples_leaf": 2,
# }
tree_params      =  tree_search.best_params_

baseline_tree    =  DecisionTreeClassifier(
                        **tree_params,
                        random_state = model_seed,
                    )
baseline_tree.fit(X_train, y_train)

# Densify sparse validation data if needed for permutation_importance.
X_val_for_importance = (
                        X_val.toarray() if hasattr(X_val, "toarray") else X_val
                    )

importance_result = permutation_importance(
                        estimator    = baseline_tree,
                        X            = X_val_for_importance,
                        y            = y_val,
                        scoring      = "accuracy",
                        n_repeats    = 20,
                        random_state = model_seed,
                        n_jobs       = -1,
                    )

feature_names    =  (
                        list(X_val.columns)
                        if hasattr(X_val, "columns")
                        else [f"feature_{i}" for i in range(X_val.shape[1])]
                    )

importance       = pd.DataFrame(
                        {
                            "feature_index": np.arange(X_val.shape[1]),
                            "feature": feature_names,
                            "importance_mean": importance_result.importances_mean,
                            "importance_std": importance_result.importances_std,
                        }
                    ).sort_values("importance_mean")

print("Feature importance:")
print(importance.to_string(index=False))


# ============================================================
# STEP 3 — SELECT CANDIDATE FEATURES AND DROP THEM
# ============================================================

# Candidate features: bottom 10% by mean importance,
# with importance variability below the median.
low_mean_cutoff  = importance["importance_mean"].quantile(0.10)
low_std_cutoff   = importance["importance_std"].median()

candidate_mask   = (
                        (importance["importance_mean"] <= low_mean_cutoff)
                        & (importance["importance_std"] <= low_std_cutoff)
                    )

features_to_drop = importance.loc[candidate_mask, "feature"].tolist()



drop_names = set(features_to_drop)
kept_names = [
    name for name in dv.get_feature_names_out()
    if name not in drop_names
]
simplified_vocabulary = {name: i for i, name in enumerate(kept_names)}
from sklearn.feature_extraction import DictVectorizer
# Fit on raw dictionaries first
simplified_dv = DictVectorizer()
simplified_dv.fit(train_dicts)
# Keep only the selected feature names, with contiguous column indices
kept_names = [
    name for name in dv.get_feature_names_out()
    if name not in set(features_to_drop)
]
simplified_dv.vocabulary_ = {name: i for i, name in enumerate(kept_names)}
# Transform raw dictionaries using the reduced vocabulary
X_train_reduced = simplified_dv.transform(train_dicts)
X_val_reduced   = simplified_dv.transform(val_dicts)
print(f"Features retained: {X_train_reduced.shape[1]}")
print(f"Features retained: {len(kept_names)} of {len(feature_names)}")





drop_indices     = importance.loc[candidate_mask, "feature_index"].to_numpy()

# keep_mask        = np.ones(X_train.shape[1], dtype=bool)
# keep_mask[drop_indices] = False


def select_columns(X, mask):
    """Select columns from a DataFrame, sparse matrix, or NumPy array."""
    if hasattr(X, "iloc"):
        return X.iloc[:, mask]
    return X[:, mask]

X_train_reduced = simplified_dv.transform(train_dicts)
X_val_reduced   = simplified_dv.transform(val_dicts)
X_test_reduced   = simplified_dv.transform(test_dicts)

print("Train shape:", X_train_reduced.shape)
print("Test shape: ", X_test_reduced.shape)

print("\nCandidate features to drop:")
print(features_to_drop)
print(f"Features retained: {keep_mask.sum()} of {len(keep_mask)}")


# ============================================================
# STEP 4 — REFIT THE SIMPLIFIED TREE AND COMPARE ACCURACY
# ============================================================

model_simplified_tree   = DecisionTreeClassifier(
                              **tree_params,
                              random_state = model_seed,
                          )
model_simplified_tree.fit(X_train_reduced, y_train)

baseline_val_accuracy   = accuracy_score(
                              y_val,
                              baseline_tree.predict(X_val),
                          )
simplified_val_accuracy = accuracy_score(
                            y_val,
                            model_simplified_tree.predict(X_val_reduced),
                          )

print(f"\nBaseline validation accuracy:   {baseline_val_accuracy:.3f}")
print(f"Simplified validation accuracy: {simplified_val_accuracy:.3f}")
print(f"Accuracy change:                {simplified_val_accuracy - baseline_val_accuracy:+.3f}")


# #### 6.3.1.5 - FINALIZE THE SIMPLIFIED MODEL

# In[60]:


from sklearn.tree import DecisionTreeClassifier

# RECREATE DICT VECTORIZER
# df


model_final_tree =  DecisionTreeClassifier(
                        max_depth         = 5,    # replace with your tuned value
                        min_samples_split = 10,   # replace with your tuned value
                        min_samples_leaf  = 2,    # replace with your tuned value
                        random_state      = model_seed,
                    )

model_final_tree.fit(X_train_reduced, y_train)


# ### 6.2.3 - MODEL TUNING WITH 'xgboost'

# In[61]:


# ============================================================
# IMPORTS
# ============================================================
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib

# ============================================================
# CROSS-VALIDATION SETUP
# ============================================================
cv               =  StratifiedKFold(
                        n_splits     = 5, 
                        shuffle      = True, 
                        random_state = model_seed
                    )

# ============================================================
# XGBOOST: DEFINE MODEL AND HYPERPARAMETER SEARCH
# ============================================================
xgb              = XGBClassifier(
                       objective    = "binary:logistic",
                       eval_metric  = "logloss",
                       random_state = model_seed,
                       n_jobs       = 1  # avoid oversubscribing CPUs while GridSearchCV runs in parallel
                    )

param_grid       = {
                        "learning_rate": [0.01, 0.03, 0.1],  # eta
                        "max_depth": [3, 5, 7],
                        "min_child_weight": [1, 3, 5],
                    }

xgb_search       = GridSearchCV(
                        estimator  = xgb,
                        param_grid = param_grid,
                        scoring    = "roc_auc",
                        cv         = cv,
                        n_jobs     = -1,
                        refit      = True,
                        verbose    = 1
                    )

# ============================================================
# FIT XGBOOST GRID SEARCH
# ============================================================
xgb_search.fit(X_train_reduced, y_train)

# ============================================================
# FIT LOGISTIC-REGRESSION BASELINE
# ============================================================
lr               = Pipeline([
                        ("scale", StandardScaler(with_mean=False)),  # supports sparse features
                        ("model", LogisticRegression(max_iter=2000)),
                    ])
lr.fit(X_train_reduced, y_train)

# ============================================================
# EVALUATE MODELS ON THE HELD-OUT TEST SET
# ============================================================
xgb_probs        = xgb_search.best_estimator_.predict_proba(X_test_reduced)[:, 1]
lr_probs         = lr.predict_proba(X_test_reduced)[:, 1]
xgb_preds        = xgb_search.best_estimator_.predict(X_test_reduced)

print(f"Best XGBoost parameters:          {xgb_search.best_params_}")
print(f"Best XGBoost CV ROC AUC:          {xgb_search.best_score_:.3f}")
print(f"XGBoost test ROC AUC:             {roc_auc_score(y_test, xgb_probs):.3f}")
print(f"Logistic regression test ROC AUC: {roc_auc_score(y_test, lr_probs):.3f}")

print("\nXGBoost classification report:")
print(classification_report(y_test, xgb_preds))

# ============================================================
# FINALIZE AND SAVE THE SELECTED MODEL
# ============================================================
# With refit=True, GridSearchCV has already refit the best estimator
# on all of X_train and y_train.
final_model      = xgb_search.best_estimator_

joblib.dump(final_model, "final_xgb_model.joblib")
print("\nFinal model saved as 'final_xgb_model.joblib'.")


# # 7.0 - PREPARE BINARY

# ## 7.1 - TEST A MESSAGE

# In[62]:


# ============================================================
#  CREATE A TEST DATAPOINT AS A DICTIONARY
# ============================================================
row              =  X_test.getrow(10)
feature_names    =  simplified_dv.get_feature_names_out()

X_test_datapoint =  {
                        feature_names[i]: float(value)
                        for i, value in zip(row.indices, row.data)
                    }
print(X_test_datapoint)



# In[ ]:




# DEBUG 
for name in simplified_dv.get_feature_names_out():
    print(name)
# ## 7.2 - MAKE A BINARY
# DEBUG 
pipeline_names = set(pipeline.named_steps["vectorizer"].get_feature_names_out())
# In[63]:


# ============================================================
# EXPORT MODEL AS A PIPELINE BINARY 
# ============================================================
from pathlib import Path
import pickle
from sklearn.pipeline import Pipeline

# Combine the DictVectorizer and your trained model into a pipeline
model_pipeline   =  Pipeline(
                        [
                            ('vectorizer', simplified_dv),
                            ('model',      final_model)
                        ]
                    )

output_file = f'model_C={best_C}.bin'  # 'best_C' is created in #6.2.1.5

with open(output_file, 'wb') as f_out:
    pickle.dump(model_pipeline, f_out)

print(f"Model pipeline binary saved to '{output_file}'")


# ## 7.3 - USE THE BINARY

# In[68]:


from pathlib import Path
import pickle

# ============================================================
# LOAD THE TRAINED PIPELINE
# ============================================================
MODEL_PATH = Path(output_file)  # or Path(f"model_C={best_C}.bin")

with MODEL_PATH.open("rb") as f_in:
    pipeline = pickle.load(f_in)

# ============================================================
# MAKE A PREDICTION
# ============================================================
probability = pipeline.predict_proba([X_test_datapoint])[0, 1]
prediction = int(probability >= 0.5)

print("prediction\t", prediction)
print("probability\t", probability)


# ## 7.4 - MAKE predict.py

# In[69]:


print("Model:\t", output_file)  # model_C=0.01.bin


# In[70]:


# ============================================================
# THIS IS THE CONTENT OF predict.py 
# IT CANNOT BE RUN FROM WITHIN A JUPYTER NOTEBOOK DUE TO WORKER CONTENTION
# ============================================================

from   pathlib  import Path
import pickle
import re
from   typing   import Any

import uvicorn
from   fastapi  import FastAPI
from   pydantic import BaseModel, ConfigDict, field_validator, model_validator


# ============================================================
# 1. LOAD THE TRAINED PIPELINE
# ============================================================
# The file exported during training contains a single sklearn Pipeline:
# DictVectorizer followed by the trained model.
MODEL_PATH       = Path("model_C=0.01.bin")

with MODEL_PATH.open("rb") as model_file:
    model_pipeline = pickle.load(model_file)


# ============================================================
# 2. CATEGORY NORMALIZATION MAPS
# ============================================================
EXCLUDE_STATUS_SUMMARY = r"incomplete|unable_to_monitor|finalised "

CONSERVATION_STATUS_MAP = {
                        "en"                        : "endangered",
                        "vu"                        : "vulnerable",
                        "cr"                        : "critically_endangered",
                        "vu_"                       : "vulnerable",
                        "rotap_2rc_"                : "rare",
                        "not_listed_rotap_3rca_"    : "rare",
                        "not_listed_rotap_"         : "not_listed",
                        "ev"                        : "extinct_in_wild",
                    }

LIFEFORM_MAP     = {
                        "shrub"                     : "shrub",
                        "tree"                      : "tree",
                        "perennial_forb"            : "perennial_forb",
                        "ground_orchid"             : "ground_orchid",
                        "perennial_vine"            : "perennial_vine",
                        "annual_forb"               : "annual_forb",
                        "perennial_grass"           : "perennial_grass",
                        "perennial_lily"            : "perennial_lily",
                        "annual_grass"              : "annual_grass",
                        "cycad"                     : "cycad",
                        "perennial_forb_fern_"      : "perennial_forb_fern",
                        "perennial_sedge"           : "perennial_sedge",
                        "perennnial_forb"           : "perennnial_forb",
                        "orchid_epiphytic_"         : "orchid_epiphytic",
                        "palm"                      : "palm",
                        "grass_tree"                : "grass_tree",
                        "tree_mallee_"              : "tree_mallee",
                        "fern"                      : "fern",
                        "epiphyte"                  : "epiphyte",
                        "annual_sedge"              : "annual_sedge",
                        "tree_"                     : "tree",
                        "perennial_aquatic_forb"    : "perennial_aquatic_forb",
                        "shrub_"                    : "shrub",
                    }

HABITAT_MAP      = {
                        "rainforest"                : "rainforest",
                        "southern_shrublands_heathlands_woodlands":
                        "southern_shrublands_heathlands_woodlands",
                        "southern_shrublands_heathlands_woodlands_":
                        "southern_shrublands_heathlands_woodlands",
                        "grassland"                 : "grassland",
                        "grassy_woodland"           : "grassy_woodland",
                        "wetland"                   : "wetland",
                        "dry_sclerophyll"           : "dry_sclerophyll",
                        "mallee"                    : "mallee",
                        "grassland_with_chenopods"  : "grassland_with_chenopods",
                        "coastal_heath"             : "coastal_heath",
                        "wet_sclerophyll"           : "wet_sclerophyll",
                        "banded_ironstone"          : "banded_ironstone",
                        "grassland_sub_tropical_"   : "grassland_sub_tropical",
                        "montane"                   : "montane",
                        "other"                     : "other",
                        "coastal_shrubland"         : "coastal_shrubland",
                        "semi_arid_woodland"        : "semi_arid_woodland",
                        "wetland_floodplain_"       : "wetland_floodplain",
                        "coastal_headland"          : "coastal_headland",
                        "coastal_forest"            : "coastal_forest",
                        "coastal_dunes"             : "coastal_dunes",
                        "grassland_"                : "grassland",
                        "grassland_lomandra_"       : "grassland_lomandra",
                        "wetland_riparian_"         : "wetland_riparian",
                    }

TRANSLOCATION_MAP ={
                        "introduction"              : "introduction",
                        "augmentation"              : "augmentation",
                        "augmentation_experimental_": "augmentation",
                        "introduction_"             : "introduction",
                        "introduction_salvage_"     : "introduction",
                        "introduction_experimental_": "introduction",
                        "introduction_seed_orchard_": "introduction",
                        "augmentation_salvage_"     : "augmentation",
                        "restocking_of_original_introduction_above_": "reintroduction",
                        "reintroduction"            : "reintroduction",
                        "introduction_reintroduction_to_riverland_":
                        "introduction_reintroduction",
                        "restocking_of_above_"      : "reintroduction",
                        "assisted_migration"        : "introduction",
                        "introduction_augmentation_": "introduction_augmentation_",
                        "introduction_different_site_but_same_remnant_as_natural_population_":
                        "introduction",
                    }

YES_NO_MAP       = {
                        "n": "no",
                        "y": "yes",
                        **{str(i): "yes" for i in range(10)},
                        "a": "yes",
                        "c": "yes",
                        "o": "yes",
                        "r": "yes",
                        "t": "yes",
                        "w": "yes",
                        "u": "unk",
                    }

BINARY_YNU_MAP   = {
                        "n": "no", 
                        "y": "yes", 
                        "u": "unk"
                    }
FENCE_MAP        =  {
                        "n": "no", 
                        "y": "yes", 
                        "p": "yes"
                    }
BINARY_NY_MAP    =  {
                        "n": "no", 
                        "y": "yes"
                    }
ZERO_ONE_MAP     =  {
                        "0": "no", 
                        "1": "yes"
                    }
DECIMAL_ZERO_ONE_MAP = {
                        "0.0": "no", 
                        "1.0": "yes", 
                        "u": "unk"
                    }

BURNT_MAP        = {
                        "0"   : "unburnt",
                        "1"   : "planned",
                        "2"   : "wildfire",
                        "1_2" : "planned_wildfire",
                    }

FLOWERING_FRUITING_MAP = {
                        "y": "yes",
                        "n": "no",
                        "t": "unk",
                        "p": "unk",
                        "<": "unk",
                        "u": "unk",
                    }


# ============================================================
# 3. INPUT VALIDATION
# ============================================================
RawValue         = str | int | float | None


class RawPlantInput(BaseModel):
    """Raw source columns accepted by the prediction API."""

    model_config = ConfigDict(extra="forbid")

    status_summary: RawValue                        = None

    # Source columns cleaned or renamed during enrichment.
    conservation_status_epbc_: RawValue             = None
    broad_habitat_group: RawValue                   = None
    translocation_type: RawValue                    = None
    soil_preparation_y_n_: RawValue                 = None
    pre_planting_burn_y_n_: RawValue                = None
    pre_planting_slashing_or_weeding_y_n_: RawValue = None
    pre_site_preparation_all_: RawValue             = None
    herbivore_exclosure_fence_y_n_: RawValue        = None
    herbivore_exclosure_cages_y_n_: RawValue        = None
    tree_guards_y_n_: RawValue                      = None
    grazing_protection_all_: RawValue               = None
    mulching_y_n_: RawValue                         = None
    watering_all_: RawValue                         = None
    burnt_0_unburnt_1_planned_2_wildfire_: RawValue = None
    weeding_includes_exotic_and_native_competitors_number_of_times_or_y_n_: (
        RawValue
    ) = None
    reproductive_flowering_fruiting_y_n_: RawValue  = None

    # Cleaned while retaining their feature names.
    lifeform: RawValue                              = None
    state: RawValue                                 = None

    # Passed through without enrichment.
    mitigation: RawValue                            = None
    second_gen: RawValue                            = None
    research: RawValue                              = None

    @field_validator("status_summary")
    @classmethod
    def reject_excluded_status_summary(
        cls, value: RawValue
    ) -> RawValue:
        if value is not None and re.search(
            EXCLUDE_STATUS_SUMMARY, str(value)
        ):
            raise ValueError(
                "status_summary contains an excluded status."
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def require_at_least_one_field(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not value:
            raise ValueError(
                "Request body must be a non-empty JSON object."
            )
        return value


# ============================================================
# 4. VALUE-CLEANING HELPERS
# ============================================================
def clean_categorical(
    value: Any,
    mapping: dict[str, str],
    missing_value: str = "missing",
) -> str:
    """Strip a categorical value, map known values, and handle missing data."""
    if value is None:
        return missing_value

    cleaned = str(value).strip()
    if not cleaned:
        return missing_value

    return mapping.get(cleaned, cleaned)


def clean_first_character(
    value: Any,
    mapping: dict[str, str],
    missing_value: str = "missing",
) -> str:
    """Normalize the first character of a source value."""
    if value is None:
        return missing_value

    first_char = str(value)[:1].strip()
    if not first_char:
        return first_char

    return mapping.get(first_char, first_char)


def clean_state(value: Any) -> Any:
    """Remove surrounding whitespace and underscores from state values."""
    if value is None:
        return None

    return str(value).strip().replace("_", "")


# ============================================================
# 5. ENRICH RAW INPUT INTO MODEL FEATURES
# ============================================================
def enrich(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert validated raw input into the feature dictionary used in training."""
    row = dict(raw)

    row["lifeform"] = clean_categorical(
        row.get("lifeform"), LIFEFORM_MAP
    )
    row["state"] = clean_state(row.get("state"))

    categorical_transforms = [
        (
            "conservation_status_epbc_",
            "conservation_status",
            CONSERVATION_STATUS_MAP,
            "missing",
        ),
        ("broad_habitat_group", "habitat", HABITAT_MAP, "unk"),
        (
            "translocation_type",
            "translocation",
            TRANSLOCATION_MAP,
            "missing",
        ),
        (
            "pre_site_preparation_all_",
            "pre_site_prep",
            {"0.0": "no", "1.0": "yes", "u": "unk"},
            "missing",
        ),
        ("watering_all_", "watering", DECIMAL_ZERO_ONE_MAP, "missing"),
        (
            "burnt_0_unburnt_1_planned_2_wildfire_",
            "burnt",
            BURNT_MAP,
            "missing",
        ),
    ]

    for source, destination, mapping, missing_value in categorical_transforms:
        row[destination] = clean_categorical(
            row.get(source), mapping, missing_value
        )
        if source != destination:
            row.pop(source, None)

    first_character_transforms = [
        ("soil_preparation_y_n_", "soil_prep", BINARY_YNU_MAP, "missing"),
        (
            "pre_planting_burn_y_n_",
            "pre_planting_burn",
            BINARY_YNU_MAP,
            "missing",
        ),
        (
            "pre_planting_slashing_or_weeding_y_n_",
            "pre_planting_slashing_or_weeding",
            BINARY_YNU_MAP,
            "missing",
        ),
        (
            "herbivore_exclosure_fence_y_n_",
            "exclosure_fence",
            FENCE_MAP,
            "missing",
        ),
        (
            "herbivore_exclosure_cages_y_n_",
            "exclosure_cage",
            BINARY_YNU_MAP,
            "unk",
        ),
        ("tree_guards_y_n_", "tree_guard", BINARY_YNU_MAP, "missing"),
        (
            "grazing_protection_all_",
            "grazing_protection",
            ZERO_ONE_MAP,
            "missing",
        ),
        ("mulching_y_n_", "mulch", BINARY_NY_MAP, "missing"),
        (
            "weeding_includes_exotic_and_native_competitors_number_of_times_or_y_n_",
            "weeding",
            YES_NO_MAP,
            "missing",
        ),
        (
            "reproductive_flowering_fruiting_y_n_",
            "flowering_fruiting",
            FLOWERING_FRUITING_MAP,
            "missing",
        ),
    ]

    for source, destination, mapping, missing_value in first_character_transforms:
        row[destination] =  clean_first_character(
                                row.get(source), 
                                mapping, 
                                missing_value
                            )
        if source != destination:
            row.pop(source, None)

    return row


# ============================================================
# 6. API AND RESPONSE FORMAT
# ============================================================
class PredictResponse(BaseModel):
    prediction_probability: float
    prediction: int


app = FastAPI(title="plant-survival-prediction")


# ============================================================
# 7. PREDICTION ENDPOINT
# ============================================================
@app.post("/predict", response_model=PredictResponse)
def predict(raw_input: RawPlantInput) -> PredictResponse:
    """Enrich input, run the pipeline, and return its prediction."""
    raw_features = raw_input.model_dump(exclude_unset=True)
    raw_features.pop("status_summary", None)

    enriched_features = enrich(raw_features)

    # The pipeline applies its DictVectorizer and then the trained model.
    probability  =  float(
                        model_pipeline.predict_proba([enriched_features])[0, 1]
                    )
    prediction   =  int(probability >= 0.5)

    return PredictResponse(
        prediction_probability = probability,
        prediction             = prediction,
    )


# ============================================================
# 8. LOCAL DEVELOPMENT ENTRY POINT
# ============================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)


# In[78]:


# ============================================================
# TEST THE MODEL 
# ============================================================
import requests
import json

url              = 'http://localhost:9696/predict'

observation      = {
                      "status_summary"                       : "finalised",
                      "conservation_status_epbc_"            : "en",
                      "broad_habitat_group"                  : "rainforest",
                      "translocation_type"                   : "introduction",
                      "soil_preparation_y_n_"                : "y_slow_release_fertilizer",
                      "pre_planting_burn_y_n_"               : "n",
                      "pre_planting_slashing_or_weeding_y_n_": "y",
                      "pre_site_preparation_all_"            : "1.0",
                      "herbivore_exclosure_fence_y_n_"       : "y",
                      "herbivore_exclosure_cages_y_n_"       : "n",
                      "tree_guards_y_n_"                     : "n",
                      "grazing_protection_all_"              : "1.0",
                      "mulching_y_n_"                        : "n",
                      "watering_all_"                        : "0.0",
                      "burnt_0_unburnt_1_planned_2_wildfire_": "0",
                      "weeding_includes_exotic_and_native_competitors_number_of_times_or_y_n_": "4_years",
                      "reproductive_flowering_fruiting_y_n_" : "n",
                      "lifeform"                             : "tree",
                      "state"                                : "nsw",
                      "mitigation"                           : "y",
                      "second_gen"                           : "0",
                      "research"                             : "previous_translocation"
                    }

response         = requests.post(url, json=observation, timeout=30)
response.raise_for_status()

result           = response.json()
print(json.dumps(result, indent=2))

# ============================================================
# CAPTURE OF A POWERSHELL SCRIPT
# ============================================================
$body = @'
{
  "status_summary": "finalised",
  "conservation_status_epbc_": "en",
  "broad_habitat_group": "rainforest",
  "translocation_type": "introduction",
  "soil_preparation_y_n_": "y_slow_release_fertilizer",
  "pre_planting_burn_y_n_": "n",
  "pre_planting_slashing_or_weeding_y_n_": "y",
  "pre_site_preparation_all_": "1.0",
  "herbivore_exclosure_fence_y_n_": "y",
  "herbivore_exclosure_cages_y_n_": "n",
  "tree_guards_y_n_": "n",
  "grazing_protection_all_": "1.0",
  "mulching_y_n_": "n",
  "watering_all_": "0.0",
  "burnt_0_unburnt_1_planned_2_wildfire_": "0",
  "weeding_includes_exotic_and_native_competitors_number_of_times_or_y_n_": "4_years",
  "reproductive_flowering_fruiting_y_n_": "n",
  "lifeform": "tree",
  "state": "nsw",
  "mitigation": "y",
  "second_gen": "0",
  "research": "previous_translocation"
}
'@

$response = Invoke-RestMethod `
  -Uri "http://localhost:9696/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

$response# ============================================================
# CAPTURE OF A POWERSHELL SESSION - DO NOT REMOVE
# ============================================================
PS C:\Users\duffl> $body = @'
{
  "status_summary": "finalised",
  "conservation_status_epbc_": "en",
  "broad_habitat_group": "rainforest",
  "translocation_type": "introduction",
  "soil_preparation_y_n_": "y_slow_release_fertilizer",
  "pre_planting_burn_y_n_": "n",
  "pre_planting_slashing_or_weeding_y_n_": "y",
  "pre_site_preparation_all_": "1.0",
  "herbivore_exclosure_fence_y_n_": "y",
  "herbivore_exclosure_cages_y_n_": "n",
  "tree_guards_y_n_": "n",
  "grazing_protection_all_": "1.0",
  "mulching_y_n_": "n",
  "watering_all_": "0.0",
  "burnt_0_unburnt_1_planned_2_wildfire_": "0",
  "weeding_includes_exotic_and_native_competitors_number_of_times_or_y_n_": "4_years",
  "reproductive_flowering_fruiting_y_n_": "n",
  "lifeform": "tree",
  "state": "nsw",
  "mitigation": "y",
  "second_gen": "0",
  "research": "previous_translocation"
}
'@

$response = Invoke-RestMethod `
  -Uri "http://localhost:9696/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

$response

prediction_probability prediction
---------------------- ----------
    0.9270420670509338          1


import pickle
from pathlib import Path
from typing import Any
import re
import joblib

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# ============================================================
# 1. LOAD THE TRAINED MODEL
# ============================================================
# Load the fitted DictVectorizer and model when this module starts.
MODEL_PATH = Path("model_C=0.01.bin")

with MODEL_PATH.open("rb") as model_file:
    dv, model = pickle.load(model_file)


MODEL_PATH = Path("final_xgb_model.joblib")
saved = joblib.load(MODEL_PATH)

simplified_dv = saved["vectorizer"]  
model = saved["model"]






# ============================================================
# 2. DEFINE AND VALIDATE RAW API INPUT
# ============================================================
# Inputs may be strings, numbers, or null. The API accepts source-column
# names; enrich() below converts them into the feature names used in training.
RawValue = str | int | float | None


class RawPlantInput(BaseModel):
    """Raw source columns accepted by the prediction API."""

    model_config = ConfigDict(extra="forbid")

    status_summary: RawValue = None

    # Source columns that are cleaned or renamed during enrichment.
    conservation_status_epbc_: RawValue = None
    broad_habitat_group: RawValue = None
    translocation_type: RawValue = None
    soil_preparation_y_n_: RawValue = None
    pre_planting_burn_y_n_: RawValue = None
    pre_planting_slashing_or_weeding_y_n_: RawValue = None
    pre_site_preparation_all_: RawValue = None
    herbivore_exclosure_fence_y_n_: RawValue = None
    herbivore_exclosure_cages_y_n_: RawValue = None
    tree_guards_y_n_: RawValue = None
    grazing_protection_all_: RawValue = None
    mulching_y_n_: RawValue = None
    watering_all_: RawValue = None
    burnt_0_unburnt_1_planned_2_wildfire_: RawValue = None
    weeding_includes_exotic_and_native_competitors_number_of_times_or_y_n_: RawValue = None
    reproductive_flowering_fruiting_y_n_: RawValue = None

    # Fields cleaned while retaining their original feature names.
    lifeform: RawValue = None
    state: RawValue = None

    # Fields passed through without enrichment.
    mitigation: RawValue = None
    second_gen: RawValue = None
    research: RawValue = None

    @field_validator("status_summary")
    @classmethod
    def reject_excluded_status_summary(cls, value: RawValue) -> RawValue:
        if value is not None and re.search(EXCLUDE_STATUS_SUMMARY, str(value)):
            raise ValueError("status_summary contains an excluded status.")
        return value

    @model_validator(mode="before")
    @classmethod
    def require_at_least_one_field(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not value:
            raise ValueError("Request body must be a non-empty JSON object.")
        return value


# ============================================================
# 3. DEFINE CATEGORY NORMALIZATION MAPS
# ============================================================
# These maps convert raw values into the normalized category labels expected
# by the model. Unmapped values are retained by the cleaning functions below.
EXCLUDE_STATUS_SUMMARY = r"incomplete|unable_to_monitor|finalised "

CONSERVATION_STATUS_MAP = {
    "en": "endangered",
    "vu": "vulnerable",
    "cr": "critically_endangered",
    "vu_": "vulnerable",
    "rotap_2rc_": "rare",
    "not_listed_rotap_3rca_": "rare",
    "not_listed_rotap_": "not_listed",
    "ev": "extinct_in_wild",
}

LIFEFORM_MAP = {
    "shrub": "shrub",
    "tree": "tree",
    "perennial_forb": "perennial_forb",
    "ground_orchid": "ground_orchid",
    "perennial_vine": "perennial_vine",
    "annual_forb": "annual_forb",
    "perennial_grass": "perennial_grass",
    "perennial_lily": "perennial_lily",
    "annual_grass": "annual_grass",
    "cycad": "cycad",
    "perennial_forb_fern_": "perennial_forb_fern",
    "perennial_sedge": "perennial_sedge",
    "perennnial_forb": "perennnial_forb",
    "orchid_epiphytic_": "orchid_epiphytic",
    "palm": "palm",
    "grass_tree": "grass_tree",
    "tree_mallee_": "tree_mallee",
    "fern": "fern",
    "epiphyte": "epiphyte",
    "annual_sedge": "annual_sedge",
    "tree_": "tree",
    "perennial_aquatic_forb": "perennial_aquatic_forb",
    "shrub_": "shrub",
}

HABITAT_MAP = {
    "rainforest": "rainforest",
    "southern_shrublands_heathlands_woodlands":
        "southern_shrublands_heathlands_woodlands",
    "southern_shrublands_heathlands_woodlands_":
        "southern_shrublands_heathlands_woodlands",
    "grassland": "grassland",
    "grassy_woodland": "grassy_woodland",
    "wetland": "wetland",
    "dry_sclerophyll": "dry_sclerophyll",
    "mallee": "mallee",
    "grassland_with_chenopods": "grassland_with_chenopods",
    "coastal_heath": "coastal_heath",
    "wet_sclerophyll": "wet_sclerophyll",
    "banded_ironstone": "banded_ironstone",
    "grassland_sub_tropical_": "grassland_sub_tropical",
    "montane": "montane",
    "other": "other",
    "coastal_shrubland": "coastal_shrubland",
    "semi_arid_woodland": "semi_arid_woodland",
    "wetland_floodplain_": "wetland_floodplain",
    "coastal_headland": "coastal_headland",
    "coastal_forest": "coastal_forest",
    "coastal_dunes": "coastal_dunes",
    "grassland_": "grassland",
    "grassland_lomandra_": "grassland_lomandra",
    "wetland_riparian_": "wetland_riparian",
}

TRANSLOCATION_MAP = {
    "introduction": "introduction",
    "augmentation": "augmentation",
    "augmentation_experimental_": "augmentation",
    "introduction_": "introduction",
    "introduction_salvage_": "introduction",
    "introduction_experimental_": "introduction",
    "introduction_seed_orchard_": "introduction",
    "augmentation_salvage_": "augmentation",
    "restocking_of_original_introduction_above_": "reintroduction",
    "reintroduction": "reintroduction",
    "introduction_reintroduction_to_riverland_": "introduction_reintroduction",
    "restocking_of_above_": "reintroduction",
    "assisted_migration": "introduction",
    "introduction_augmentation_": "introduction_augmentation_",
    "introduction_different_site_but_same_remnant_as_natural_population_":
        "introduction",
}

# General yes/no-like values used by the weeding feature.
YES_NO_MAP = {
    "n": "no",
    "y": "yes",
    **{str(i): "yes" for i in range(10)},
    "a": "yes",
    "c": "yes",
    "o": "yes",
    "r": "yes",
    "t": "yes",
    "w": "yes",
    "u": "unk",
}

# Maps used by different binary or categorical source fields.
BINARY_YNU_MAP = {"n": "no", "y": "yes", "u": "unk"}
FENCE_MAP = {"n": "no", "y": "yes", "p": "yes"}
BINARY_NY_MAP = {"n": "no", "y": "yes"}
ZERO_ONE_MAP = {"0": "no", "1": "yes"}
DECIMAL_ZERO_ONE_MAP = {"0.0": "no", "1.0": "yes", "u": "unk"}
BURNT_MAP = {
    "0": "unburnt",
    "1": "planned",
    "2": "wildfire",
    "1_2": "planned_wildfire",
}
FLOWERING_FRUITING_MAP = {
    "y": "yes",
    "n": "no",
    "t": "unk",
    "p": "unk",
    "<": "unk",
    "u": "unk",
}


# ============================================================
# 4. CLEAN INDIVIDUAL VALUES
# ============================================================
def clean_categorical(
    value: Any,
    mapping: dict[str, str],
    missing_value: str = "missing",
) -> str:
    """Strip a full categorical value, map known values, and handle missing data."""
    if value is None:
        return missing_value

    cleaned = str(value).strip()
    if not cleaned:
        return missing_value

    return mapping.get(cleaned, cleaned)


def clean_first_character(
    value: Any,
    mapping: dict[str, str],
    missing_value: str = "missing",
) -> str:
    """Use and normalize the first character of a source value."""
    if value is None:
        return missing_value

    # Matches the supplied pandas pattern: .str[0].str.strip()
    first_char = str(value)[:1].strip()
    if not first_char:
        return first_char

    return mapping.get(first_char, first_char)


def clean_state(value: Any) -> Any:
    """Remove surrounding whitespace and underscores from state values."""
    if value is None:
        return None

    return str(value).strip().replace("_", "")


# ============================================================
# 5. ENRICH RAW INPUT INTO MODEL FEATURES
# ============================================================
def enrich(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert validated raw input into the feature dictionary used in training."""
    row = dict(raw)

    # Clean fields that retain their original feature names.
    row["lifeform"] = clean_categorical(row.get("lifeform"), LIFEFORM_MAP)
    row["state"] = clean_state(row.get("state"))

    # Normalize full-string values and rename them to training feature names.
    categorical_transforms = [
        (
            "conservation_status_epbc_",
            "conservation_status",
            CONSERVATION_STATUS_MAP,
            "missing",
        ),
        ("broad_habitat_group", "habitat", HABITAT_MAP, "unk"),
        ("translocation_type", "translocation", TRANSLOCATION_MAP, "missing"),
        (
            "pre_site_preparation_all_",
            "pre_site_prep",
            {"0.0": "no", "1.0": "yes", "u": "unk"},
            "missing",
        ),
        ("watering_all_", "watering", DECIMAL_ZERO_ONE_MAP, "missing"),
        (
            "burnt_0_unburnt_1_planned_2_wildfire_",
            "burnt",
            BURNT_MAP,
            "missing",
        ),
    ]

    for source, destination, mapping, missing_value in categorical_transforms:
        row[destination] = clean_categorical(
            row.get(source),
            mapping,
            missing_value,
        )
        if source != destination:
            row.pop(source, None)

    # Normalize the first character of these source values and rename features.
    first_character_transforms = [
        ("soil_preparation_y_n_", "soil_prep", BINARY_YNU_MAP, "missing"),
        ("pre_planting_burn_y_n_", "pre_planting_burn", BINARY_YNU_MAP, "missing"),
        (
            "pre_planting_slashing_or_weeding_y_n_",
            "pre_planting_slashing_or_weeding",
            BINARY_YNU_MAP,
            "missing",
        ),
        ("herbivore_exclosure_fence_y_n_", "exclosure_fence", FENCE_MAP, "missing"),
        ("herbivore_exclosure_cages_y_n_", "exclosure_cage", BINARY_YNU_MAP, "unk"),
        ("tree_guards_y_n_", "tree_guard", BINARY_YNU_MAP, "missing"),
        ("grazing_protection_all_", "grazing_protection", ZERO_ONE_MAP, "missing"),
        ("mulching_y_n_", "mulch", BINARY_NY_MAP, "missing"),
        (
            "weeding_includes_exotic_and_native_competitors_number_of_times_or_y_n_",
            "weeding",
            YES_NO_MAP,
            "missing",
        ),
        (
            "reproductive_flowering_fruiting_y_n_",
            "flowering_fruiting",
            FLOWERING_FRUITING_MAP,
            "missing",
        ),
    ]

    for source, destination, mapping, missing_value in first_character_transforms:
        row[destination] = clean_first_character(
            row.get(source),
            mapping,
            missing_value,
        )
        if source != destination:
            row.pop(source, None)

    return row


# ============================================================
# 6. DEFINE THE API AND RESPONSE FORMAT
# ============================================================
class PredictResponse(BaseModel):
    """JSON response returned by the prediction endpoint."""

    prediction_probability: float
    prediction: int


app = FastAPI(title="plant-survival-prediction")


# ============================================================
# 7. PREDICTION ENDPOINT
# ============================================================
@app.post("/predict", response_model=PredictResponse)

def predict(raw_input: RawPlantInput) -> PredictResponse:
    """Enrich validated input, run the model, and return its prediction."""
    # Keep only fields the caller supplied, then transform them into model inputs.
    raw_features = raw_input.model_dump(exclude_unset=True)
    raw_features.pop("status_summary", None)
    
    enriched_features = enrich(raw_features)

    # Vectorize the feature dictionary and get the probability of class 1.
    X = dv.transform([enriched_features])
    probability = float(model.predict_proba(X)[0, 1])

    # Convert the probability to a binary prediction using a 0.5 threshold.
    prediction = int(probability >= 0.5)

    return PredictResponse(
        prediction_probability=probability,
        prediction=prediction,
    )


# ============================================================
# 8. LOCAL DEVELOPMENT ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Run the API on port 9696 when this file is executed directly.
    uvicorn.run(app, host="0.0.0.0", port=9696)

# In[ ]:




