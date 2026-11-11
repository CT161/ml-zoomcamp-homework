#!/usr/bin/env python
# coding: utf-8

# # 1.0 - LOAD PACKAGES

# ## 1.1 - LOAD PACKAGES AND GET DATA

import pandas       as pd
import numpy        as np
import re
from pathlib import Path
import pickle
import joblib

# PLOTTING
import seaborn      as sns
from   matplotlib               import pyplot as plt

# LOGISTIC REGRESSION
from sklearn.linear_model         import LogisticRegression

# DECISION TREES AND METRICS
from   sklearn.feature_extraction import DictVectorizer
from   sklearn.model_selection    import train_test_split, GridSearchCV, StratifiedKFold
from   sklearn.metrics            import (
                                    roc_auc_score,
                                    mean_squared_error,
                                    classification_report,
                                    accuracy_score
                                )
from   sklearn.tree               import (
                                    DecisionTreeClassifier,
                                    DecisionTreeRegressor,
                                    export_text
                                )
from   sklearn.ensemble           import RandomForestRegressor
from   sklearn.pipeline           import Pipeline
from   sklearn.preprocessing      import StandardScaler
from   sklearn.inspection         import permutation_importance
from   xgboost                    import XGBClassifier


# # 2.0 - HELPER FUNCTIONS

def fn_clean_text(value):
    if not isinstance(value, str):
        return value

    value = value.lower()
    value = value.replace('%', '_pct_')
    value = re.sub(r'[\s()/\-]+', '_', value)
    value = re.sub(r"""['"]""", '', value)
    value = re.sub(r'[,;?]', '', value)
    return value

def drop_column(in_dataframe, in_columns):
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


def fn_view_col_value_status_table(in_data, in_df=None):
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


# # 3.0 - PREPARE DATA

# ## 3.1 - GET AND PREVIEW DATA

data_path  = "C:/Users/duffl/anaconda3/_CT161_GitHub_Repos/ml-zoomcamp-homework/07_midterm_project_plants/data/Australian_plant_translocation_database_PUBLIC_2018(ALL_TRANSLOCATIONS).csv"
df         = pd.read_csv(data_path)

df.columns             = [fn_clean_text(column) for column in df.columns]
df                     = df.map(fn_clean_text)

if 'habitat_condition_e.g._highly_disturbed_small_remnant_large_remnant_' in df.columns:
    df['habitat_condition'] = df['habitat_condition_e.g._highly_disturbed_small_remnant_large_remnant_']
    df.drop(columns=['habitat_condition_e.g._highly_disturbed_small_remnant_large_remnant_'], inplace=True)

fn_view_col_value_status_table(df)


# ## 3.2 - FIRST PASS CLEANING

exclude_status_summary = r'incomplete|unable_to_monitor|finalised '

if 'status_summary' in df.columns:
    df                     = df.loc[
                              ~df['status_summary'].str.contains(
                                exclude_status_summary, 
                                na=False
                              )
                            ].copy()
    print('records dropped')

exclude_features_list   =    [
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

df                     = df.drop(columns=exclude_features_list, errors='ignore')


# ## 3.3 - DATA CLEANING

source_column          = 'treatment_number'
destination_column     = 'treatment'

if source_column in df.columns:
    fn_view_feature_data_quality(source_column, in_df=df)
    dict_map               = {'0' : 'no', '1' : 'yes', 'u' : 'unk'}
    df[destination_column] = df[source_column].astype('string').str.strip().replace(dict_map).fillna('missing')
    if (source_column != destination_column): 
        df.drop(columns=[source_column], inplace=True)

source_column          = 'lifeform'
destination_column     = 'lifeform'

if source_column in df.columns:
    dict_map               = {
                                'shrub': 'shrub', 'tree': 'tree', 'perennial_forb': 'perennial_forb',
                                'ground_orchid': 'ground_orchid', 'perennial_vine': 'perennial_vine',
                                'annual_forb': 'annual_forb', 'perennial_grass': 'perennial_grass',
                                'perennial_lily': 'perennial_lily', 'annual_grass': 'annual_grass',
                                'cycad': 'cycad', 'perennial_forb_fern_': 'perennial_forb_fern',
                                'perennial_sedge': 'perennial_sedge', 'perennnial_forb': 'perennnial_forb',
                                'orchid_epiphytic_': 'orchid_epiphytic', 'palm': 'palm',
                                'grass_tree': 'grass_tree', 'tree_mallee_': 'tree_mallee',
                                'fern': 'fern', 'epiphyte': 'epiphyte', 'annual_sedge': 'annual_sedge',
                                'tree_': 'tree', 'perennial_aquatic_forb': 'perennial_aquatic_forb',
                                'shrub_' : 'shrub'
                             }
    df[destination_column] = df[source_column].astype('string').str.strip().replace(dict_map).fillna('missing')

source_column           = 'state' 
destination_column      = 'state' 

if source_column in df.columns:
    df[destination_column]  = df[source_column].str.strip().replace('qld_', 'qld')

source_column           = 'conservation_status_epbc_' 
destination_column      = 'conservation_status' 

if source_column in df.columns:
    dict_map               = {
                                'en' : 'endangered', 'vu' : 'vulnerable', 'cr' : 'critically_endangered',
                                'vu_' : 'vulnerable', 'rotap_2rc_' : 'rare', 'not_listed_rotap_3rca_' : 'rare',
                                'not_listed_rotap_' : 'not_listed', 'ev' : 'extinct_in_wild',
                             }
    df[destination_column] = df[source_column].astype('string').str.strip().replace(dict_map).fillna('missing') 
    if (source_column != destination_column): 
        df.drop(columns=[source_column], inplace=True)

source_column          = 'broad_habitat_group'
destination_column     = 'habitat'

if source_column in df.columns:
    dict_map               = {
                                'rainforest': 'rainforest',
                                'southern_shrublands_heathlands_woodlands': 'southern_shrublands_heathlands_woodlands',
                                'southern_shrublands_heathlands_woodlands_': 'southern_shrublands_heathlands_woodlands',
                                'grassland': 'grassland', 'grassy_woodland': 'grassy_woodland',
                                'wetland': 'wetland', 'dry_sclerophyll': 'dry_sclerophyll',
                                'mallee': 'mallee', 'grassland_with_chenopods': 'grassland_with_chenopods',
                                'coastal_heath': 'coastal_heath', 'wet_sclerophyll': 'wet_sclerophyll',
                                'banded_ironstone': 'banded_ironstone', 'grassland_sub_tropical_': 'grassland_sub_tropical',
                                'montane': 'montane', 'other': 'other', 'coastal_shrubland': 'coastal_shrubland',
                                'semi_arid_woodland': 'semi_arid_woodland', 'wetland_floodplain_': 'wetland_floodplain',
                                'coastal_headland': 'coastal_headland', 'coastal_forest': 'coastal_forest',
                                'coastal_dunes': 'coastal_dunes', 'grassland_': 'grassland',
                                'grassland_lomandra_': 'grassland_lomandra', 'wetland_riparian_': 'wetland_riparian',
                           }
    df[destination_column] = df[source_column].astype('string').str.strip().replace(dict_map).fillna('unk')
    if (source_column != destination_column): 
        df.drop(columns=[source_column], inplace=True)

source_column          = 'research_to_support_translocation_'
destination_column     = 'research'

if source_column in df.columns:
    is_empty               = df[source_column].isna() | df[source_column].astype('string').str.strip().eq('')
    is_underscore          = df[source_column].astype('string').str.strip().eq('_')
    df[destination_column] = 'yes'
    df.loc[is_empty | is_underscore, destination_column] = 'no'
    if (source_column != destination_column): 
        df.drop(columns=[source_column], inplace=True)

source_column          = 'mitigation_translocation'
destination_column     = 'mitigation'

if source_column in df.columns:
    df[destination_column] = df[source_column].astype('string').str[0]
    dict_map               = {'n' : 'no', 'y' : 'yes', 'u' : 'unk'}
    df[destination_column] = df[destination_column].astype('string').str.strip().replace(dict_map).fillna('missing')
    if (source_column != destination_column): 
        df.drop(columns=[source_column], inplace=True)

source_column           = 'habitat_condition'
if source_column in df.columns:
    df = drop_column(in_dataframe=df, in_columns=source_column)

source_column          = 'translocation_type'
destination_column     = 'translocation'

if source_column in df.columns:
    dict_map               = {
                                'introduction'                : 'introduction',
                                'augmentation'                : 'augmentation',
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
    if (source_column != destination_column): 
        df.drop(columns=[source_column], inplace=True)

source_column           = 'propagule_type_seed_seedling_cutting_tissue_culture_whole_plants_etc_'
if source_column in df.columns:
    df = drop_column(in_dataframe=df, in_columns=source_column)

for col_name, dest_name in [
    ('soil_preparation_y_n_', 'soil_prep'),
    ('pre_planting_burn_y_n_', 'pre_planting_burn'),
    ('pre_planting_slashing_or_weeding_y_n_', 'pre_planting_slashing_or_weeding'),
    ('herbivore_exclosure_fence_y_n_', 'exclosure_fence'),
    ('herbivore_exclosure_cages_y_n_', 'exclosure_cage'),
    ('tree_guards_y_n_', 'tree_guard'),
    ('grazing_protection_all_', 'grazing_protection'),
    ('mulching_y_n_', 'mulch')
]:
    if col_name in df.columns:
        df[dest_name] = df[col_name].astype('string').str[0].str.strip().replace({'n': 'no', 'y': 'yes', 'u': 'unk', 'p': 'yes', '0': 'no', '1': 'yes'}).fillna('missing')
        if col_name != dest_name:
            df.drop(columns=[col_name], inplace=True)

if 'pre_site_preparation_all_' in df.columns:
    df['pre_site_prep'] = df['pre_site_preparation_all_'].astype('string').str.strip().replace({'0.0': 'no', '1.0': 'yes', 'u': 'unk'}).fillna('missing')
    df.drop(columns=['pre_site_preparation_all_'], inplace=True)

if 'watering_all_' in df.columns:
    df['watering'] = df['watering_all_'].astype('string').str.strip().replace({'0.0': 'no', '1.0': 'yes', 'u': 'unk'}).fillna('missing')
    df.drop(columns=['watering_all_'], inplace=True)

if 'burnt_0_unburnt_1_planned_2_wildfire_' in df.columns:
    dict_map = {'0': 'unburnt', '1': 'planned', '2': 'wildfire', '1_2': 'planned_wildfire'}
    df['burnt'] = df['burnt_0_unburnt_1_planned_2_wildfire_'].astype('string').str.strip().replace(dict_map).fillna('missing')
    df.drop(columns=['burnt_0_unburnt_1_planned_2_wildfire_'], inplace=True)

if 'weeding_includes_exotic_and_native_competitors_number_of_times_or_y_n_' in df.columns:
    dict_map = {
        'n': 'no', 'y': 'yes', '0': 'yes', '1': 'yes', '2': 'yes', '3': 'yes', '4': 'yes', 
        '5': 'yes', '6': 'yes', '7': 'yes', '8': 'yes', '9': 'yes', 'a': 'yes', 'c': 'yes', 
        'o': 'yes', 'r': 'yes', 't': 'yes', 'w': 'yes', 'u': 'unk'
    }
    df['weeding'] = df['weeding_includes_exotic_and_native_competitors_number_of_times_or_y_n_'].astype('string').str.strip().str[0].replace(dict_map).fillna('missing')
    df.drop(columns=['weeding_includes_exotic_and_native_competitors_number_of_times_or_y_n_'], inplace=True)

if 'plants_surviving__pct__' in df.columns:
    source = df['plants_surviving__pct__'].astype('string').str.strip()
    dict_map = {
        '<NA>': '0.00', '<na>': '0.00', '??': '0.00', '#div/0!': '0.00', '#ref!': '0.00', '#value!': '0.00',
        'apr_00': '0.00', 'n_a_annual_': '0.00', 'n_a_for_annual_species': '0.00', 'n_a_annual_plant_': '0.00',
        'n_a_direct_seeding_': '0.00', 'n_a_short_lived_': '0.00', 'Not monitored yet': '0.00',
        'unknown_probably_0_': '0.00', '0_but_substantial_recruitment_': '0.00', '<5': '5.00', '<10': '10.00',
        '>10': '10.00', '>10_pct_': '10.00', '16_pct_': '16.00', '4_pct_': '4.00', '11_pct_': '11.00',
        'c20': '20.00', 'c25_pct_': '25.00', '17_pct_': '17.00', '<20': '20.00', 'c.20': '20.00',
        '>25': '25.00', '>25_pct_': '25.00', 'c.25': '25.00', '20-40_pct_': '30.00', '20-40': '30.00',
        '29_pct_': '29.00', '~30': '30.00', '>30': '30.00', '>33': '33.00', '~40': '40.00',
        '>40': '40.00', '<50?': '50.00', '<50_pct_': '50.00', '<50_pct_?': '50.00', '50_pct_': '50.00',
        '~50': '50.00', '>50_pct_': '50.00', '>50': '50.00', '52_pct_': '52.00', '60_pct_': '60.00',
        '69_pct_': '69.00', 'unknown_but_probably_>75_pct_': '75.00', '77_pct_': '77.00', '80_pct_': '80.0',
        '80_pct__across_all_sites_and_treatments_': '80.00', '80_pct_ (across all sites and treatments)': '80.00',
        '83_pct_': '83.00', '85_pct_': '85.0', '>90': '90.00', '>95': '95.00', '>95_pct_': '95.00',
        '100_pct_': '100.00', '>100_with_recruitment_': '0.00', 'good': '0.00', 'unknown': '0.00',
        'Unknown': '0.00', 'unknown_number_germinating_not_documented_': '0.00',
        'unknown_how_many_were_planted_in_total': '0.00', 'unknown_number_of_germinants_not_counted_': '0.00',
        'too_early': '0.00', 'plants_in_the_ground_for_<_12_months': '0.00', '<12_months_since_planting': '0.00',
        '<_12_mnths_since_translocation': '0.00', '<12_months_elapsed': '0.00', '_pct_': '0.00',
    }
    cleaned = source.replace(dict_map).str.replace('%', '', regex=False).str.strip()
    parts = cleaned.str.split(':', n=1)
    surviving = pd.to_numeric(parts.str[0].str.strip(), errors="coerce")
    df['plants_surviving_pct'] = surviving.fillna(surviving.mean()).astype("Float64")
    df.drop(columns=['plants_surviving__pct__'], inplace=True)

if 'reproductive_flowering_fruiting_y_n_' in df.columns:
    dict_map = {'y': 'yes', 'n': 'no', 't': 'unk', 'p': 'unk', '<': 'unk', 'u': 'unk', 'unk': 'unk'}
    df['flowering_fruiting'] = df['reproductive_flowering_fruiting_y_n_'].astype('string').str[0].str.strip().replace(dict_map).fillna('missing')
    df.drop(columns=['reproductive_flowering_fruiting_y_n_'], inplace=True)

if 'second_generation_y_n_' in df.columns:
    dict_map = {
        'y': 'yes', 'y_': 'yes', 'yes': 'yes', 'y_4': 'yes', 'y_f': 'yes', 'y_i': 'yes', 'y_r': 'yes', 'y_v': 'yes',
        'n': 'no', 'n_': 'no', '0': 'no', '0.0': 'no', 'too': 'too_soon', 'pla': 'too_soon', '<12': 'too_soon',
        '': 'missing', 'not': 'missing', 'u': 'missing', 'unk': 'missing'
    }
    df['second_gen'] = df['second_generation_y_n_'].astype('string').str[0:3].str.strip().replace(dict_map).fillna('missing')
    df.drop(columns=['second_generation_y_n_'], inplace=True)

if 'number_of_plants_surviving' in df.columns:
    dict_map = {
        "na": "", "+": "999", "<12_months_elapsed": "999", ">100": "999", "n_a_short_lived_": "999",
        "34_30_just_planted_july2017_": "999", "good_numbers_not_counted_": "999", "<12_months_since_planting": "999",
        "<_12_mnths_since_translocation": "999", "present": "999", "40_40_just_planted_july2017_": "999", ">20": "999",
        "not_monitored_yet": "", "not_detailed": "999", ">50": "999", "19_+_12_planted_in_july_2016": "999",
        "105_+_suckers": "999", "17_+_suckers": "999", "2_+_suckers": "999", "3_+_suckers": "999",
        "plants_in_the_ground_for_<_12_months": "999", ">200": "999", "unknown_probably_0_": "",
        "0_non_viable_seed_was_used_": "0.00", "c.500": "999", "5_patches": "999", ">600": "999",
        ">400": "999", "c.1000": "999", "c.225": "999", "c.15": "999", "4000+": "999", "20_30": "999",
        ">25": "999", ">2000": "999", "all_plants_surviving": "999", "pending": "", "40_50": "999",
        "unknown": "", "c.100": "999", "<200": "999", "300_250_just_planted_july2017_": "999",
        "hundreds": "999", "37_30_just_planted_july2017_": "999", "48_40_just_planted_july2017_": "999",
    }
    cleaned = df['number_of_plants_surviving'].astype("string").str.strip().replace(dict_map)
    df['plants_surviving'] = pd.to_numeric(cleaned, errors="coerce").fillna(0.0).gt(0).astype(int)
    df.drop(columns=['number_of_plants_surviving'], inplace=True)

if 'no._second_generation_plants' in df.columns:
    df = df.drop(columns='no._second_generation_plants', errors='ignore')


# # 6.0 - PREPARE SPLIT DATASETS

# ## 6.1 - REMOVE MULTI-CLASS RECORDS

exclude_features_list   = ['translocation', 'burnt']
exclude_values          = r'introduction_augmentation_|introduction_reintroduction|planned_wildfire'

for feature in exclude_features_list:
    if feature in df.columns:
        df = df.loc[
            ~df[feature].astype(str).str.contains(
                exclude_values, 
                na = False
            )
        ].copy()
        print(f"'{feature}' cleaned")


# ## 6.2 - MODEL TRAINING

target_column          = 'plants_surviving'
model_seed             = 123456789

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

df_train               = df_train.reset_index(drop=True)
df_val                 = df_val.reset_index(drop=True)
df_test                = df_test.reset_index(drop=True)

y_train                = (df_train[target_column]).astype('int').values
y_val                  = (df_val[target_column]).astype('int').values
y_test                 = (df_test[target_column]).astype('int').values

for in_df in (df_train, df_val, df_test):
    drop_cols = [c for c in [target_column, 'plants_surviving_pct'] if c in in_df.columns]
    in_df.drop(columns=drop_cols, inplace=True)

train_dicts            = df_train.fillna(0).to_dict(orient='records')
val_dicts              = df_val.fillna(0).to_dict(orient='records')
test_dicts             = df_test.fillna(0).to_dict(orient='records')

dv                     = DictVectorizer(sparse=True)

X_train                = dv.fit_transform(train_dicts)
X_val                  = dv.transform(val_dicts)
X_test                 = dv.transform(test_dicts)


# ### 6.2.1 - LOGISTIC REGRESSION GRID SEARCH & TUNING

cv                     = StratifiedKFold(
                            n_splits     = 5, 
                            shuffle      = True, 
                            random_state = 42
                        )

lr_pipeline            = Pipeline(
                            [
                                ("scale",   StandardScaler(with_mean=False)),
                                ("model",   LogisticRegression(
                                                solver       = "lbfgs",
                                                max_iter     = 5000,
                                                random_state = 42,
                                            )
                                ),
                            ]
                        )

param_grid             = { "model__C": [0.01, 0.1, 1, 10, 100] }

lr_search              = GridSearchCV(
                            estimator  = lr_pipeline,
                            param_grid = param_grid,
                            scoring    = "roc_auc",
                            cv         = cv,
                            n_jobs     = -1,
                            refit      = True,
                            verbose    = 1,
                        )

lr_search.fit(X_train, y_train)

best_C = lr_search.best_params_["model__C"]
print(f"Best C value: {best_C}")
print("Best logistic regression parameters:", lr_search.best_params_)
print(f"Best CV ROC AUC: {lr_search.best_score_:.3f}")

model_final_lr = Pipeline(
    [
        ("scale", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(C=best_C, solver="lbfgs", max_iter=5000, random_state=42))
    ]
)
model_final_lr.fit(X_train, y_train)


# ### 6.2.2 - DECISION TREE SEARCH & TUNING

tree_params = {
    "max_depth": 5,
    "min_samples_split": 10,
    "min_samples_leaf": 2
}

baseline_tree = DecisionTreeClassifier(
    **tree_params,
    random_state = model_seed
)
baseline_tree.fit(X_train, y_train)

X_val_for_importance = X_val.toarray() if hasattr(X_val, "toarray") else X_val

importance_result = permutation_importance(
                        estimator    = baseline_tree,
                        X            = X_val_for_importance,
                        y            = y_val,
                        scoring      = "accuracy",
                        n_repeats    = 20,
                        random_state = model_seed,
                        n_jobs       = -1
                    )

feature_names    = list(X_val.columns) if hasattr(X_val, "columns") else [f"feature_{i}" for i in range(X_val.shape[1])]

importance       = pd.DataFrame(
                        {
                            "feature_index": np.arange(X_val.shape[1]),
                            "feature": feature_names,
                            "importance_mean": importance_result.importances_mean,
                            "importance_std": importance_result.importances_std,
                        }
                    ).sort_values("importance_mean")

low_mean_cutoff  = importance["importance_mean"].quantile(0.10)
low_std_cutoff   = importance["importance_std"].median()

candidate_mask   = (
                        (importance["importance_mean"] <= low_mean_cutoff)
                        & (importance["importance_std"] <= low_std_cutoff)
                    )

features_to_drop = importance.loc[candidate_mask, "feature"].tolist()
drop_names       = set(features_to_drop)

kept_names       = [
                    name for name in dv.get_feature_names_out()
                    if name not in drop_names
                   ]

simplified_dv = DictVectorizer()
simplified_dv.fit(train_dicts)
simplified_dv.vocabulary_ = {name: i for i, name in enumerate(kept_names)}

X_train_reduced = simplified_dv.transform(train_dicts)
X_val_reduced   = simplified_dv.transform(val_dicts)
X_test_reduced  = simplified_dv.transform(test_dicts)

model_simplified_tree   = DecisionTreeClassifier(
                             **tree_params,
                             random_state = model_seed
                          )
model_simplified_tree.fit(X_train_reduced, y_train)


# ### 6.2.3 - MODEL TUNING WITH 'xgboost'

cv               = StratifiedKFold(
                        n_splits     = 5, 
                        shuffle      = True, 
                        random_state = model_seed
                    )

xgb              = XGBClassifier(
                       objective    = "binary:logistic",
                       eval_metric  = "logloss",
                       random_state = model_seed,
                       n_jobs       = 1
                    )

param_grid       = {
                        "learning_rate": [0.01, 0.03, 0.1],
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

xgb_search.fit(X_train_reduced, y_train)

final_model      = xgb_search.best_estimator_

joblib.dump(final_model, "final_xgb_model.joblib")
print("Final model saved as 'final_xgb_model.joblib'.")


# # 7.0 - PREPARE BINARY & EXPORT PIPELINE

output_file = f'model_C={best_C}.bin'

model_pipeline   = Pipeline(
                        [
                            ('vectorizer', simplified_dv),
                            ('model',      final_model)
                        ]
                    )

with open(output_file, 'wb') as f_out:
    pickle.dump(model_pipeline, f_out)

print(f"Model pipeline binary saved to '{output_file}'")

# TEST PREDICTION SETUP WITH PIPELINE
row              = X_test.getrow(10)
feature_names    = simplified_dv.get_feature_names_out()

X_test_datapoint = {
    feature_names[i]: float(value)
    for i, value in zip(row.indices, row.data)
}

with Path(output_file).open("rb") as f_in:
    pipeline = pickle.load(f_in)

probability = pipeline.predict_proba([X_test_datapoint])[0, 1]
prediction = int(probability >= 0.5)

print("prediction\t", prediction)
print("probability\t", probability)
