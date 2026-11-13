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
BASE_DIR         = Path(__file__).resolve().parent
MODEL_PATH       = BASE_DIR / "model_C=0.01.bin"

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
    # NOTE: THERE ARE MORE DATA POINTS IN THE RAW DATA THAN THE MODEL USES
    model_config = ConfigDict(extra="ignore")

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