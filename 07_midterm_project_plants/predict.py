import pickle
from   pathlib   import Path
from   typing    import Literal

import uvicorn
from   fastapi   import FastAPI
from   pydantic  import BaseModel, ConfigDict, Field, create_model


# ============================================================
# LOAD THE FITTED DICTVECTORIZER AND MODEL
# ============================================================
MODEL_PATH = Path("model_C=0.1.bin")

with MODEL_PATH.open("rb") as f_in:
    dv, model = pickle.load(f_in)


# ============================================================
# CREATE PYDANTIC INPUT MODEL FROM VECTORIZER FEATURES
#  THIS WORKS BECAUSE ALL FEATURES ARE TRUE/FALSE
# ============================================================
feature_names    = dv.get_feature_names_out().tolist()

PlantFeatures    = create_model(
                        "PlantFeatures",
                        __config__=ConfigDict(extra="forbid"),
                        **{
                            f"feature_{i}": (Literal[0, 1], Field(alias=feature_name))
                            for i, feature_name in enumerate(feature_names)
                        },
                    )


# ============================================================
# DEFINE RESPONSE MODEL AND FASTAPI APP
# ============================================================
class PredictResponse(BaseModel):
    prediction_probability: float
    prediction: int


app = FastAPI(title="plant-survival-prediction")


# ============================================================
# MAKE A PREDICTION
# ============================================================
def predict_single(features: dict) -> tuple[float, int]:
    # Convert the validated feature dictionary to the format expected by dv.
    X            = dv.transform([features])

    probability  = float(model.predict_proba(X)[0, 1])
    prediction   = int(probability >= 0.5)

    return probability, prediction


# ============================================================
# PREDICTION ENDPOINT
# ============================================================
@app.post("/predict", response_model=PredictResponse)
def predict(features: PlantFeatures) -> PredictResponse:
    feature_dict               = features.model_dump(by_alias=True)
    probability, prediction    = predict_single(feature_dict)

    return PredictResponse(
        prediction_probability = probability,
        prediction             = prediction,
    )


# ============================================================
# RUN THE API
# ============================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
