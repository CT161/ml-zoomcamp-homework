# SYSTEM IMPORTS
import pickle  # NEEDED TO READ AND WRITE WITH BINARY FILES
import uvicorn # NEEDED TO HOST A WEB SERVICE

# PACKAGE IMPORTS



# NAMED FUNCTIONS FROM PACKAGES
from typing import Literal, Dict, Any
from typing import Dict, Any  # NEEDED FOR WEB API VALIDATION
from pydantic import BaseModel, Field
from fastapi import FastAPI   # NEEDED TO CREATE THE WEB SERVICE


# VALIDATION CLASS
class Customer(BaseModel):
    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)

# CREATE A PREDICTION RESPONSE CLASS
class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool

# CRATE A FastAPI INSTANCE
app = FastAPI(title="customer-churn-prediction")

# CREATE A PIPELINE
with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# CREATE A PREDICTION FUNCTION
# THIS WILL NOt wORK RELIABLY IF 'customer' IS NOT JSON
def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    # first parameter 0 means first row
    # second parameter 1 means get the probability for 1 (1=churn 0=no churn)
    return float(result)

# CREATE AN ENDPOINT
@app.post("/predict")

# CREATE A PREDICTION FUNCTION
def predict(customer: Customer) -> PredictResponse:
    # TO FIX THE ISSUE OF 'predict_single(customer)' REQUIRING JSON STRING IS TO USE A TYPE HINT IN THE FUNCTION DEFINITION
    prob = predict_single(customer.model_dump())

    # ORIGINAL FUNCTION RETURN
    #return PredictResponse(
    #    churn_probability=prob,
    #    churn=prob >= 0.5
    #)

    # NEW FUNCTION RETURN
    return{
        "churn_probability" : prob,
        "churn" : bool(prob >= 0.5)
    }

# SET ENDPOINT ENGINE, HOST AND PORT
if __name__ == "__main__":
    # we want 'uvicorn' to run this application
    # this should be a localhost service (host="0.0.0.0")
    # application should run on port 9696
    uvicorn.run(app, host="0.0.0.0", port=9696)




