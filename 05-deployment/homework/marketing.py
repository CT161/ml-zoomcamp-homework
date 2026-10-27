# from fastapi import FastAPI
import uvicorn
import requests

url = "http://localhost:9696/predict"

customer = {
  "gender": "male",
  "seniorcitizen": 0,
  "partner": "no",
  "dependents": "yes",
  "phoneservice": "no",
  "multiplelines": "no_phone_service",
  "internetservice": "dsl",
  "onlinesecurity": "no",
  "onlinebackup": "yes",
  "deviceprotection": "no",
  "techsupport": "no",
  "streamingtv": "no",
  "streamingmovies": "no",
  "contract": "month-to-month",
  "paperlessbilling": "yes",
  "paymentmethod": "electronic_check",
  "tenure": 1,
  "monthlycharges": 29.85,
  "totalcharges": 129.85
}

response = requests.post(url, json=customer)
churn = response.json()

print("response: ", churn)

if churn['churn_probability'] >= 0.5:
    print("send email with promo")
else:
        print("do not do anythin")

    
# app = FastAPI(title="churn-prediction")

# '@app' is an endpoint, meaning a place you can get to through a url

# '@app.get' tells the api what to do when it receives a 'get' request
# @app.get("/ping")
# def ping():
#    return "PONG"
