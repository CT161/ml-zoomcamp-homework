from fastapi import FastAPI
import uvicorn

app = FastAPI(title="ping")

# '@app' is an endpoint, meaning a place you can get to through a url

# '@app.get' tells the api what to do when it receives a 'get' request
@app.get("/ping")
def ping():
    return "PONG"

if __name__ == "__main__":
    # we want 'uvicorn' to run this application
    # this should be a localhost service (host="0.0.0.0")
    # application should run on port 9696
    uvicorn.run(app, host="0.0.0.0", port=9696)
