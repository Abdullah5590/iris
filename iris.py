import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import pickle

# Create FastAPI app
app = FastAPI()

# Load trained model once
with open("model.pkl", "rb") as pickle_in:
    abd = pickle.load(pickle_in)

# Root route -> redirect to docs
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

# Prediction route
@app.post("/predict")
def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    prediction = abd.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    label_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return {"prediction": label_map.get(prediction[0], "Unknown")}

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
