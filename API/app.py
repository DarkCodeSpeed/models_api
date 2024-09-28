from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model dynamically
model_path = [file for file in os.listdir('models') if file.endswith('.pkl')]
if model_path:
    model = joblib.load(f'models/{model_path[0]}')
else:
    raise FileNotFoundError("Model not found! Make sure the model file is in the 'models' directory.")

# Define the input data schema based on the insurance data
class InsuranceData(BaseModel):
    sex: int  # categorical (0 or 1)
    bmi: float
    children: int
    smoker: int  # categorical (0 or 1)
    region: int  # categorical (0 to 3)
    expenses: float

# Prediction endpoint
@app.post("/predict")
def predict(data: InsuranceData):
    try:
        # Prepare the input data as a NumPy array
        input_data = np.array([[data.sex, data.bmi, data.children, data.smoker, data.region, data.expenses]])
        
        # Predict the age using the model
        prediction = model.predict(input_data)[0].astype(int)
        
        return {"predicted_age": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"message": "Insurance Age Prediction API is running!"}
