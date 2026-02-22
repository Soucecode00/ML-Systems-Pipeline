from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model.pkl")

# Define request schema
class LoanApplication(BaseModel):
    age: int
    income: float
    savings: float

@app.get("/")
def home():
    return {"message": "Loan Approval Model API is running"}

@app.post("/predict")
def predict(data: LoanApplication):

    input_df = pd.DataFrame(
        [[data.age, data.income, data.savings]],
        columns=["age", "income", "savings"]
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "approved": int(prediction),
        "approval_probability": float(probability)
    }