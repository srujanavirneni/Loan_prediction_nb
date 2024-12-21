from fastapi import FastAPI
import dill
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

import dill
with open('./rfr_v1.pkl', 'rb') as f:
    reloaded_model = dill.load(f)


class Payload(BaseModel):
    ApplicantIncome: int
    CoapplicantIncome: int
    LoanAmount: int
    Loan_Amount_Term: int
    Credit_History: int
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    Property_Area: str

app = FastAPI()


@app.get("/")
def read_root():
    return {
        "Name": "Srujan Avirneni",
        "Project": "FINAL PROJECT - Loan Prediction",
    }


@app.post("/predict")
def predict(payload: Payload):
    df = pd.DataFrame([payload.model_dump().values()], columns=payload.model_dump().keys())
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']

    y_hat = reloaded_model.predict(df)
    return {"prediction": int(y_hat[0])}