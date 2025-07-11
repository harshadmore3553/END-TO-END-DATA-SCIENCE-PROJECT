
import joblib
from app.schema import PatientData, PredictionResponse

def load_model():
    return joblib.load("model/hospital_model.pkl")

def predict_readmission(model, data: PatientData) -> PredictionResponse:
    input_data = [[
        data.age,
        1 if data.gender == "M" else 0,
        data.days_in_hospital,
        data.lab_procedures,
        data.medications,
        data.visits_last_year
    ]]
    prediction = model.predict(input_data)[0]
    return PredictionResponse(readmitted=bool(prediction))
