
from pydantic import BaseModel

class PatientData(BaseModel):
    age: int
    gender: str
    days_in_hospital: int
    lab_procedures: int
    medications: int
    visits_last_year: int

class PredictionResponse(BaseModel):
    readmitted: bool
