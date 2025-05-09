from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Pipeline modeli yükle
pipeline = joblib.load('pipeline_model.pkl')

# FastAPI uygulaması
app = FastAPI()

# İstek için veri modeli
class Applicant(BaseModel):
    tecrube_yili: float
    teknik_puan: float

@app.post("/predict")
def predict(applicant: Applicant):
    X = np.array([[applicant.tecrube_yili, applicant.teknik_puan]])
    prediction = pipeline.predict(X)
    
    sonuc = "Aday işe alınır." if prediction[0] == 0 else "Aday işe alınmaz."
    
    return {"prediction": int(prediction[0]), "sonuc": sonuc}