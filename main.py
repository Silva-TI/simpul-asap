
import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="SIMPUL ASAP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

model = joblib.load("model_simpul_asap.pkl")

class InputPrediksi(BaseModel):
    suhu: float
    kelembapan: float = Field(ge=0, le=100)
    curah_hujan: float = Field(ge=0)
    kecepatan_angin: float = Field(ge=0)
    hotspot_200km: int = Field(ge=0)

@app.get("/")
def status():
    return {"status": "aktif", "aplikasi": "SIMPUL ASAP"}

@app.post("/predict")
def prediksi(data: InputPrediksi):
    try:
        fitur = pd.DataFrame([{
            "suhu": data.suhu,
            "kelembapan": data.kelembapan,
            "curah_hujan": data.curah_hujan,
            "kecepatan_angin": data.kecepatan_angin,
            "hotspot_200km": data.hotspot_200km,
        }])

        nilai = float(model.predict(fitur)[0])

        return {"prediksi_pm25": round(nilai, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
