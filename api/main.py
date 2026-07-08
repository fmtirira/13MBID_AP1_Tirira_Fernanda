"""
API REST para servir el modelo de predicción de mora en créditos.
Corresponde a la Actividad 3 (Despliegue de la solución - parte 1) de AP2.

"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

RUTA_MODELO = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_prediccion_mora.pkl")

app = FastAPI(
    title="API de Predicción de Mora en Créditos",
    description="Actividad Práctica II - 13MBID",
    version="1.0.0",
)

modelo = None


@app.on_event("startup")
def cargar_modelo():
    global modelo
    modelo = joblib.load(RUTA_MODELO)


class SolicitudPrediccion(BaseModel):
    importe_solicitado: float
    duracion_credito: float
    situacion_vivienda: str
    ingresos: float
    objetivo_credito: str
    pct_ingreso: float
    tasa_interes: float
    antiguedad_cliente: float
    estado_cliente: str
    gastos_ult_12m: float
    genero: str
    limite_credito_tc: float
    nivel_educativo: str
    operaciones_ult_12m: float
    personas_a_cargo: float
    estado_civil_N: str
    estado_credito_N: str
    antiguedad_empleado_N: str
    edad_N: str


class RespuestaPrediccion(BaseModel):
    prediccion: str
    probabilidad_mora: float


@app.get("/")
def raiz():
    return {"mensaje": "API de predicción de mora en créditos - 13MBID", "estado": "activo"}


@app.post("/predecir", response_model=RespuestaPrediccion)
def predecir(solicitud: SolicitudPrediccion):
    datos = pd.DataFrame([solicitud.dict()])
    try:
        prediccion = modelo.predict(datos)[0]
        probabilidad = modelo.predict_proba(datos)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {e}")

    return RespuestaPrediccion(
        prediccion="Mora" if prediccion == "Y" else "Sin mora",
        probabilidad_mora=round(float(probabilidad), 4),
    )