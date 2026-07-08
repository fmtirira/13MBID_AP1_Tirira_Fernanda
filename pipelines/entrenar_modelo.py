"""
Script de generación y registro de modelos con MLflow.

Corresponde a la Actividad 1 (Generación de modelos del escenario) de la
Actividad Práctica II.

Ejecución:
    python pipelines/entrenar_modelo.py
"""

import os
import sys
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.modelado import (
    dividir_datos,
    construir_preprocesador,
    evaluar_modelos_candidatos,
    entrenar_modelo_final,
)

RUTA_BASE = os.path.join(os.path.dirname(__file__), "..")
RUTA_DATOS = os.path.join(RUTA_BASE, "data", "processed", "datos_integrados.csv")
RUTA_MLFLOW_DB = os.path.join(RUTA_BASE, "mlflow.db")
RUTA_MODELO = os.path.join(RUTA_BASE, "models", "modelo_prediccion_mora.pkl")


def calcular_metricas(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="Y"),
        "recall": recall_score(y_true, y_pred, pos_label="Y"),
        "f1": f1_score(y_true, y_pred, pos_label="Y"),
    }


def ejecutar_entrenamiento():
    mlflow.set_tracking_uri(f"sqlite:///{RUTA_MLFLOW_DB}")
    mlflow.set_experiment("13MBID_prediccion_mora_creditos")

    print("=" * 70)
    print("GENERACIÓN DE MODELOS - Actividad Práctica II")
    print("=" * 70)

    df = pd.read_csv(RUTA_DATOS)
    print(f"\n[1/4] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(df)
    print(f"[2/4] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    preprocessor = construir_preprocesador(X_train)

    print("\n[3/4] Evaluando modelos candidatos con validación cruzada...")
    df_resultados = evaluar_modelos_candidatos(X_train, y_train, preprocessor)
    print(df_resultados.to_string(index=False))

    with mlflow.start_run(run_name="comparacion_modelos_candidatos"):
        for _, fila in df_resultados.iterrows():
            with mlflow.start_run(run_name=fila["modelo"], nested=True):
                mlflow.log_param("modelo", fila["modelo"])
                mlflow.log_metric("accuracy_cv_media", fila["accuracy_media"])
                mlflow.log_metric("accuracy_cv_std", fila["accuracy_std"])

    mejor_modelo_nombre = df_resultados.iloc[0]["modelo"]
    print(f"\n[4/4] Mejor modelo candidato: {mejor_modelo_nombre}")

    with mlflow.start_run(run_name=f"modelo_final_{mejor_modelo_nombre}") as run:
        pipeline_final = entrenar_modelo_final(X_train, y_train, preprocessor, mejor_modelo_nombre)

        metricas_val = calcular_metricas(y_val, pipeline_final.predict(X_val))
        metricas_test = calcular_metricas(y_test, pipeline_final.predict(X_test))

        mlflow.log_param("modelo", mejor_modelo_nombre)
        mlflow.log_param("filas_entrenamiento", X_train.shape[0])
        for k, v in metricas_val.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in metricas_test.items():
            mlflow.log_metric(f"test_{k}", v)

        try:
            mlflow.sklearn.log_model(
                pipeline_final, name="modelo",
                registered_model_name="modelo_prediccion_mora",
            )
        except mlflow.exceptions.MlflowException as e:
            print(f"\n[aviso] No se pudo registrar el artefacto del modelo en MLflow ({e}).")
            print("[aviso] El modelo igualmente se guarda en disco con joblib para su uso en la API.")

        os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
        joblib.dump(pipeline_final, RUTA_MODELO)

        print("\nMétricas en validación:", metricas_val)
        print("Métricas en test:", metricas_test)
        print(f"\nRun ID: {run.info.run_id}")

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 70)


if __name__ == "__main__":
    ejecutar_entrenamiento()