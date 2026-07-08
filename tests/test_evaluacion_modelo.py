"""
Casos de prueba automatizados para la evaluación del modelo de predicción.
Actividad 2 de la Actividad Práctica II.

Ejecución:
    pytest tests/test_evaluacion_modelo.py -v
"""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.modelado import dividir_datos, construir_preprocesador, entrenar_modelo_final
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

RUTA_DATOS = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "datos_integrados.csv")

UMBRAL_ACCURACY_MINIMO = 0.85
UMBRAL_PRECISION_MINIMA = 0.60
UMBRAL_RECALL_MINIMO = 0.45


@pytest.fixture(scope="module")
def datos_divididos():
    df = pd.read_csv(RUTA_DATOS)
    return dividir_datos(df)


@pytest.fixture(scope="module")
def modelo_entrenado(datos_divididos):
    X_train, X_val, X_test, y_train, y_val, y_test = datos_divididos
    preprocessor = construir_preprocesador(X_train)
    pipeline = entrenar_modelo_final(X_train, y_train, preprocessor, "LinearSVC")
    return pipeline


def test_modelo_supera_al_baseline(datos_divididos, modelo_entrenado):
    """El modelo entrenado debe superar a un clasificador base (DummyClassifier)."""
    X_train, X_val, X_test, y_train, y_val, y_test = datos_divididos
    preprocessor = construir_preprocesador(X_train)
    baseline = Pipeline([("prep", preprocessor), ("model", DummyClassifier(strategy="most_frequent"))])
    baseline.fit(X_train, y_train)

    acc_baseline = accuracy_score(y_test, baseline.predict(X_test))
    acc_modelo = accuracy_score(y_test, modelo_entrenado.predict(X_test))

    assert acc_modelo > acc_baseline


def test_accuracy_supera_umbral_minimo(datos_divididos, modelo_entrenado):
    """La accuracy en test debe superar el umbral mínimo aceptable para el negocio."""
    X_train, X_val, X_test, y_train, y_val, y_test = datos_divididos
    y_pred = modelo_entrenado.predict(X_test)
    assert accuracy_score(y_test, y_pred) >= UMBRAL_ACCURACY_MINIMO


def test_precision_supera_umbral_minimo(datos_divididos, modelo_entrenado):
    """La precision en test debe superar el umbral mínimo aceptable."""
    X_train, X_val, X_test, y_train, y_val, y_test = datos_divididos
    y_pred = modelo_entrenado.predict(X_test)
    assert precision_score(y_test, y_pred, pos_label="Y") >= UMBRAL_PRECISION_MINIMA


def test_recall_supera_umbral_minimo(datos_divididos, modelo_entrenado):
    """El recall en test debe superar el umbral mínimo aceptable."""
    X_train, X_val, X_test, y_train, y_val, y_test = datos_divididos
    y_pred = modelo_entrenado.predict(X_test)
    assert recall_score(y_test, y_pred, pos_label="Y") >= UMBRAL_RECALL_MINIMO


def test_modelo_predice_ambas_clases(datos_divididos, modelo_entrenado):
    """El modelo no debe colapsar a predecir siempre la misma clase."""
    X_train, X_val, X_test, y_train, y_val, y_test = datos_divididos
    y_pred = modelo_entrenado.predict(X_test)
    assert len(set(y_pred)) > 1