"""
Casos de prueba automatizados para la evaluación de calidad de datos.

Corresponde a la Actividad 4 (Comprensión de los datos) de la Actividad
Práctica I. Se automatiza la evaluación de calidad tomando como referencia
el Capítulo 13 del DMBoK y la norma ISO/IEC 25012, cubriendo las
dimensiones de Exactitud, Completitud y Consistencia.

Ejecución:
    pytest tests/ -v
"""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.calidad_datos import (
    verificar_formato,
    verificar_rango,
    verificar_valores_permitidos,
    verificar_nulos,
    verificar_unicidad,
    verificar_integridad_referencial,
)

RUTA_CREDITOS = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "datos_creditos.csv")
RUTA_TARJETAS = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "datos_tarjetas.csv")


@pytest.fixture(scope="module")
def creditos():
    return pd.read_csv(RUTA_CREDITOS, sep=";")


@pytest.fixture(scope="module")
def tarjetas():
    return pd.read_csv(RUTA_TARJETAS, sep=";")


# ---------------------------------------------------------------------------
# EXACTITUD
# ---------------------------------------------------------------------------

def test_formato_dataset_creditos(creditos):
    columnas_esperadas = {
        "id_cliente": "float",
        "edad": "int",
        "importe_solicitado": "int",
        "falta_pago": "object",
    }
    resultado = verificar_formato(creditos, columnas_esperadas)
    assert resultado["cumple"], f"Columnas/tipos incorrectos: {resultado}"


def test_edad_dentro_de_rango(creditos):
    """Tolerancia 1%: se detectaron 4 casos (0.04%) de edades fuera de 18-90."""
    resultado = verificar_rango(creditos, "edad", 18, 90, tolerancia_pct=1.0)
    assert resultado["cumple"], f"Demasiados valores fuera de rango: {resultado}"


def test_antiguedad_empleado_dentro_de_rango(creditos):
    """Tolerancia 1%: se detectaron 2 casos (0.02%) fuera de 0-60 años."""
    resultado = verificar_rango(creditos, "antiguedad_empleado", 0, 60, tolerancia_pct=1.0)
    assert resultado["cumple"], f"Demasiados valores fuera de rango: {resultado}"


def test_situacion_vivienda_valores_validos(creditos):
    resultado = verificar_valores_permitidos(
        creditos, "situacion_vivienda", {"ALQUILER", "PROPIA", "HIPOTECA", "OTROS"}
    )
    assert resultado["cumple"], f"Valores no esperados: {resultado['valores_invalidos']}"


def test_falta_pago_valores_validos(creditos):
    resultado = verificar_valores_permitidos(creditos, "falta_pago", {"Y", "N"})
    assert resultado["cumple"], f"Valores no esperados: {resultado['valores_invalidos']}"


# ---------------------------------------------------------------------------
# COMPLETITUD
# ---------------------------------------------------------------------------

def test_nulos_antiguedad_empleado_dentro_de_tolerancia(creditos):
    """3.33% de nulos, dentro de la tolerancia definida de 5%."""
    resultado = verificar_nulos(creditos, "antiguedad_empleado", tolerancia_pct=5.0)
    assert resultado["cumple"], f"Supera la tolerancia de nulos: {resultado}"


@pytest.mark.xfail(reason=(
    "Hallazgo real de calidad: 'tasa_interes' tiene 9.01% de nulos, "
    "supera la tolerancia definida (5%). Requiere estrategia de imputación "
    "antes de la fase de modelado (ver Actividad 5)."
))
def test_nulos_tasa_interes_dentro_de_tolerancia(creditos):
    resultado = verificar_nulos(creditos, "tasa_interes", tolerancia_pct=5.0)
    assert resultado["cumple"], f"Supera la tolerancia de nulos: {resultado}"


# ---------------------------------------------------------------------------
# CONSISTENCIA
# ---------------------------------------------------------------------------

def test_unicidad_id_cliente_creditos(creditos):
    resultado = verificar_unicidad(creditos, "id_cliente")
    assert resultado["cumple"], f"Existen id_cliente duplicados: {resultado}"


def test_unicidad_id_cliente_tarjetas(tarjetas):
    resultado = verificar_unicidad(tarjetas, "id_cliente")
    assert resultado["cumple"], f"Existen id_cliente duplicados: {resultado}"


def test_integridad_referencial_tarjetas_en_creditos(creditos, tarjetas):
    resultado = verificar_integridad_referencial(tarjetas, creditos, "id_cliente")
    assert resultado["cumple"], f"Existen id_cliente huérfanos: {resultado}"


def test_integridad_referencial_creditos_en_tarjetas(creditos, tarjetas):
    resultado = verificar_integridad_referencial(creditos, tarjetas, "id_cliente")
    assert resultado["cumple"], f"Existen id_cliente huérfanos: {resultado}"
