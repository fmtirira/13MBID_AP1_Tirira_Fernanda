"""
Módulo de preparación de los datos.

Corresponde a la Actividad 5 (Preparación de los datos) de la Actividad
Práctica I. Contiene las funciones de limpieza, selección, transformación
e integración de los datasets de créditos y tarjetas, de forma que
puedan ejecutarse de manera reproducible desde el pipeline
(pipelines/pipeline_datos.py).
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------

def limpiar_creditos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de limpieza sobre el dataset de créditos, replicando
    el criterio ya validado en el MVP (notebook CD02_Procesamiento.ipynb):

    - Elimina filas con "edad" >= 90 (valores no representativos detectados
      en la Actividad 4).
    - Elimina cualquier fila remanente con valores nulos (dropna general),
      criterio adoptado en el MVP en lugar de imputación.
    """
    df = df.copy()
    filas_antes = len(df)

    df = df[df["edad"] < 90]
    df = df.dropna()

    filas_eliminadas = filas_antes - len(df)
    print(f"[limpiar_creditos] filas eliminadas (edad>=90 o con nulos): {filas_eliminadas}")

    return df


def limpiar_tarjetas(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de limpieza sobre el dataset de tarjetas.

    No se detectaron valores nulos ni fuera de rango en este dataset
    (ver Actividad 4), por lo que se retorna una copia sin modificaciones
    a la espera de futuras reglas de negocio.
    """
    return df.copy()


# ---------------------------------------------------------------------------
# Selección de datos (columnas)
# ---------------------------------------------------------------------------

COLUMNAS_A_ELIMINAR_TARJETAS = ["nivel_tarjeta"]  # alta correlación con limite_credito_tc


def seleccionar_columnas(df_tarjetas: pd.DataFrame) -> pd.DataFrame:
    """Elimina atributos que no aportan al modelo (ver Actividad 5:
    Selección de datos)."""
    return df_tarjetas.drop(columns=COLUMNAS_A_ELIMINAR_TARJETAS, errors="ignore")


# ---------------------------------------------------------------------------
# Transformación
# ---------------------------------------------------------------------------

def transformar_integrado(df: pd.DataFrame) -> pd.DataFrame:
    """Genera los atributos derivados definidos en el MVP oficial
    (notebook CD02_Procesamiento.ipynb) sobre el dataset ya integrado,
    dado que combina columnas de ambas fuentes ("estado_civil" viene de
    tarjetas, "estado_credito"/"antiguedad_empleado"/"edad" de créditos).
    """
    df = df.copy()

    mapa_estado_civil = {"CASADO": "C", "SOLTERO": "S", "DESCONOCIDO": "N", "DIVORCIADO": "D"}
    mapa_estado_credito = {0: "P", 1: "C"}

    df["estado_civil_N"] = df["estado_civil"].map(mapa_estado_civil)
    df["estado_credito_N"] = df["estado_credito"].map(mapa_estado_credito)

    df["antiguedad_empleado_N"] = pd.cut(
        df["antiguedad_empleado"],
        bins=[0, 4, 10, 50],
        labels=["menor_5", "5_a_10", "mayor_10"],
        right=False,
    )

    df["edad_N"] = pd.cut(
        df["edad"],
        bins=[0, 24, 50],
        labels=["menor_25", "25_a_30"],
    )

    df = df.drop(columns=["estado_civil", "estado_credito", "antiguedad_empleado", "edad"])

    return df


# ---------------------------------------------------------------------------
# Integración
# ---------------------------------------------------------------------------

def integrar_datasets(creditos: pd.DataFrame, tarjetas: pd.DataFrame) -> pd.DataFrame:
    """Integra (mezcla) los datasets de créditos y tarjetas por id_cliente."""
    return pd.merge(creditos, tarjetas, on="id_cliente", how="inner")


# ---------------------------------------------------------------------------
# Formateo final
# ---------------------------------------------------------------------------

def formatear_dataset_final(df: pd.DataFrame) -> pd.DataFrame:
    """Formateo final del dataset integrado.

    El MVP oficial no aplica transformaciones adicionales sobre
    "falta_pago" ni elimina "id_cliente" en esta etapa (se conservan
    para trazabilidad); se documenta explícitamente la ausencia de
    duplicados como validación final.
    """
    df = df.copy()
    duplicados = df.duplicated().sum()
    print(f"[formatear_dataset_final] filas duplicadas en el dataset integrado: {duplicados}")
    return df
