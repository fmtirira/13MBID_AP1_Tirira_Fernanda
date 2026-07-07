"""
Pipeline de preparación de datos.

Corresponde al entregable de la Actividad 5: un script que permite
reproducir de forma automática las operaciones de limpieza,
transformación e integración de los datos, independientemente de las
libretas (notebooks) usadas originalmente en la experimentación del MVP.

Ejecución:
    python pipelines/pipeline_datos.py

Entrada:
    data/raw/datos_creditos.csv
    data/raw/datos_tarjetas.csv

Salida:
    data/processed/datos_integrados.csv
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preparacion_datos import (
    limpiar_creditos,
    limpiar_tarjetas,
    seleccionar_columnas,
    transformar_integrado,
    integrar_datasets,
    formatear_dataset_final,
)
from src.calidad_datos import generar_reporte_calidad

RUTA_BASE = os.path.join(os.path.dirname(__file__), "..")
RUTA_CREDITOS_RAW = os.path.join(RUTA_BASE, "data", "raw", "datos_creditos.csv")
RUTA_TARJETAS_RAW = os.path.join(RUTA_BASE, "data", "raw", "datos_tarjetas.csv")
RUTA_SALIDA = os.path.join(RUTA_BASE, "data", "processed", "datos_integrados.csv")


def ejecutar_pipeline():
    print("=" * 70)
    print("PIPELINE DE PREPARACIÓN DE DATOS - Actividad Práctica I")
    print("=" * 70)

    # 1. Carga de datos crudos
    print("\n[1/6] Cargando datos de origen...")
    creditos = pd.read_csv(RUTA_CREDITOS_RAW, sep=";")
    tarjetas = pd.read_csv(RUTA_TARJETAS_RAW, sep=";")
    print(f"      datos_creditos.csv: {creditos.shape[0]} filas, {creditos.shape[1]} columnas")
    print(f"      datos_tarjetas.csv: {tarjetas.shape[0]} filas, {tarjetas.shape[1]} columnas")

    # 2. Reporte de calidad (Actividad 4) previo a limpiar, a modo de diagnóstico
    print("\n[2/6] Evaluando calidad de los datos de origen...")
    reporte = generar_reporte_calidad(creditos, tarjetas)
    for r in reporte:
        estado = "OK" if r["cumple"] else "NC (no cumplimiento)"
        print(f"      [{r['dimension']:12s}] {r['chequeo']:38s} -> {estado}")

    # 3. Limpieza
    print("\n[3/6] Limpiando datasets...")
    creditos_limpio = limpiar_creditos(creditos)
    tarjetas_limpio = limpiar_tarjetas(tarjetas)

    # 4. Selección de columnas
    print("\n[4/6] Seleccionando columnas relevantes...")
    tarjetas_limpio = seleccionar_columnas(tarjetas_limpio)

    # 5. Integración y transformación (el MVP transforma post-integración,
    #    ya que combina atributos de ambas fuentes)
    print("\n[5/6] Integrando y transformando datasets...")
    dataset_integrado = integrar_datasets(creditos_limpio, tarjetas_limpio)
    dataset_final = transformar_integrado(dataset_integrado)
    dataset_final = formatear_dataset_final(dataset_final)

    # 6. Persistencia del resultado
    print("\n[6/6] Guardando dataset final...")
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    dataset_final.to_csv(RUTA_SALIDA, index=False)
    print(f"      Guardado en: {RUTA_SALIDA}")
    print(f"      Filas: {dataset_final.shape[0]} | Columnas: {dataset_final.shape[1]}")

    print("\n" + "=" * 70)
    print("PIPELINE FINALIZADO CORRECTAMENTE")
    print("=" * 70)

    return dataset_final


if __name__ == "__main__":
    ejecutar_pipeline()
