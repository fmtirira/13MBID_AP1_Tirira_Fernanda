"""
Pipeline que integra el entrenamiento y la evaluación del modelo.
Entregable de la Actividad 2 (Evaluación de los modelos generados) de AP2.

"""

import subprocess
import sys
import os

RUTA_BASE = os.path.join(os.path.dirname(__file__), "..")


def ejecutar_pipeline():
    print("=" * 70)
    print("PIPELINE: Entrenamiento + Evaluación del modelo")
    print("=" * 70)

    print("\n[1/2] Entrenando y registrando modelo en MLflow...")
    resultado_entrenamiento = subprocess.run(
        [sys.executable, os.path.join(RUTA_BASE, "pipelines", "entrenar_modelo.py")],
        cwd=RUTA_BASE,
    )
    if resultado_entrenamiento.returncode != 0:
        print("\n[ERROR] Falló el entrenamiento del modelo. Pipeline detenido.")
        sys.exit(1)

    print("\n[2/2] Ejecutando tests de evaluación automatizados...")
    resultado_tests = subprocess.run(
        [sys.executable, "-m", "pytest", os.path.join(RUTA_BASE, "tests", "test_evaluacion_modelo.py"), "-v"],
        cwd=RUTA_BASE,
    )
    if resultado_tests.returncode != 0:
        print("\n[ERROR] El modelo generado no pasó los tests de evaluación. Revisar antes de desplegar.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("PIPELINE FINALIZADO CON ÉXITO - modelo listo para despliegue")
    print("=" * 70)


if __name__ == "__main__":
    ejecutar_pipeline()