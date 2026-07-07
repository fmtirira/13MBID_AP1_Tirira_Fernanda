"""
Script de exploración de datos.

Corresponde a la Actividad 4 (Comprensión de los datos) de la Actividad
Práctica I. Convierte la exploración realizada originalmente en la libreta
"CD01_Visualizacion.ipynb" a un script independiente, reproducible y
apto para integrarse en un flujo de trabajo automatizado (DataOps).

Ejecución:
    python src/exploracion_datos.py

Salidas (carpeta reports/figures/):
    - distribucion_variable_objetivo.png
    - distribucion_variables_categoricas.png
    - matriz_correlacion.png
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin interfaz gráfica, apto para scripts/CI
import matplotlib.pyplot as plt
import seaborn as sns

RUTA_BASE = os.path.join(os.path.dirname(__file__), "..")
RUTA_CREDITOS = os.path.join(RUTA_BASE, "data", "raw", "datos_creditos.csv")
RUTA_TARJETAS = os.path.join(RUTA_BASE, "data", "raw", "datos_tarjetas.csv")
RUTA_SALIDA = os.path.join(RUTA_BASE, "reports", "figures")

sns.set_theme(style="whitegrid", palette="muted")


def graficar_distribucion_objetivo(df: pd.DataFrame):
    """Gráfico de distribución de la variable objetivo (falta_pago)."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x="falta_pago", data=df)
    plt.title("Distribución de la variable objetivo (mora)")
    plt.xlabel("¿Presentó mora el cliente?")
    plt.ylabel("Cantidad de clientes")
    plt.tight_layout()
    ruta = os.path.join(RUTA_SALIDA, "distribucion_variable_objetivo.png")
    plt.savefig(ruta, dpi=100)
    plt.close()
    print(f"[exploracion] guardado: {ruta}")

    porcentajes = df["falta_pago"].value_counts(normalize=True).mul(100).round(2)
    print(f"[exploracion] distribución falta_pago (%):\n{porcentajes}")


def graficar_variables_categoricas(df_creditos: pd.DataFrame, df_tarjetas: pd.DataFrame):
    """Grilla de distribución de las variables categóricas de ambos datasets."""
    cols_creditos = df_creditos.select_dtypes(include=["object"]).columns.drop("falta_pago", errors="ignore")
    cols_tarjetas = df_tarjetas.select_dtypes(include=["object"]).columns
    columnas = [("creditos", c) for c in cols_creditos] + [("tarjetas", c) for c in cols_tarjetas]

    n = len(columnas)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, (origen, col) in enumerate(columnas):
        df = df_creditos if origen == "creditos" else df_tarjetas
        orden = df[col].value_counts().index
        sns.countplot(y=col, data=df, order=orden, ax=axes[i])
        axes[i].set_title(f"{col} ({origen})")

    for j in range(len(columnas), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    ruta = os.path.join(RUTA_SALIDA, "distribucion_variables_categoricas.png")
    plt.savefig(ruta, dpi=100)
    plt.close()
    print(f"[exploracion] guardado: {ruta}")


def graficar_matriz_correlacion(df_creditos: pd.DataFrame, df_tarjetas: pd.DataFrame):
    """Matriz de correlación entre variables numéricas (dataset integrado)."""
    df_integrado = pd.merge(df_creditos, df_tarjetas, on="id_cliente", how="inner")
    numericas = df_integrado.select_dtypes(include=["int64", "float64"])

    plt.figure(figsize=(10, 8))
    sns.heatmap(numericas.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Matriz de correlación - variables numéricas")
    plt.tight_layout()
    ruta = os.path.join(RUTA_SALIDA, "matriz_correlacion.png")
    plt.savefig(ruta, dpi=100)
    plt.close()
    print(f"[exploracion] guardado: {ruta}")


def ejecutar_exploracion():
    os.makedirs(RUTA_SALIDA, exist_ok=True)

    creditos = pd.read_csv(RUTA_CREDITOS, sep=";")
    tarjetas = pd.read_csv(RUTA_TARJETAS, sep=";")

    print("=" * 70)
    print("EXPLORACIÓN DE DATOS - Actividad Práctica I")
    print("=" * 70)

    graficar_distribucion_objetivo(creditos)
    graficar_variables_categoricas(creditos, tarjetas)
    graficar_matriz_correlacion(creditos, tarjetas)

    print("=" * 70)
    print("EXPLORACIÓN FINALIZADA - salidas en reports/figures/")
    print("=" * 70)


if __name__ == "__main__":
    ejecutar_exploracion()
