"""
Funciones de modelado para la predicción de mora en créditos.

Corresponde a la Actividad Práctica II - fase de Modelado.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

COLUMNA_OBJETIVO = "falta_pago"
COLUMNAS_A_EXCLUIR = ["id_cliente"]


def dividir_datos(df: pd.DataFrame, test_size: float = 0.10, val_size: float = 0.22, random_state: int = 42):
    """Divide el dataset en conjuntos de entrenamiento, validación y test,
    con estratificación sobre la variable objetivo.
    """
    features_X = df.drop(columns=[COLUMNA_OBJETIVO] + COLUMNAS_A_EXCLUIR, errors="ignore")
    labels_y = df[COLUMNA_OBJETIVO]

    X_temp, X_test, y_temp, y_test = train_test_split(
        features_X, labels_y, test_size=test_size, random_state=random_state, stratify=labels_y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def construir_preprocesador(features_X: pd.DataFrame) -> ColumnTransformer:
    """preprocesador (escalado de numéricas + codificación de
    categóricas) a partir de los tipos de datos detectados."""
    num_cols = features_X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = features_X.select_dtypes(include=["object", "category"]).columns.tolist()

    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def obtener_modelos_candidatos() -> dict:
    """Retorna el diccionario de modelos candidatos a evaluar."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "LinearSVC": LinearSVC(max_iter=5000),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
    }


def evaluar_modelos_candidatos(X_train, y_train, preprocessor, cv: int = 5) -> pd.DataFrame:
    """Evalua cada modelo candidato con validación cruzada y retorna un
    DataFrame ordenado por accuracy promedio descendente."""
    resultados = []
    for nombre, modelo in obtener_modelos_candidatos().items():
        pipeline = Pipeline([("prep", preprocessor), ("model", modelo)])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        resultados.append({
            "modelo": nombre,
            "accuracy_media": scores.mean(),
            "accuracy_std": scores.std(),
        })
    return pd.DataFrame(resultados).sort_values("accuracy_media", ascending=False).reset_index(drop=True)


def entrenar_modelo_final(X_train, y_train, preprocessor, nombre_modelo: str) -> Pipeline:
    """Entrena el pipeline completo (preprocesamiento + modelo) para el
    modelo seleccionado como mejor candidato.
    """
    modelo = obtener_modelos_candidatos()[nombre_modelo]
    if nombre_modelo == "LinearSVC":
        modelo = CalibratedClassifierCV(modelo, cv=5)
    pipeline_final = Pipeline([("prep", preprocessor), ("model", modelo)])
    pipeline_final.fit(X_train, y_train)
    return pipeline_final