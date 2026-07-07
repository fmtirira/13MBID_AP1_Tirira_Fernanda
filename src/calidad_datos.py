"""
Módulo de verificación de calidad de datos.

Implementa comprobaciones automatizadas basadas en las dimensiones de
calidad de datos establecidas en el Capítulo 13 del DMBoK y en la norma
ISO/IEC 25012, aplicadas al escenario de créditos otorgados por una
entidad financiera.

Dimensiones cubiertas:
    - Exactitud (formato correcto, valores dentro de rango)
    - Completitud (ausencia de valores nulos)
    - Consistencia (unicidad de claves, integridad referencial)

Cada función retorna un diccionario con el resultado de la verificación
para poder ser reutilizado tanto en los tests automatizados (PyTest)
como en el pipeline de reporte de calidad.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Dimensión: EXACTITUD
# ---------------------------------------------------------------------------

def verificar_formato(df: pd.DataFrame, columnas_esperadas: dict) -> dict:
    """Verifica que el dataset tenga las columnas y tipos de datos esperados.

    columnas_esperadas: dict {nombre_columna: tipo_pandas_esperado}

    Nota: para columnas de texto, pandas 2.x reporta el dtype como "object"
    mientras que versiones más recientes (pandas 3.x) pueden reportarlo como
    "str". Ambos se consideran equivalentes para no depender de la versión
    de pandas instalada.
    """
    equivalencias_texto = {"object", "str"}

    def _tipos_compatibles(tipo_real: str, tipo_esperado: str) -> bool:
        if tipo_esperado in equivalencias_texto and tipo_real in equivalencias_texto:
            return True
        return tipo_real.startswith(tipo_esperado)

    columnas_faltantes = [c for c in columnas_esperadas if c not in df.columns]
    tipos_incorrectos = []
    for col, tipo_esperado in columnas_esperadas.items():
        if col in df.columns:
            tipo_real = str(df[col].dtype)
            if not _tipos_compatibles(tipo_real, tipo_esperado):
                tipos_incorrectos.append((col, tipo_real, tipo_esperado))

    cumple = (len(columnas_faltantes) == 0) and (len(tipos_incorrectos) == 0)
    return {
        "dimension": "exactitud",
        "chequeo": "formato_correcto",
        "cumple": cumple,
        "columnas_faltantes": columnas_faltantes,
        "tipos_incorrectos": tipos_incorrectos,
    }


def verificar_rango(df: pd.DataFrame, columna: str, minimo, maximo, tolerancia_pct: float = 0.0) -> dict:
    """Verifica que los valores de una columna estén dentro de un rango válido.

    tolerancia_pct: porcentaje máximo de filas fuera de rango que se
    considera aceptable (0.0 = tolerancia cero).
    """
    total = len(df)
    fuera_de_rango = df[(df[columna] < minimo) | (df[columna] > maximo)]
    cantidad = len(fuera_de_rango)
    porcentaje = (cantidad / total * 100) if total > 0 else 0.0

    return {
        "dimension": "exactitud",
        "chequeo": f"rango_valido[{columna}]",
        "rango_esperado": (minimo, maximo),
        "cantidad_fuera_de_rango": cantidad,
        "porcentaje_fuera_de_rango": round(porcentaje, 4),
        "tolerancia_pct": tolerancia_pct,
        "cumple": porcentaje <= tolerancia_pct,
    }


def verificar_valores_permitidos(df: pd.DataFrame, columna: str, valores_validos: set) -> dict:
    """Verifica que una columna categórica solo contenga valores permitidos."""
    valores_encontrados = set(df[columna].dropna().unique())
    valores_invalidos = valores_encontrados - valores_validos
    return {
        "dimension": "exactitud",
        "chequeo": f"valores_permitidos[{columna}]",
        "valores_invalidos": list(valores_invalidos),
        "cumple": len(valores_invalidos) == 0,
    }


# ---------------------------------------------------------------------------
# Dimensión: COMPLETITUD
# ---------------------------------------------------------------------------

def verificar_nulos(df: pd.DataFrame, columna: str, tolerancia_pct: float = 5.0) -> dict:
    """Verifica el porcentaje de valores nulos de una columna contra una tolerancia."""
    total = len(df)
    nulos = df[columna].isnull().sum()
    porcentaje = (nulos / total * 100) if total > 0 else 0.0

    return {
        "dimension": "completitud",
        "chequeo": f"valores_nulos[{columna}]",
        "cantidad_nulos": int(nulos),
        "porcentaje_nulos": round(porcentaje, 4),
        "tolerancia_pct": tolerancia_pct,
        "cumple": porcentaje <= tolerancia_pct,
    }


# ---------------------------------------------------------------------------
# Dimensión: CONSISTENCIA
# ---------------------------------------------------------------------------

def verificar_unicidad(df: pd.DataFrame, columna: str) -> dict:
    """Verifica que una columna clave no tenga valores duplicados."""
    duplicados = df[columna].duplicated().sum()
    return {
        "dimension": "consistencia",
        "chequeo": f"unicidad[{columna}]",
        "cantidad_duplicados": int(duplicados),
        "cumple": duplicados == 0,
    }


def verificar_integridad_referencial(df_hijo: pd.DataFrame, df_padre: pd.DataFrame, columna: str) -> dict:
    """Verifica que todos los valores de la clave en df_hijo existan en df_padre."""
    ids_padre = set(df_padre[columna])
    ids_hijo = set(df_hijo[columna])
    huerfanos = ids_hijo - ids_padre

    return {
        "dimension": "consistencia",
        "chequeo": f"integridad_referencial[{columna}]",
        "cantidad_huerfanos": len(huerfanos),
        "cumple": len(huerfanos) == 0,
    }


# ---------------------------------------------------------------------------
# Reporte consolidado (utilizado por el pipeline)
# ---------------------------------------------------------------------------

def generar_reporte_calidad(creditos: pd.DataFrame, tarjetas: pd.DataFrame) -> list:
    """Ejecuta todas las verificaciones de calidad definidas para el proyecto
    y retorna la lista de resultados, lista para ser impresa o exportada.
    """
    resultados = []

    # Exactitud
    resultados.append(verificar_rango(creditos, "edad", 18, 90, tolerancia_pct=1.0))
    resultados.append(verificar_rango(creditos, "antiguedad_empleado", 0, 60, tolerancia_pct=1.0))
    resultados.append(verificar_valores_permitidos(
        creditos, "situacion_vivienda", {"ALQUILER", "PROPIA", "HIPOTECA", "OTROS"}
    ))
    resultados.append(verificar_valores_permitidos(
        creditos, "falta_pago", {"Y", "N"}
    ))

    # Completitud
    resultados.append(verificar_nulos(creditos, "antiguedad_empleado", tolerancia_pct=5.0))
    resultados.append(verificar_nulos(creditos, "tasa_interes", tolerancia_pct=5.0))

    # Consistencia
    resultados.append(verificar_unicidad(creditos, "id_cliente"))
    resultados.append(verificar_unicidad(tarjetas, "id_cliente"))
    resultados.append(verificar_integridad_referencial(tarjetas, creditos, "id_cliente"))
    resultados.append(verificar_integridad_referencial(creditos, tarjetas, "id_cliente"))

    return resultados


if __name__ == "__main__":
    creditos = pd.read_csv("data/raw/datos_creditos.csv", sep=";")
    tarjetas = pd.read_csv("data/raw/datos_tarjetas.csv", sep=";")

    reporte = generar_reporte_calidad(creditos, tarjetas)

    print("=" * 70)
    print("REPORTE DE CALIDAD DE DATOS")
    print("=" * 70)
    for r in reporte:
        estado = "OK" if r["cumple"] else "NC (no cumplimiento)"
        print(f"[{r['dimension'].upper():12s}] {r['chequeo']:40s} -> {estado}")
    print("=" * 70)
