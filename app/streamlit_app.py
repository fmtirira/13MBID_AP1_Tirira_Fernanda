"""
Interfaz web para consultar predicciones de mora en créditos.
Corresponde a la Actividad 3 (Despliegue de la solución - parte 1) de AP2.

"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Predicción de Mora - 13MBID", page_icon="💳")
st.title("💳 Predicción de Mora en Créditos")
st.caption("Actividad Práctica II - 13MBID")

with st.form("formulario_prediccion"):
    col1, col2 = st.columns(2)

    with col1:
        importe_solicitado = st.number_input("Importe solicitado", min_value=0.0, value=10000.0)
        duracion_credito = st.number_input("Duración del crédito (meses)", min_value=0.0, value=36.0)
        situacion_vivienda = st.selectbox("Situación de vivienda", ["ALQUILER", "PROPIA", "HIPOTECA", "OTROS"])
        ingresos = st.number_input("Ingresos", min_value=0.0, value=50000.0)
        objetivo_credito = st.selectbox("Objetivo del crédito", ["PERSONAL", "EDUCACIÓN", "SALUD", "INVERSIONES", "MEJORAS_HOGAR", "PAGO_DEUDAS"])
        pct_ingreso = st.number_input("% del ingreso comprometido", min_value=0.0, max_value=1.0, value=0.2)
        tasa_interes = st.number_input("Tasa de interés", min_value=0.0, value=12.5)
        estado_civil_N = st.selectbox("Estado civil", ["C", "S", "N", "D"])
        estado_credito_N = st.selectbox("Estado del crédito", ["C", "P"])

    with col2:
        antiguedad_cliente = st.number_input("Antigüedad del cliente (años)", min_value=0.0, value=5.0)
        estado_cliente = st.selectbox("Estado del cliente", ["ACTIVO", "PASIVO"])
        gastos_ult_12m = st.number_input("Gastos últimos 12 meses", min_value=0.0, value=3000.0)
        genero = st.selectbox("Género", ["M", "F"])
        limite_credito_tc = st.number_input("Límite de crédito (tarjeta)", min_value=0.0, value=5000.0)
        nivel_educativo = st.selectbox("Nivel educativo", ["UNIVERSITARIO_COMPLETO", "SECUNDARIO_COMPLETO", "DESCONOCIDO", "UNIVERSITARIO_INCOMPLETO", "POSGRADO_INCOMPLETO", "POSGRADO_COMPLETO"])
        operaciones_ult_12m = st.number_input("Operaciones últimos 12 meses", min_value=0.0, value=10.0)
        personas_a_cargo = st.number_input("Personas a cargo", min_value=0.0, value=2.0)
        antiguedad_empleado_N = st.selectbox("Antigüedad empleado", ["menor_5", "5_a_10", "mayor_10"])
        edad_N = st.selectbox("Rango de edad", ["menor_25", "25_a_30"])

    enviado = st.form_submit_button("Predecir")

if enviado:
    payload = {
        "importe_solicitado": importe_solicitado,
        "duracion_credito": duracion_credito,
        "situacion_vivienda": situacion_vivienda,
        "ingresos": ingresos,
        "objetivo_credito": objetivo_credito,
        "pct_ingreso": pct_ingreso,
        "tasa_interes": tasa_interes,
        "antiguedad_cliente": antiguedad_cliente,
        "estado_cliente": estado_cliente,
        "gastos_ult_12m": gastos_ult_12m,
        "genero": genero,
        "limite_credito_tc": limite_credito_tc,
        "nivel_educativo": nivel_educativo,
        "operaciones_ult_12m": operaciones_ult_12m,
        "personas_a_cargo": personas_a_cargo,
        "estado_civil_N": estado_civil_N,
        "estado_credito_N": estado_credito_N,
        "antiguedad_empleado_N": antiguedad_empleado_N,
        "edad_N": edad_N,
    }
    try:
        respuesta = requests.post(f"{API_URL}/predecir", json=payload, timeout=10)
        respuesta.raise_for_status()
        resultado = respuesta.json()

        if resultado["prediccion"] == "Mora":
            st.error(f"⚠️ Predicción: {resultado['prediccion']}")
        else:
            st.success(f"✅ Predicción: {resultado['prediccion']}")
        st.metric("Probabilidad de mora", f"{resultado['probabilidad_mora']*100:.2f}%")
    except Exception as e:
        st.error(f"Error al consultar la API: {e}")