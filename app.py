import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Predicción de riesgo actuarial",
    page_icon="📊",
    layout="centered",
)

st.title("Predicción de riesgo actuarial - Marcio Rivera PTI-0620")

st.write(
    "Ingrese los datos solicitados para estimar el nivel de riesgo actuarial."
)


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# IMPORTANTE:
# La aplicación utiliza el modelo SVM, NO el K-means.
MODEL_PATH = BASE_DIR / "svm_riesgo_actuarial.pkl"

METADATA_PATH = BASE_DIR / "model_metadata.json"


# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def cargar_modelo(ruta):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {ruta.name}. "
            "Verifique que esté en la misma carpeta que app.py."
        )

    return joblib.load(ruta)


# ============================================================
# CARGAR METADATA
# ============================================================

@st.cache_data
def cargar_metadata(ruta):

    if not ruta.exists():
        return {}

    with ruta.open("r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    if not isinstance(datos, dict):
        raise ValueError(
            "model_metadata.json debe contener un objeto JSON."
        )

    return datos


# ============================================================
# CONFIGURACIÓN DE CAMPOS
# ============================================================

def crear_campo(variable):

    # --------------------------------------------------------
    # EDAD
    # --------------------------------------------------------

    if variable == "age":

        return st.number_input(
            "Edad",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
            help="Edad del cliente.",
        )

    # --------------------------------------------------------
    # IMC
    # --------------------------------------------------------

    if variable == "bmi":

        return st.number_input(
            "IMC",
            min_value=10.0,
            max_value=80.0,
            value=25.0,
            step=0.1,
            format="%.1f",
            help="Índice de masa corporal.",
        )

    # --------------------------------------------------------
    # HIJOS
    # --------------------------------------------------------

    if variable == "children":

        return st.number_input(
            "Número de hijos",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Número de hijos o dependientes.",
        )

    # --------------------------------------------------------
    # SEXO
    # --------------------------------------------------------

    if variable == "sex":

        return st.selectbox(
            "Sexo",
            options=[
                "male",
                "female",
            ],
        )

    # --------------------------------------------------------
    # FUMADOR
    # --------------------------------------------------------

    if variable == "smoker":

        return st.selectbox(
            "¿Es fumador?",
            options=[
                "no",
                "yes",
            ],
        )

    # --------------------------------------------------------
    # REGIÓN
    # --------------------------------------------------------

    if variable == "region":

        return st.selectbox(
            "Región",
            options=[
                "southwest",
                "southeast",
                "northwest",
                "northeast",
            ],
        )

    # --------------------------------------------------------
    # VARIABLE DESCONOCIDA
    # --------------------------------------------------------

    raise ValueError(
        f"La variable '{variable}' no está configurada "
        "para el formulario."
    )


# ============================================================
# OBTENER VARIABLES DEL SVM
# ============================================================

def obtener_variables_svm(modelo, metadata):

    # Primero intentamos usar exactamente las variables
    # guardadas por el notebook.
    variables = metadata.get("features_svm")

    if isinstance(variables, list) and variables:

        return [str(x) for x in variables]

    # Si metadata no las contiene, utilizamos las variables
    # que sabemos que fueron utilizadas para entrenar el SVM.
    variables = [
        "age",
        "bmi",
        "children",
        "sex",
        "smoker",
        "region",
    ]

    # Si el modelo tiene feature_names_in_, comprobamos que
    # corresponda al conjunto esperado.
    if hasattr(modelo, "feature_names_in_"):

        nombres_modelo = [
            str(x)
            for x in modelo.feature_names_in_
        ]

        # Si el modelo conserva nombres de columnas originales,
        # utilizamos esos nombres.
        if all(
            variable in nombres_modelo
            for variable in variables
        ):
            return variables

    return variables


# ============================================================
# NORMALIZAR DATOS
# ============================================================

def preparar_entrada(valores):

    entrada = pd.DataFrame(
        [valores]
    )

    # --------------------------------------------------------
    # NUMÉRICAS
    # --------------------------------------------------------

    entrada["age"] = pd.to_numeric(
        entrada["age"],
        errors="raise"
    )

    entrada["bmi"] = pd.to_numeric(
        entrada["bmi"],
        errors="raise"
    )

    entrada["children"] = pd.to_numeric(
        entrada["children"],
        errors="raise"
    )

    # --------------------------------------------------------
    # CATEGÓRICAS
    #
    # El notebook utiliza .lower() para estas variables.
    # Las dejamos explícitamente como texto.
    # --------------------------------------------------------

    entrada["sex"] = (
        entrada["sex"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    entrada["smoker"] = (
        entrada["smoker"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    entrada["region"] = (
        entrada["region"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Orden EXACTO utilizado por el SVM
    columnas = [
        "age",
        "bmi",
        "children",
        "sex",
        "smoker",
        "region",
    ]

    return entrada[columnas]


# ============================================================
# CARGAR MODELO Y METADATA
# ============================================================

try:

    modelo = cargar_modelo(
        MODEL_PATH
    )

    metadata = cargar_metadata(
        METADATA_PATH
    )

except Exception as error:

    st.error(
        f"No fue posible iniciar la aplicación: {error}"
    )

    st.stop()


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

try:

    variables = obtener_variables_svm(
        modelo,
        metadata
    )

except Exception as error:

    st.error(
        f"No fue posible determinar las variables del modelo: {error}"
    )

    st.stop()


# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================

with st.sidebar:

    st.header("Información técnica")

    st.write(
        f"**Modelo:** `{MODEL_PATH.name}`"
    )

    st.write(
        f"**Metadatos:** `{METADATA_PATH.name}`"
    )

    st.write(
        "**Algoritmo:** SVM"
    )

    if "svm" in metadata:

        datos_svm = metadata["svm"]

        if isinstance(datos_svm, dict):

            if "accuracy_test" in datos_svm:

                accuracy = datos_svm["accuracy_test"]

                st.write(
                    f"**Accuracy:** {float(accuracy):.2%}"
                )

            if "mejores_parametros" in datos_svm:

                with st.expander(
                    "Parámetros del SVM"
                ):

                    st.json(
                        datos_svm["mejores_parametros"]
                    )

    st.caption(
        "El resultado es una estimación del modelo y debe "
        "interpretarse junto con criterios técnicos y actuariales."
    )


# ============================================================
# FORMULARIO
# ============================================================

with st.form(
    "formulario_prediccion"
):

    st.subheader(
        "Datos para la predicción"
    )

    valores = {}

    for variable in variables:

        valores[variable] = crear_campo(
            variable
        )

    enviar = st.form_submit_button(
        "Calcular riesgo",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# PREDICCIÓN
# ============================================================

if enviar:

    try:

        # ----------------------------------------------------
        # PREPARAR DATAFRAME
        # ----------------------------------------------------

        entrada = preparar_entrada(
            valores
        )

        # ----------------------------------------------------
        # PREDICCIÓN
        # ----------------------------------------------------

        prediccion = modelo.predict(
            entrada
        )[0]

        # ----------------------------------------------------
        # CONVERTIR RESULTADO A TEXTO
        # ----------------------------------------------------

        nivel = str(
            prediccion
        )

        # ----------------------------------------------------
        # MOSTRAR RESULTADO
        # ----------------------------------------------------

        st.success(
            f"Nivel de riesgo estimado: {nivel}"
        )

        # ----------------------------------------------------
        # INFORMACIÓN ADICIONAL
        # ----------------------------------------------------

        if hasattr(
            modelo,
            "predict_proba"
        ):

            try:

                probabilidades = modelo.predict_proba(
                    entrada
                )[0]

                clases = getattr(
                    modelo,
                    "classes_",
                    range(
                        len(probabilidades)
                    ),
                )

                tabla = pd.DataFrame(
                    {
                        "Nivel": [
                            str(clase)
                            for clase in clases
                        ],
                        "Probabilidad": probabilidades,
                    }
                )

                tabla["Probabilidad"] = (
                    tabla["Probabilidad"]
                    .map(
                        lambda x: f"{x:.2%}"
                    )
                )

                st.subheader(
                    "Probabilidades"
                )

                st.dataframe(
                    tabla,
                    hide_index=True,
                    use_container_width=True,
                )

            except Exception:
                # Algunos modelos SVM no tienen probabilidades
                # habilitadas. En ese caso no mostramos la tabla.
                pass

        # ----------------------------------------------------
        # DATOS ENVIADOS AL MODELO
        # ----------------------------------------------------

        with st.expander(
            "Datos enviados al modelo"
        ):

            st.dataframe(
                entrada,
                hide_index=True,
                use_container_width=True,
            )

    except Exception as error:

        st.error(
            f"No fue posible realizar la predicción: {error}"
        )

        st.info(
            "Los datos enviados deben coincidir con las variables "
            "utilizadas durante el entrenamiento del modelo SVM."
        )
