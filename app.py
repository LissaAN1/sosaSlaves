import os
import sys
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf

# Permite importar desde src/
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils.helpers import process_image_from_streamlit


# ============================================================
# Configuración general
# ============================================================

QUALITY_CLASSES = ["Mala", "Buena", "Regular"]
SIZE_CLASSES = ["Pequeño", "Mediano", "Grande"]

MODEL_CONFIGS = {
    "XGBoost — mejor F1 promedio": {
        "path": "saved_models/xgboost_multioutput.pkl",
        "type": "ml",
        "description": "Modelo tradicional seleccionado como mejor balance global entre calidad y tamaño."
    },
    "Random Forest": {
        "path": "saved_models/random_forest_multioutput.pkl",
        "type": "ml",
        "description": "Modelo tradicional basado en ensamble de árboles."
    },
    "SVM lineal": {
        "path": "saved_models/svm_lineal_multioutput.pkl",
        "type": "ml",
        "description": "Modelo tradicional con margen lineal sobre características HSV + área."
    },
    "Mejor modelo tradicional guardado": {
        "path": "saved_models/best_traditional_multioutput.pkl",
        "type": "ml",
        "description": "Archivo exportado con el mejor modelo tradicional según los experimentos."
    },
    "CNN multi-salida": {
        "path": "saved_models/cnn_multioutput.keras",
        "type": "dl",
        "description": "Red neuronal convolucional bifurcada para calidad y tamaño."
    },
}


# ============================================================
# Funciones auxiliares
# ============================================================

def decode_prediction(value, class_names):
    """
    Convierte una predicción numérica o textual en una etiqueta legible.
    """
    if isinstance(value, str):
        value_lower = value.lower()

        mapping = {
            "bad": "Mala",
            "mala": "Mala",
            "good": "Buena",
            "buena": "Buena",
            "regular": "Regular",
            "small": "Pequeño",
            "pequeño": "Pequeño",
            "pequeno": "Pequeño",
            "medium": "Mediano",
            "mediano": "Mediano",
            "large": "Grande",
            "grande": "Grande",
        }

        return mapping.get(value_lower, value)

    try:
        idx = int(value)
        if 0 <= idx < len(class_names):
            return class_names[idx]
    except Exception:
        pass

    return str(value)


def get_ml_confidences(model, features):
    """
    Intenta obtener confianza para modelos tradicionales.
    MultiOutputClassifier normalmente devuelve una lista:
    [probabilidades_calidad, probabilidades_tamaño].
    Si el modelo no soporta predict_proba, retorna None.
    """
    if not hasattr(model, "predict_proba"):
        return None, None

    try:
        probabilities = model.predict_proba(features)

        if isinstance(probabilities, list) and len(probabilities) >= 2:
            conf_quality = float(np.max(probabilities[0])) * 100
            conf_size = float(np.max(probabilities[1])) * 100
            return conf_quality, conf_size

    except Exception:
        pass

    return None, None


@st.cache_resource
def load_ml_model(model_path):
    """
    Carga modelos tradicionales serializados con joblib.
    """
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_resource
def load_dl_model(model_path):
    """
    Carga modelos Keras.
    """
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None


# ============================================================
# Interfaz Streamlit
# ============================================================

st.set_page_config(
    page_title="Clasificador Multi-Salida de Frutas",
    layout="centered"
)

st.title("Clasificador de Calidad y Tamaño de Frutas")

st.markdown("""
Esta aplicación utiliza visión por computadora y aprendizaje automático para predecir simultáneamente:

1. **Calidad**: Mala, Buena o Regular.  
2. **Tamaño**: Pequeño, Mediano o Grande.

Puedes subir una imagen o capturar una foto usando la cámara del dispositivo.
""")


# ============================================================
# Panel lateral
# ============================================================

st.sidebar.title("Configuración")

model_choice = st.sidebar.selectbox(
    "Selecciona el modelo predictivo:",
    list(MODEL_CONFIGS.keys())
)

selected_model = MODEL_CONFIGS[model_choice]
selected_model_path = selected_model["path"]
selected_model_type = selected_model["type"]

st.sidebar.markdown("### Modelo seleccionado")
st.sidebar.write(selected_model["description"])
st.sidebar.caption(f"Ruta esperada: `{selected_model_path}`")

st.sidebar.markdown("---")

confidence_threshold = st.sidebar.slider(
    "Umbral mínimo de confianza",
    min_value=0,
    max_value=100,
    value=60,
    step=5,
    help="Si la confianza queda por debajo de este valor, se recomienda revisión humana."
)

st.sidebar.markdown("---")

st.sidebar.info(
    "**Proyecto Final — Algoritmos y Programación III**\n\n"
    "**Equipo sosaSlaves**\n"
    "- Angy Hurtado\n"
    "- Hideki Tamura\n"
    "- David Vergara"
)

st.sidebar.caption(
    "Privacidad: las imágenes se usan únicamente para generar la predicción "
    "en la sesión actual. La aplicación no almacena imágenes de usuarios."
)


# ============================================================
# Entrada de imagen
# ============================================================

input_method = st.radio(
    "Método de entrada de imagen:",
    ("Cargar archivo", "Usar cámara")
)

uploaded_file = None

if input_method == "Cargar archivo":
    uploaded_file = st.file_uploader(
        "Sube una imagen de la fruta",
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_file = st.camera_input("Toma una foto con tu cámara web")


# ============================================================
# Inferencia
# ============================================================

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Vista previa de la imagen",
        use_container_width=True
    )

    if st.button("Predecir simultáneamente"):
        with st.spinner("Procesando imagen y ejecutando inferencia..."):

            try:
                # --------------------------------------------------------
                # Modelos tradicionales: XGBoost, Random Forest, SVM
                # --------------------------------------------------------
                if selected_model_type == "ml":
                    model = load_ml_model(selected_model_path)

                    if model is None:
                        st.error(
                            f"No se encontró el modelo seleccionado: `{selected_model_path}`.\n\n"
                            "Verifica que la carpeta `saved_models/` exista y que el archivo esté guardado correctamente."
                        )
                        st.stop()

                    features = process_image_from_streamlit(
                        uploaded_file,
                        model_type="ml"
                    )

                    predictions = model.predict(features)

                    pred_quality_raw = predictions[0][0]
                    pred_size_raw = predictions[0][1]

                    pred_quality = decode_prediction(
                        pred_quality_raw,
                        QUALITY_CLASSES
                    )
                    pred_size = decode_prediction(
                        pred_size_raw,
                        SIZE_CLASSES
                    )

                    conf_quality, conf_size = get_ml_confidences(model, features)

                    st.success(f"Análisis completado con: {model_choice}")

                    col1, col2 = st.columns(2)

                    if conf_quality is not None:
                        col1.metric(
                            "Calidad",
                            pred_quality,
                            f"{conf_quality:.1f}% confianza"
                        )
                    else:
                        col1.metric("Calidad", pred_quality)

                    if conf_size is not None:
                        col2.metric(
                            "Tamaño",
                            pred_size,
                            f"{conf_size:.1f}% confianza"
                        )
                    else:
                        col2.metric("Tamaño", pred_size)

                    if conf_quality is not None and conf_size is not None:
                        if conf_quality < confidence_threshold or conf_size < confidence_threshold:
                            st.warning(
                                "La confianza del modelo está por debajo del umbral definido. "
                                "Se recomienda revisión humana antes de tomar una decisión."
                            )

                # --------------------------------------------------------
                # CNN multi-salida
                # --------------------------------------------------------
                else:
                    model = load_dl_model(selected_model_path)

                    if model is None:
                        st.error(
                            f"No se encontró el modelo CNN: `{selected_model_path}`.\n\n"
                            "Verifica que la carpeta `saved_models/` exista y que el archivo `.keras` esté guardado correctamente."
                        )
                        st.stop()

                    img_tensor = process_image_from_streamlit(
                        uploaded_file,
                        model_type="dl"
                    )

                    predictions = model.predict(img_tensor)

                    # Keras puede devolver dict o lista, según cómo se haya guardado el modelo.
                    if isinstance(predictions, dict):
                        quality_probs = predictions["quality_output"]
                        size_probs = predictions["size_output"]
                    else:
                        quality_probs = predictions[0]
                        size_probs = predictions[1]

                    pred_quality_idx = int(np.argmax(quality_probs, axis=1)[0])
                    pred_size_idx = int(np.argmax(size_probs, axis=1)[0])

                    conf_quality = float(np.max(quality_probs)) * 100
                    conf_size = float(np.max(size_probs)) * 100

                    pred_quality = decode_prediction(
                        pred_quality_idx,
                        QUALITY_CLASSES
                    )
                    pred_size = decode_prediction(
                        pred_size_idx,
                        SIZE_CLASSES
                    )

                    st.success(f"Análisis completado con: {model_choice}")

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Calidad",
                        pred_quality,
                        f"{conf_quality:.1f}% confianza"
                    )

                    col2.metric(
                        "Tamaño",
                        pred_size,
                        f"{conf_size:.1f}% confianza"
                    )

                    if conf_quality < confidence_threshold or conf_size < confidence_threshold:
                        st.warning(
                            "La confianza del modelo está por debajo del umbral definido. "
                            "Se recomienda revisión humana antes de tomar una decisión."
                        )

            except Exception as e:
                st.error("Ocurrió un error inesperado durante la predicción.")
                st.exception(e)


# ============================================================
# Validación visual de archivos esperados
# ============================================================

with st.expander("Verificar archivos de modelos esperados"):
    for model_name, config in MODEL_CONFIGS.items():
        path = config["path"]

        if os.path.exists(path):
            st.success(f"{model_name}: encontrado en `{path}`")
        else:
            st.warning(f"{model_name}: no encontrado en `{path}`")