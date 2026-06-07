import os
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    tf = None
    TF_AVAILABLE = False

# Permite importar desde src/
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils.helpers import process_image_from_streamlit


# ============================================================
# Configuración general
# ============================================================

MODEL_DIR = Path("saved_models")
METADATA_PATH = MODEL_DIR / "traditional_model_metadata.json"

DEFAULT_QUALITY_CLASSES = ["Mala", "Buena", "Regular"]
DEFAULT_SIZE_CLASSES = ["Pequeño", "Mediano", "Grande"]

# Artefactos esperados para la entrega final.
# Nota: random_forest_tuned_multioutput.pkl NO es obligatorio.
EXPECTED_ARTIFACTS = {
    "random_forest_multioutput.pkl": "Random Forest base generado por notebook 02",
    "svm_lineal_multioutput.pkl": "SVM lineal generado por notebook 02",
    "xgboost_multioutput.pkl": "XGBoost generado por notebook 02",
    "best_traditional_multioutput.pkl": "Mejor modelo tradicional; corresponde al XGBoost ganador",
    "best_rf_multioutput.pkl": "Copia de Random Forest para compatibilidad con versiones previas",
    "traditional_model_metadata.json": "Metadatos del pipeline tradicional",
    "traditional_models_results.csv": "Tabla final de resultados de modelos tradicionales",
    "cnn_best_checkpoint.keras": "Checkpoint del mejor epoch de la CNN",
    "cnn_multioutput.keras": "CNN final cargada desde el checkpoint",
}

# Modelos seleccionables por el usuario.
# No se incluyen modelos auxiliares como best_rf_multioutput.pkl ni cnn_best_checkpoint.keras.
MODEL_CONFIGS = {
    "Mejor tradicional — XGBoost": {
        "path": MODEL_DIR / "best_traditional_multioutput.pkl",
        "type": "ml",
        "description": (
            "Modelo tradicional seleccionado por mayor F1-macro promedio multi-salida. "
            "En esta entrega corresponde al XGBoost ganador."
        ),
    },
    "XGBoost": {
        "path": MODEL_DIR / "xgboost_multioutput.pkl",
        "type": "ml",
        "description": "XGBoost entrenado sobre histograma HSV de 512 bins + área relativa.",
    },
    "Random Forest": {
        "path": MODEL_DIR / "random_forest_multioutput.pkl",
        "type": "ml",
        "description": "Random Forest base usado como modelo comparativo tradicional.",
    },
    "SVM lineal": {
        "path": MODEL_DIR / "svm_lineal_multioutput.pkl",
        "type": "ml",
        "description": "SVM lineal entrenado sobre características HSV + área relativa.",
    },
    "CNN final": {
        "path": MODEL_DIR / "cnn_multioutput.keras",
        "type": "dl",
        "description": "CNN multi-salida final cargada desde el mejor checkpoint.",
    },
}


# ============================================================
# Utilidades de metadatos y etiquetas
# ============================================================

def english_to_spanish_label(label):
    """Normaliza etiquetas numéricas/textuales a nombres legibles en español."""
    if label is None:
        return "Desconocido"

    # Si llega una etiqueta numérica, se devuelve como texto. La decodificación
    # correcta de índices ocurre en decode_prediction().
    if isinstance(label, (int, np.integer)):
        return str(int(label))

    text = str(label).strip()
    key = text.lower()

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

    return mapping.get(key, text)


@st.cache_data(show_spinner=False)
def load_metadata(metadata_path):
    """Carga metadatos del modelo tradicional si existen."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def _get_first_existing(metadata, keys, default_value):
    for key in keys:
        if key in metadata and metadata[key]:
            return metadata[key]
    return default_value


def _looks_numeric_list(values):
    try:
        return all(str(value).strip().isdigit() for value in values)
    except Exception:
        return False


def get_class_names(metadata):
    """
    Obtiene el orden de clases desde traditional_model_metadata.json cuando está disponible.

    Importante:
    El notebook 02 guarda nombres legibles en `quality_names` y `size_names`.
    Por eso estas claves se priorizan sobre `quality_labels` y `size_labels`,
    que normalmente contienen códigos numéricos [0, 1, 2].
    """
    quality_raw = _get_first_existing(
        metadata,
        ["quality_names", "quality_classes", "quality_class_names", "quality_order"],
        DEFAULT_QUALITY_CLASSES,
    )
    size_raw = _get_first_existing(
        metadata,
        ["size_names", "size_classes", "size_class_names", "size_order"],
        DEFAULT_SIZE_CLASSES,
    )

    quality_classes = [english_to_spanish_label(x) for x in quality_raw]
    size_classes = [english_to_spanish_label(x) for x in size_raw]

    if len(quality_classes) != 3 or _looks_numeric_list(quality_classes):
        quality_classes = DEFAULT_QUALITY_CLASSES
    if len(size_classes) != 3 or _looks_numeric_list(size_classes):
        size_classes = DEFAULT_SIZE_CLASSES

    return quality_classes, size_classes


def decode_prediction(value, class_names):
    """Convierte una predicción numérica o textual en una etiqueta legible."""
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, str):
        # Si el modelo devuelve texto en inglés/español.
        return english_to_spanish_label(value)

    try:
        idx = int(value)
        if 0 <= idx < len(class_names):
            return english_to_spanish_label(class_names[idx])
    except Exception:
        pass

    return english_to_spanish_label(value)


# ============================================================
# Utilidades de modelos
# ============================================================

def get_available_model_configs():
    """Retorna solo los modelos seleccionables cuyo archivo existe en saved_models/."""
    return {
        name: config
        for name, config in MODEL_CONFIGS.items()
        if Path(config["path"]).exists()
    }


def get_default_model_index(model_names):
    """Prioriza el mejor modelo tradicional; si no existe, cae a XGBoost o CNN."""
    priority = [
        "Mejor tradicional — XGBoost",
        "XGBoost",
        "CNN final",
        "Random Forest",
        "SVM lineal",
    ]

    for preferred in priority:
        if preferred in model_names:
            return model_names.index(preferred)

    return 0


@st.cache_resource(show_spinner=False)
def load_ml_model(model_path):
    """Carga modelos tradicionales serializados con joblib."""
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def load_dl_model(model_path):
    """Carga modelos Keras."""
    if not TF_AVAILABLE:
        return None

    model_path = Path(model_path)
    if not model_path.exists():
        return None

    return tf.keras.models.load_model(model_path)


def get_ml_confidences(model, features):
    """
    Intenta obtener confianza para modelos tradicionales.
    MultiOutputClassifier normalmente devuelve una lista:
    [probabilidades_calidad, probabilidades_tamaño].
    """
    if not hasattr(model, "predict_proba"):
        return None, None

    try:
        probabilities = model.predict_proba(features)

        if isinstance(probabilities, list) and len(probabilities) >= 2:
            conf_quality = float(np.max(probabilities[0], axis=1)[0]) * 100
            conf_size = float(np.max(probabilities[1], axis=1)[0]) * 100
            return conf_quality, conf_size

    except Exception:
        return None, None

    return None, None


def parse_ml_predictions(predictions):
    """
    Extrae las dos salidas de un modelo tradicional multi-salida.
    Esperado: shape (1, 2), donde columna 0 es calidad y columna 1 es tamaño.
    """
    arr = np.asarray(predictions)

    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[0, 0], arr[0, 1]

    if isinstance(predictions, (list, tuple)) and len(predictions) >= 2:
        return predictions[0], predictions[1]

    raise ValueError(
        "La predicción del modelo tradicional no tiene el formato esperado. "
        "Se esperaba una salida multi-output con calidad y tamaño."
    )


def parse_cnn_predictions(predictions):
    """
    Extrae las probabilidades de calidad y tamaño de una CNN multi-salida.
    Soporta salida tipo dict o lista/tupla.
    """
    if isinstance(predictions, dict):
        quality_probs = predictions.get("quality_output")
        size_probs = predictions.get("size_output")

        if quality_probs is None or size_probs is None:
            keys = list(predictions.keys())
            raise ValueError(
                f"La CNN devolvió un diccionario sin las claves esperadas. "
                f"Claves encontradas: {keys}"
            )

        return np.asarray(quality_probs), np.asarray(size_probs)

    if isinstance(predictions, (list, tuple)) and len(predictions) >= 2:
        return np.asarray(predictions[0]), np.asarray(predictions[1])

    raise ValueError(
        "La predicción de la CNN no tiene el formato esperado. "
        "Se esperaban dos salidas: quality_output y size_output."
    )


def validate_feature_shape(features, metadata):
    """Valida la dimensión del vector de características tradicional."""
    expected_dim = metadata.get("feature_dim") or metadata.get("n_features") or 513

    try:
        expected_dim = int(expected_dim)
    except Exception:
        expected_dim = 513

    if features.ndim != 2 or features.shape[1] != expected_dim:
        st.warning(
            f"Advertencia: el vector de características tiene forma {features.shape}, "
            f"pero se esperaba dimensión (1, {expected_dim}). "
            "Revisa que el entrenamiento y la inferencia usen el mismo pipeline."
        )


# ============================================================
# Interfaz Streamlit
# ============================================================

st.set_page_config(
    page_title="Clasificador Multi-Salida de Frutas",
    layout="centered",
)

st.title("Clasificador de Calidad y Tamaño de Frutas")

st.markdown(
    """
Esta aplicación predice simultáneamente:

1. **Calidad**: Mala, Buena o Regular.  
2. **Tamaño**: Pequeño, Mediano o Grande.

Puedes subir una imagen o capturar una foto usando la cámara del dispositivo.
"""
)

metadata = load_metadata(METADATA_PATH)
QUALITY_CLASSES, SIZE_CLASSES = get_class_names(metadata)
available_model_configs = get_available_model_configs()


# ============================================================
# Panel lateral
# ============================================================

st.sidebar.title("Configuración")

if not MODEL_DIR.exists():
    st.sidebar.error("No existe la carpeta `saved_models/`.")

if not available_model_configs:
    st.error(
        "No se encontró ningún modelo seleccionable en `saved_models/`. "
        "Por ahora la app no puede ejecutar inferencia."
    )

    st.markdown("### Archivos esperados")
    st.dataframe(
        [
            {
                "Archivo": filename,
                "Estado": "Encontrado" if (MODEL_DIR / filename).exists() else "Falta",
                "Descripción": description,
            }
            for filename, description in EXPECTED_ARTIFACTS.items()
        ],
        use_container_width=True,
    )
    st.stop()

model_names = list(available_model_configs.keys())
default_index = get_default_model_index(model_names)

model_choice = st.sidebar.selectbox(
    "Selecciona el modelo predictivo:",
    model_names,
    index=default_index,
)

selected_model = available_model_configs[model_choice]
selected_model_path = Path(selected_model["path"])
selected_model_type = selected_model["type"]

st.sidebar.markdown("### Modelo seleccionado")
st.sidebar.write(selected_model["description"])
st.sidebar.caption(f"Ruta: `{selected_model_path}`")

confidence_threshold = st.sidebar.slider(
    "Umbral mínimo de confianza",
    min_value=0,
    max_value=100,
    value=60,
    step=5,
    help="Si la confianza queda por debajo de este valor, se recomienda revisión humana.",
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

with st.sidebar.expander("Estado de artefactos esperados"):
    for filename, description in EXPECTED_ARTIFACTS.items():
        path = MODEL_DIR / filename
        if path.exists():
            st.success(f"{filename}")
        else:
            st.warning(f"Falta: {filename}")
        st.caption(description)

if metadata:
    with st.sidebar.expander("Metadatos cargados"):
        st.json(metadata)
else:
    st.sidebar.warning(
        "No se encontró `traditional_model_metadata.json`. "
        "La app usará etiquetas y dimensión de features por defecto."
    )


# ============================================================
# Entrada de imagen
# ============================================================

input_method = st.radio(
    "Método de entrada de imagen:",
    ("Cargar archivo", "Usar cámara"),
)

uploaded_file = None

if input_method == "Cargar archivo":
    uploaded_file = st.file_uploader(
        "Sube una imagen de la fruta",
        type=["jpg", "jpeg", "png"],
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
        use_container_width=True,
    )

    if st.button("Predecir simultáneamente"):
        with st.spinner("Procesando imagen y ejecutando inferencia..."):
            try:
                if selected_model_type == "ml":
                    model = load_ml_model(selected_model_path)

                    if model is None:
                        st.error(
                            f"No se pudo cargar el modelo seleccionado: `{selected_model_path}`."
                        )
                        st.stop()

                    features = process_image_from_streamlit(
                        uploaded_file,
                        model_type="ml",
                    )
                    features = np.asarray(features)
                    validate_feature_shape(features, metadata)

                    predictions = model.predict(features)
                    pred_quality_raw, pred_size_raw = parse_ml_predictions(predictions)

                    pred_quality = decode_prediction(
                        pred_quality_raw,
                        QUALITY_CLASSES,
                    )
                    pred_size = decode_prediction(
                        pred_size_raw,
                        SIZE_CLASSES,
                    )

                    conf_quality, conf_size = get_ml_confidences(model, features)

                    st.success(f"Análisis completado con: {model_choice}")

                    col1, col2 = st.columns(2)

                    if conf_quality is not None:
                        col1.metric(
                            "Calidad",
                            pred_quality,
                            f"{conf_quality:.1f}% confianza",
                        )
                    else:
                        col1.metric("Calidad", pred_quality)

                    if conf_size is not None:
                        col2.metric(
                            "Tamaño",
                            pred_size,
                            f"{conf_size:.1f}% confianza",
                        )
                    else:
                        col2.metric("Tamaño", pred_size)

                    if conf_quality is not None and conf_size is not None:
                        if (
                            conf_quality < confidence_threshold
                            or conf_size < confidence_threshold
                        ):
                            st.warning(
                                "La confianza del modelo está por debajo del umbral definido. "
                                "Se recomienda revisión humana antes de tomar una decisión."
                            )
                    else:
                        st.info(
                            "Este modelo no expuso probabilidades con `predict_proba`; "
                            "por eso no se muestra porcentaje de confianza."
                        )

                else:
                    if not TF_AVAILABLE:
                        st.error(
                            "TensorFlow no está disponible en este entorno. "
                            "Instala las dependencias de `requirements.txt` para usar la CNN."
                        )
                        st.stop()

                    model = load_dl_model(selected_model_path)

                    if model is None:
                        st.error(
                            f"No se pudo cargar la CNN seleccionada: `{selected_model_path}`."
                        )
                        st.stop()

                    img_tensor = process_image_from_streamlit(
                        uploaded_file,
                        model_type="dl",
                    )

                    predictions = model.predict(img_tensor, verbose=0)
                    quality_probs, size_probs = parse_cnn_predictions(predictions)

                    pred_quality_idx = int(np.argmax(quality_probs, axis=1)[0])
                    pred_size_idx = int(np.argmax(size_probs, axis=1)[0])

                    conf_quality = float(np.max(quality_probs, axis=1)[0]) * 100
                    conf_size = float(np.max(size_probs, axis=1)[0]) * 100

                    pred_quality = decode_prediction(
                        pred_quality_idx,
                        QUALITY_CLASSES,
                    )
                    pred_size = decode_prediction(
                        pred_size_idx,
                        SIZE_CLASSES,
                    )

                    st.success(f"Análisis completado con: {model_choice}")

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Calidad",
                        pred_quality,
                        f"{conf_quality:.1f}% confianza",
                    )

                    col2.metric(
                        "Tamaño",
                        pred_size,
                        f"{conf_size:.1f}% confianza",
                    )

                    if conf_quality < confidence_threshold or conf_size < confidence_threshold:
                        st.warning(
                            "La confianza del modelo está por debajo del umbral definido. "
                            "Se recomienda revisión humana antes de tomar una decisión."
                        )

                    with st.expander("Probabilidades CNN"):
                        st.markdown("#### Calidad")
                        st.dataframe(
                            [
                                {
                                    "Clase": QUALITY_CLASSES[i],
                                    "Probabilidad": float(quality_probs[0][i]),
                                }
                                for i in range(min(len(QUALITY_CLASSES), quality_probs.shape[1]))
                            ],
                            use_container_width=True,
                        )

                        st.markdown("#### Tamaño")
                        st.dataframe(
                            [
                                {
                                    "Clase": SIZE_CLASSES[i],
                                    "Probabilidad": float(size_probs[0][i]),
                                }
                                for i in range(min(len(SIZE_CLASSES), size_probs.shape[1]))
                            ],
                            use_container_width=True,
                        )

            except Exception as error:
                st.error("Ocurrió un error inesperado durante la predicción.")
                st.exception(error)


# ============================================================
# Validación visual de archivos esperados
# ============================================================

with st.expander("Verificación completa de artefactos"):
    rows = []
    for filename, description in EXPECTED_ARTIFACTS.items():
        path = MODEL_DIR / filename
        rows.append(
            {
                "Archivo": filename,
                "Estado": "Encontrado" if path.exists() else "Falta",
                "Ruta": str(path),
                "Descripción": description,
            }
        )

    st.dataframe(rows, use_container_width=True)

    st.caption(
        "Nota: `random_forest_tuned_multioutput.pkl` no es un requisito de la entrega final. "
        "Puede generarse opcionalmente si se activa el ajuste de hiperparámetros de Random Forest."
    )
