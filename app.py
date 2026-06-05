import streamlit as st
import numpy as np
from PIL import Image
import os
import joblib
import tensorflow as tf

# Configurar el path para importar desde la carpeta src
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils.helpers import process_image_from_streamlit

# Diccionarios de decodificación
QUALITY_CLASSES = ['Mala', 'Regular', 'Buena']
SIZE_CLASSES = ['Pequeño', 'Mediano', 'Grande']

# Rutas de los modelos entrenados
ML_MODEL_PATH = 'models/best_rf_multioutput.pkl'
DL_MODEL_PATH = 'models/cnn_multioutput.keras'

# Configuración inicial de Streamlit
st.set_page_config(page_title="Clasificador Multi-Salida de Frutas", layout="centered")

st.title("Clasificador de Calidad y Tamaño")
st.markdown("""
Esta aplicación utiliza los modelos entrenados en la **Fase 2 de CRISP-DM** para evaluar imágenes de frutas o verduras y predecir de forma simultánea:
1. **Calidad**: Mala, Regular, Buena.
2. **Tamaño**: Pequeño, Mediano, Grande.
""")

# Menú lateral para opciones
st.sidebar.title("Configuración")
model_choice = st.sidebar.selectbox("Selecciona la arquitectura predictiva:", 
                                   ["Machine Learning Tradicional (RF)", "Deep Learning (CNN)"])

st.sidebar.markdown("---")
st.sidebar.info("**Proyecto Final - CRISP-DM**\n\n**Integrantes:**\n- Angy Hurtado\n- Hideki Tamura\n- David Vergara")

# Método de entrada
input_method = st.radio("Método de entrada de imagen:", ("Cargar archivo", "Usar cámara"))

uploaded_file = None
if input_method == "Cargar archivo":
    uploaded_file = st.file_uploader("Sube una imagen de la fruta/verdura", type=["jpg", "png", "jpeg"])
else:
    uploaded_file = st.camera_input("Toma una foto con tu cámara web")

# Funciones de carga de modelos con caché para no afectar rendimiento entre recargas
@st.cache_resource
def load_ml_model():
    if os.path.exists(ML_MODEL_PATH):
        return joblib.load(ML_MODEL_PATH)
    return None

@st.cache_resource
def load_dl_model():
    if os.path.exists(DL_MODEL_PATH):
        return tf.keras.models.load_model(DL_MODEL_PATH)
    # Fallback por si lo guardaron con extensión anterior
    alt_path = 'models/cnn_multioutput.h5'
    if os.path.exists(alt_path):
        return tf.keras.models.load_model(alt_path)
    return None

# Lógica principal de inferencia
if uploaded_file is not None:
    st.image(uploaded_file, caption='Vista previa de la imagen', use_container_width=True)
    
    if st.button("Predecir Simultáneamente"):
        with st.spinner("Analizando extracción de características y ejecutando inferencia..."):
            try:
                if model_choice == "Machine Learning Tradicional (RF)":
                    model = load_ml_model()
                    if model is None:
                        st.error(f"No se encontró el modelo. Asegúrate de ejecutar el notebook 02_experimentos_ml_tradicional.ipynb y que guarde en: `{ML_MODEL_PATH}`")
                    else:
                        # Extracción manual de características (HSV Histograma + Área de contorno)
                        features = process_image_from_streamlit(uploaded_file, model_type='ml')
                        
                        # Inferencia
                        predictions = model.predict(features)
                        
                        # Extraer ambas salidas
                        pred_quality_idx = int(predictions[0][0])
                        pred_size_idx = int(predictions[0][1])
                        
                        st.success("Análisis completado mediante Machine Learning Tradicional")
                        
                        # Mostrar resultados en columnas
                        col1, col2 = st.columns(2)
                        col1.metric("Calidad", QUALITY_CLASSES[pred_quality_idx])
                        col2.metric("Tamaño", SIZE_CLASSES[pred_size_idx])

                else: # Flujo para Deep Learning
                    model = load_dl_model()
                    if model is None:
                        st.error(f"No se encontró el modelo CNN. Asegúrate de ejecutar el notebook 03_experimentos_deep_learning.ipynb y que guarde en: `{DL_MODEL_PATH}`")
                    else:
                        # Preprocesamiento a Tensor 4D
                        img_tensor = process_image_from_streamlit(uploaded_file, model_type='dl')
                        
                        # Inferencia bifurcada (Devuelve lista [probs_calidad, probs_tamaño])
                        predictions = model.predict(img_tensor)
                        
                        # Aplicar Argmax para obtener el índice ganador
                        pred_quality_idx = np.argmax(predictions[0], axis=1)[0]
                        pred_size_idx = np.argmax(predictions[1], axis=1)[0]
                        
                        # Obtener porcentajes de confianza
                        conf_quality = np.max(predictions[0]) * 100
                        conf_size = np.max(predictions[1]) * 100
                        
                        st.success("Análisis completado mediante Red Neuronal Convolucional")
                        
                        # Mostrar resultados en columnas con confianzas
                        col1, col2 = st.columns(2)
                        col1.metric("Calidad", QUALITY_CLASSES[pred_quality_idx], f"{conf_quality:.1f}% Confianza")
                        col2.metric("Tamaño", SIZE_CLASSES[pred_size_idx], f"{conf_size:.1f}% Confianza")
                        
            except Exception as e:
                st.error(f"Ocurrió un error inesperado al procesar la imagen: {str(e)}")
