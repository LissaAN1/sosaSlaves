import os
import cv2
import numpy as np
from PIL import Image

def load_image_for_dl(image_path, target_size=(128, 128)):
    """
    Carga y preprocesa una imagen desde una ruta local para el modelo de Deep Learning (CNN).
    Retorna la imagen redimensionada y normalizada en un tensor de forma (1, W, H, C).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen en: {image_path}")
        
    # OpenCV carga en BGR, convertimos a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar a la entrada requerida por la CNN
    img = cv2.resize(img, target_size)
    
    # Normalización para redes neuronales [0, 1]
    img = img.astype('float32') / 255.0
    
    # Añadir la dimensión del batch (1, 128, 128, 3) para predicciones individuales
    img_tensor = np.expand_dims(img, axis=0)
    return img_tensor

def extract_features_for_ml(image_path):
    """
    Extrae características (features) de una imagen local para modelos de ML tradicional.
    Extrae el histograma de color 3D y el área del contorno principal.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen en: {image_path}")
        
    # --- 1. Características de Color (Histograma HSV 3D) ---
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Calculamos el histograma en los canales H, S, V (8 bins por canal = 512 features)
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    color_features = hist.flatten()
    
    # --- 2. Características de Forma/Tamaño (Área del objeto principal) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Umbral de Otsu para separar el objeto del fondo (asumiendo fondos simples)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Encontrar el contorno más grande asumiendo que es la fruta/verdura
    max_area = 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        max_area = cv2.contourArea(c)
        
    shape_features = np.array([max_area])
    
    # Concatenar características en un solo vector (513 dimensiones en total)
    combined_features = np.concatenate([color_features, shape_features])
    
    # Devolver como (1, n_features) para que scikit-learn lo acepte en .predict()
    return combined_features.reshape(1, -1)

def process_image_from_streamlit(uploaded_file, model_type='dl', target_size=(128, 128)):
    """
    Toma un archivo subido a través de Streamlit (st.file_uploader o st.camera_input)
    y lo procesa dependiendo del tipo de modelo a evaluar.
    
    Args:
        uploaded_file: Archivo en memoria retornado por Streamlit.
        model_type: 'dl' para Deep Learning o 'ml' para Machine Learning Tradicional.
        target_size: Tupla con la resolución de entrada para la CNN.
    """
    # Leer la imagen desde memoria usando PIL
    image = Image.open(uploaded_file)
    # Convertir a arreglo numpy en formato RGB
    img_array = np.array(image.convert('RGB'))
    
    if model_type == 'dl':
        # Procesamiento equivalente a load_image_for_dl()
        img_resized = cv2.resize(img_array, target_size)
        img_normalized = img_resized.astype('float32') / 255.0
        return np.expand_dims(img_normalized, axis=0)
        
    elif model_type == 'ml':
        # Procesamiento equivalente a extract_features_for_ml()
        # Convertimos a BGR para aplicar los métodos de OpenCV de manera consistente
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Color Features
        hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        color_features = hist.flatten()
        
        # Shape Features
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        if contours:
            c = max(contours, key=cv2.contourArea)
            max_area = cv2.contourArea(c)
            
        shape_features = np.array([max_area])
        combined_features = np.concatenate([color_features, shape_features])
        return combined_features.reshape(1, -1)
        
    else:
        raise ValueError("model_type debe ser estrictamente 'dl' o 'ml'")
