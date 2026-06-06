"""
helpers.py
==========
Fruit Quality Classification — src/utils/
Algoritmos y Programación III · ICESI · Semestre 2026-1

Funciones auxiliares compartidas entre los notebooks de experimentación
(02_experimentos_ml_tradicional.ipynb, 03_experimentos_deep_learning.ipynb)
y la aplicación de despliegue (app.py).

Responsabilidades principales:
  - load_image_for_dl()          → preprocesa una imagen local para la CNN
  - extract_features_for_ml()    → extrae vector de características para ML tradicional
  - process_image_from_streamlit() → mismo procesamiento pero con entrada desde Streamlit

IMPORTANTE — consistencia de normalización:
  La función normalize_zscore() replica los parámetros de ImageNet usados en
  preprocess.py (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
  Esto garantiza que el modelo CNN vea en inferencia exactamente el mismo
  rango de valores numéricos que vio durante el entrenamiento.
"""

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES — deben coincidir con preprocess.py
# ─────────────────────────────────────────────────────────────────────────────

# Parámetros de normalización z-score de ImageNet (canal R, G, B)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Tamaño de entrada por defecto (debe coincidir con el entrenamiento)
DEFAULT_IMG_SIZE = (224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# 1. NORMALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def normalize_zscore(img_rgb_float: np.ndarray) -> np.ndarray:
    """
    Aplica normalización z-score por canal RGB con parámetros de ImageNet.

    Fórmula (por canal c):
        x'_c = (x_c - μ_c) / σ_c

    Parámetros ImageNet:
        μ = (0.485, 0.456, 0.406)   — media por canal
        σ = (0.229, 0.224, 0.225)   — desviación estándar por canal

    Args:
        img_rgb_float: Array float32 en [0.0, 1.0], shape (H, W, 3), canal orden RGB.

    Returns:
        Array float32 normalizado, mismo shape.
    """
    return (img_rgb_float - _IMAGENET_MEAN) / _IMAGENET_STD


# ─────────────────────────────────────────────────────────────────────────────
# 2. CARGA PARA DEEP LEARNING
# ─────────────────────────────────────────────────────────────────────────────

def load_image_for_dl(
    image_path: str,
    target_size: tuple = DEFAULT_IMG_SIZE,
) -> np.ndarray:
    """
    Carga y preprocesa una imagen local para inferencia con la CNN.

    Pipeline:
        1. Lectura BGR con OpenCV → conversión a RGB
        2. Redimensionado a target_size (H, W)
        3. Cast a float32 y escala [0, 255] → [0.0, 1.0]
        4. Normalización z-score con parámetros ImageNet (igual que preprocess.py)
        5. Adición de dimensión batch → shape (1, H, W, 3)

    Args:
        image_path:  Ruta al archivo de imagen.
        target_size: Tupla (alto, ancho) requerida por la CNN.

    Returns:
        Array float32 de shape (1, H, W, 3) listo para model.predict().

    Raises:
        ValueError: Si OpenCV no puede leer el archivo.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen en: {image_path}")

    # BGR → RGB (OpenCV lee en BGR, TensorFlow/Keras espera RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Redimensionado — cv2.resize toma (ancho, alto), target_size es (alto, ancho)
    img_resized = cv2.resize(img_rgb, (target_size[1], target_size[0]),
                             interpolation=cv2.INTER_AREA)

    # Escalar a [0.0, 1.0] y normalizar con z-score de ImageNet
    img_float = img_resized.astype(np.float32) / 255.0
    img_norm  = normalize_zscore(img_float)

    # Añadir dimensión de batch: (H, W, C) → (1, H, W, C)
    return np.expand_dims(img_norm, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. EXTRACCIÓN DE CARACTERÍSTICAS PARA ML TRADICIONAL
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_for_ml(image_path: str) -> np.ndarray:
    """
    Extrae un vector de características de una imagen local para modelos de
    Machine Learning tradicional (Random Forest, SVM, KNN, etc.).

    Características extraídas (513 dimensiones totales):
        - Histograma de color HSV 3D (8×8×8 = 512 bins), normalizado L2.
          Captura la distribución cromática de la fruta independientemente
          de la posición o rotación.
        - Área del contorno principal (1 valor), normalizada por el área total
          de la imagen. Representa el tamaño relativo de la fruta, equivalente
          al area_ratio calculado por preprocess.py.

    Args:
        image_path: Ruta al archivo de imagen.

    Returns:
        Array float32 de shape (1, 513) compatible con sklearn.predict().

    Raises:
        ValueError: Si OpenCV no puede leer el archivo.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen en: {image_path}")

    # ── Característica 1: histograma de color HSV (512 bins)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [8, 8, 8],
        [0, 180, 0, 256, 0, 256],
    )
    cv2.normalize(hist, hist)          # Normalización L2 — invariante al brillo global
    color_features = hist.flatten()    # shape (512,)

    # ── Característica 2: área relativa de la fruta (1 valor)
    #    Se usa umbral de Otsu sobre la escala de grises para separar la fruta
    #    del fondo uniforme.  area_ratio = píxeles_fruta / (H × W)
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    max_area = 0.0
    if contours:
        max_area = cv2.contourArea(max(contours, key=cv2.contourArea))

    # Normalizar por el área total de la imagen → rango [0, 1]
    area_ratio = max_area / (h * w) if (h * w) > 0 else 0.0
    shape_features = np.array([area_ratio], dtype=np.float32)  # shape (1,)

    # Concatenar → (513,) → reshape para sklearn → (1, 513)
    combined = np.concatenate([color_features, shape_features])
    return combined.reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROCESAMIENTO DESDE STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

def process_image_from_streamlit(
    uploaded_file,
    model_type: str = "dl",
    target_size: tuple = DEFAULT_IMG_SIZE,
) -> np.ndarray:
    """
    Procesa un archivo subido a través de Streamlit (st.file_uploader o
    st.camera_input) y produce el tensor/vector listo para inferencia.

    Replica internamente el mismo pipeline que load_image_for_dl() y
    extract_features_for_ml(), garantizando consistencia con el entrenamiento.

    Args:
        uploaded_file: Objeto BytesIO retornado por Streamlit.
        model_type:    'dl' → tensor para CNN | 'ml' → vector para ML tradicional.
        target_size:   Tupla (alto, ancho) para la CNN.

    Returns:
        - model_type='dl': Array float32 de shape (1, H, W, 3).
        - model_type='ml': Array float32 de shape (1, 513).

    Raises:
        ValueError: Si model_type no es 'dl' ni 'ml'.
    """
    # Leer desde memoria usando PIL (acepta cualquier formato soportado)
    pil_image = Image.open(uploaded_file)
    img_rgb = np.array(pil_image.convert("RGB"), dtype=np.uint8)  # (H, W, 3), RGB

    if model_type == "dl":
        # ── Pipeline DL: resize → [0,1] → z-score → batch
        img_resized = cv2.resize(
            img_rgb, (target_size[1], target_size[0]),
            interpolation=cv2.INTER_AREA,
        )
        img_float = img_resized.astype(np.float32) / 255.0
        img_norm  = normalize_zscore(img_float)
        return np.expand_dims(img_norm, axis=0)   # (1, H, W, 3)

    elif model_type == "ml":
        # ── Pipeline ML: BGR → histograma HSV + área
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)

        # Histograma HSV (512 bins)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [8, 8, 8],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist)
        color_features = hist.flatten()

        # Área relativa
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        max_area = 0.0
        if contours:
            max_area = cv2.contourArea(max(contours, key=cv2.contourArea))

        h, w = img_bgr.shape[:2]
        area_ratio    = max_area / (h * w) if (h * w) > 0 else 0.0
        shape_features = np.array([area_ratio], dtype=np.float32)

        combined = np.concatenate([color_features, shape_features])
        return combined.reshape(1, -1)             # (1, 513)

    else:
        raise ValueError("model_type debe ser 'dl' o 'ml'.")
