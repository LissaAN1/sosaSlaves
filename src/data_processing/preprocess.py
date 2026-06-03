"""
preprocess.py
=============
Fruit Quality Classification
Algoritmos y Programación III · ICESI · Semestre 2026-1

Responsabilidades:
  - Carga de imágenes desde data/raw/ (estructura: Bad/Good/Regular Quality_Fruits)
  - Remoción de fondo mediante segmentación HSV + operaciones morfológicas
  - Reescalado con padding y funciones auxiliares de normalización matemática (z-score y min-max)
  - Data Augmentation reproducible
  - Guardado de imágenes procesadas en data/processed/
  - Generación de metadata.csv con rutas, etiquetas y tamaños

Uso:
  python preprocess.py                          # Procesa todo con parámetros por defecto
  python preprocess.py --size 128 --augment 3  # Tamaño 128x128, 3 aumentos por imagen
  python preprocess.py --no-bg-remove          # Omite remoción de fondo (más rápido)

Estructura esperada de data/raw/:
  data/raw/
  ├── Bad Quality_Fruits/
  │   ├── Apple_Bad/
  │   ├── Banana_Bad/
  │   └── ...
  ├── Good Quality_Fruits/
  │   ├── Apple_Good/
  │   └── ...
  └── Regular Qualit_Fruits/  
      ├── Apple_Regular/
      └── ...

Nota:
  - Si existe una carpeta como Mix Quality_Fruits, no se procesa porque el
    problema está definido con tres clases: Bad, Good y Regular.
  - Las imágenes JPG guardadas conservan la versión visual procesada
    (fondo/remoción, resize y augmentation). La normalización numérica debe
    aplicarse durante el entrenamiento o guardarse aparte como arreglo .npy.
"""

import os
import argparse
import warnings
import math
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES Y CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

# Mapeo robusto de nombres de carpeta → etiqueta numérica y texto
QUALITY_MAP = {
    "bad":     {"label": 0, "name": "Bad"},
    "good":    {"label": 1, "name": "Good"},
    "regular": {"label": 2, "name": "Regular"},
}

CLASS_ORDER = ["Bad", "Good", "Regular"]

# Semilla de aleatoriedad para reproducibilidad del experimento
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Rutas por defecto relativas al script
DEFAULT_RAW_DIR       = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_IMG_SIZE      = 224          # Tamaño estándar compatible con CNNs (224×224)
DEFAULT_AUGMENTS      = 4            # Imágenes aumentadas por original


# ─────────────────────────────────────────────────────────────────────────────
# 1. DETECCIÓN DE CALIDAD DESDE EL NOMBRE DE CARPETA
# ─────────────────────────────────────────────────────────────────────────────

def detect_quality(folder_name: str) -> dict | None:
    """
    Infiere la clase de calidad a partir del nombre de la carpeta padre.

    Ejemplos:
      'Bad Quality_Fruits'        → {"label": 0, "name": "Bad"}
      'Good Quality_Fruits'       → {"label": 1, "name": "Good"}
      'Regular Qualit_Fruits'     → {"label": 2, "name": "Regular"}

    Retorna None si el nombre no es reconocido.
    """
    name_lower = folder_name.lower()
    for key, value in QUALITY_MAP.items():
        if key in name_lower:
            return value
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. REMOCIÓN DE FONDO
# ─────────────────────────────────────────────────────────────────────────────

def remove_background_hsv(img_bgr: np.ndarray) -> np.ndarray:
    """
    Segmenta la fruta del fondo usando umbralización HSV + morfología.

    Pasos matemáticos:
      1. Conversión BGR → HSV:  H = arctan(G-B / R-B) · (60/π)
                                S = (max - min) / max
                                V = max(R, G, B)
      2. Máscara de fondo claro (fondos blancos / uniformes):
         pixel es fondo si S < S_thresh  AND  V > V_thresh
      3. Máscara invertida → región de la fruta
      4. Operaciones morfológicas para cerrar huecos y eliminar ruido:
         - Dilatación: δ(A) = A ⊕ B  (expande la región)
         - Erosión:    ε(A) = A ⊖ B  (contrae)
         - Cierre:     A • B = δ(ε(A)) (rellena huecos)
      5. Aplicación de máscara: pixel_out = pixel_in if mask else 255 (blanco)

    Args:
        img_bgr: Imagen de entrada en formato BGR (uint8).

    Returns:
        Imagen BGR con el fondo reemplazado por blanco.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Parámetros de umbral (ajustables según el dataset)
    S_THRESH = 30    # Saturación mínima para considerar un pixel "de color" (fruta)
    V_THRESH = 200   # Valor máximo para fondos claros

    # Máscara booleana: 1 = fruta, 0 = fondo
    mask_saturation = img_hsv[:, :, 1] > S_THRESH
    mask_brightness  = img_hsv[:, :, 2] < V_THRESH
    mask_fruit = (mask_saturation | mask_brightness).astype(np.uint8) * 255

    # Kernel morfológico 5×5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_fruit = cv2.morphologyEx(mask_fruit, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_fruit = cv2.morphologyEx(mask_fruit, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Relleno de agujeros con floodFill desde las esquinas
    mask_filled = mask_fruit.copy()
    h, w = mask_filled.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask_filled, flood_mask, (0, 0), 255)
    mask_holes = cv2.bitwise_not(mask_filled)
    mask_fruit = cv2.bitwise_or(mask_fruit, mask_holes)

    # Aplicar máscara: fondo → blanco (255, 255, 255)
    result = img_bgr.copy()
    result[mask_fruit == 0] = [255, 255, 255]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. REESCALADO Y NORMALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def resize_with_padding(img_bgr: np.ndarray, target_size: int) -> np.ndarray:
    """
    Reescala la imagen manteniendo la proporción original (aspect ratio)
    y añade padding blanco para alcanzar target_size × target_size.

    Fórmula de escala:
        scale = target_size / max(alto, ancho)
        nuevo_alto = round(alto × scale)
        nuevo_ancho = round(ancho × scale)

    El padding se distribuye simétricamente:
        pad_top    = (target_size - nuevo_alto) // 2
        pad_bottom = target_size - nuevo_alto - pad_top
        pad_left   = (target_size - nuevo_ancho) // 2
        pad_right  = target_size - nuevo_ancho - pad_left

    Args:
        img_bgr:     Imagen BGR de cualquier tamaño.
        target_size: Lado del cuadrado de salida en píxeles.

    Returns:
        Imagen BGR de tamaño (target_size, target_size).
    """
    h, w = img_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_h = round(h * scale)
    new_w = round(w * scale)

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top    = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left   = (target_size - new_w) // 2
    pad_right  = target_size - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    return padded


def normalize_minmax(img_bgr: np.ndarray) -> np.ndarray:
    """
    Normalización Min-Max por canal: x' = (x - x_min) / (x_max - x_min)

    Escala cada canal independientemente al rango [0, 1].
    Útil como preprocesamiento previo antes de alimentar modelos ML.

    Args:
        img_bgr: Imagen uint8 [0, 255].

    Returns:
        Array float32 en [0.0, 1.0].
    """
    img_float = img_bgr.astype(np.float32)
    result = np.zeros_like(img_float)
    for c in range(3):
        channel = img_float[:, :, c]
        c_min, c_max = channel.min(), channel.max()
        if c_max - c_min > 1e-8:
            result[:, :, c] = (channel - c_min) / (c_max - c_min)
        else:
            result[:, :, c] = 0.0
    return result


def normalize_zscore(img_bgr: np.ndarray,
                     mean: tuple = (0.485, 0.456, 0.406),
                     std:  tuple = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Normalización Z-Score (estandarización) por canal:
        x' = (x - μ) / σ

    Usa los valores estándar de ImageNet (mean/std en RGB) que son
    los más comunes al usar transfer learning con CNNs preentrenadas.

    Orden de canales: la imagen se convierte internamente a RGB para
    aplicar la normalización y se regresa a BGR para compatibilidad con OpenCV.

    Args:
        img_bgr: Imagen uint8 BGR [0, 255].
        mean:    Media por canal (R, G, B).
        std:     Desviación estándar por canal (R, G, B).

    Returns:
        Array float32 normalizado con z-score.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    result = np.zeros_like(img_rgb)
    for c in range(3):
        result[:, :, c] = (img_rgb[:, :, c] - mean[c]) / std[c]
    # Regresa a BGR
    return result[:, :, ::-1]


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def augment_image(img_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Genera variantes aumentadas de una imagen de forma determinista pero variada.

    Transformaciones aplicadas (una combinación aleatoria por variante):
      - Flip horizontal:  refleja en el eje Y   → diversidad de orientación
      - Rotación:         ángulo ∈ {-30°, +30°} → invarianza rotacional
      - Brillo:           factor ∈ [0.6, 1.4]   → condiciones de iluminación
      - Contraste:        factor ∈ [0.7, 1.3]   → variabilidad de cámara
      - Zoom (recorte):   crop_ratio ∈ [0.8, 1] → variación de encuadre
      - Ruido gaussiano:  σ = 8                 → robustez frente a ruido

    Args:
        img_bgr: Imagen BGR de tamaño fijo (ya reescalada).

    Returns:
        Lista de imágenes aumentadas (uint8, BGR).
    """
    augmented = []
    h, w = img_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # ── Augmentation 1: Flip horizontal + leve rotación
    aug1 = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    angle1 = random.uniform(-20, 20)
    aug1 = aug1.rotate(angle1, fillcolor=(255, 255, 255), expand=False)
    augmented.append(cv2.cvtColor(np.array(aug1), cv2.COLOR_RGB2BGR))

    # ── Augmentation 2: Cambio de brillo y contraste
    bright_factor   = random.uniform(0.65, 1.35)
    contrast_factor = random.uniform(0.7, 1.30)
    aug2 = ImageEnhance.Brightness(pil_img).enhance(bright_factor)
    aug2 = ImageEnhance.Contrast(aug2).enhance(contrast_factor)
    augmented.append(cv2.cvtColor(np.array(aug2), cv2.COLOR_RGB2BGR))

    # ── Augmentation 3: Zoom (recorte central + resize)
    crop_ratio = random.uniform(0.75, 0.95)
    cx, cy = w // 2, h // 2
    crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
    x1 = max(cx - crop_w // 2, 0)
    y1 = max(cy - crop_h // 2, 0)
    x2 = min(x1 + crop_w, w)
    y2 = min(y1 + crop_h, h)
    aug3_np = img_bgr[y1:y2, x1:x2]
    aug3 = cv2.resize(aug3_np, (w, h), interpolation=cv2.INTER_LINEAR)
    augmented.append(aug3)

    # ── Augmentation 4: Ruido gaussiano aditivo
    noise = np.random.normal(0, 8, img_bgr.shape).astype(np.float32)
    aug4 = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    augmented.append(aug4)

    return augmented


# ─────────────────────────────────────────────────────────────────────────────
# 5. ESTIMACIÓN DE TAMAÑO RELATIVO
# ─────────────────────────────────────────────────────────────────────────────

def estimate_size(img_bgr: np.ndarray) -> dict:
    """
    Estima el tamaño relativo de la fruta en la imagen usando el área de
    la máscara binaria de la fruta respecto al área total de la imagen.

    Métrica:
        area_ratio = n_pixels_fruta / (alto × ancho)

    Clasificación:
        area_ratio < 0.20  → "small"
        area_ratio < 0.50  → "medium"
        area_ratio ≥ 0.50  → "large"

    También calcula el diámetro equivalente en píxeles normalizados:
        d_norm = sqrt(4 × área_fruta / π) / max(alto, ancho)

    Args:
        img_bgr: Imagen BGR (con fondo blanco idealmente).

    Returns:
        Diccionario con keys: 'size_category', 'area_ratio', 'diameter_norm'.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = (img_hsv[:, :, 1] > 30).astype(np.uint8)

    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    fruit_pixels = int(mask.sum())
    area_ratio   = fruit_pixels / total_pixels

    # Diámetro equivalente de un círculo con la misma área (en píxeles normalizados)
    d_norm = math.sqrt(4 * fruit_pixels / math.pi) / max(img_bgr.shape[:2])

    if area_ratio < 0.20:
        size_cat = "small"
    elif area_ratio < 0.50:
        size_cat = "medium"
    else:
        size_cat = "large"

    return {
        "size_category": size_cat,
        "area_ratio":    round(area_ratio, 4),
        "diameter_norm": round(d_norm, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(
    raw_dir:       Path,
    processed_dir: Path,
    img_size:      int  = DEFAULT_IMG_SIZE,
    n_augments:    int  = DEFAULT_AUGMENTS,
    remove_bg:     bool = True,
) -> None:
    """
    Pipeline completo de preprocesamiento.

    Flujo por imagen:
      raw_img → [remove_background] → resize_with_padding → augmentations
              → guardar imagen visual en processed/ → registrar en metadata.csv

    Nota sobre normalización:
      Las funciones normalize_minmax() y normalize_zscore() quedan disponibles
      para la etapa de entrenamiento. No se guardan imágenes JPG normalizadas
      como float, porque JPG no conserva directamente valores numéricos [0,1]
      ni z-score.

    Args:
        raw_dir:       Ruta a data/raw/ con subcarpetas de calidad.
        processed_dir: Ruta de salida data/processed/.
        img_size:      Lado del cuadrado de salida (píxeles).
        n_augments:    Número de variantes aumentadas a generar por imagen original.
        remove_bg:     Si True, aplica remoción de fondo HSV.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = processed_dir / "metadata.csv"

    # Recopilar todas las imágenes
    all_images: list[tuple[Path, int, str, str]] = []  # (ruta, label, quality_name, fruit_type)

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for quality_folder in sorted(raw_dir.iterdir()):
        if not quality_folder.is_dir():
            continue
        quality_info = detect_quality(quality_folder.name)
        if quality_info is None:
            print(f"  [WARN] Carpeta no reconocida: {quality_folder.name} — omitida.")
            continue

        for fruit_folder in sorted(quality_folder.iterdir()):
            if not fruit_folder.is_dir():
                continue
            fruit_type = fruit_folder.name.split("_")[0]  # "Apple", "Banana", etc.

            for img_path in sorted(fruit_folder.iterdir()):
                if img_path.suffix.lower() in supported_exts:
                    all_images.append((
                        img_path,
                        quality_info["label"],
                        quality_info["name"],
                        fruit_type,
                    ))

    if not all_images:
        print(f"\n[ERROR] No se encontraron imágenes en {raw_dir}")
        print("  Verifica que la estructura sea:")
        print("  data/raw/Bad Quality_Fruits/Apple_Bad/*.jpg")
        return

    raw_counts = {quality: 0 for quality in CLASS_ORDER}
    for _, _, quality_name, _ in all_images:
        raw_counts[quality_name] = raw_counts.get(quality_name, 0) + 1

    missing_classes = [q for q, count in raw_counts.items() if count == 0]
    if missing_classes:
        print(f"\n[WARN] No se encontraron imágenes para: {missing_classes}")
        print("       Revisa data/raw/ antes de entrenar, porque el problema es de 3 clases.")

    print(f"\n{'='*60}")
    print(f"  Fruit Quality Classification — Preprocesamiento")
    print(f"{'='*60}")
    print(f"  Imágenes crudas encontradas : {len(all_images)}")
    print(f"  Tamaño de salida            : {img_size}×{img_size} px")
    print(f"  Aumentos por imagen         : {n_augments}")
    print(f"  Remoción de fondo           : {'Sí (HSV)' if remove_bg else 'No'}")
    print(f"  Salida                      : {processed_dir}\n")

    metadata_rows = []
    errors = 0

    for img_path, label, quality_name, fruit_type in tqdm(all_images, desc="Procesando"):
        try:
            # ── Leer imagen
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise ValueError(f"No se pudo leer: {img_path}")

            # ── Remoción de fondo
            if remove_bg:
                img_bgr = remove_background_hsv(img_bgr)

            # ── Reescalado con padding
            img_resized = resize_with_padding(img_bgr, img_size)

            # ── Estimación de tamaño
            size_info = estimate_size(img_resized)

            # ── Directorio de salida: processed/Bad/Apple/
            out_dir = processed_dir / quality_name / fruit_type
            out_dir.mkdir(parents=True, exist_ok=True)

            # ── Guardar imagen original procesada
            stem = img_path.stem
            out_path = out_dir / f"{stem}_processed.jpg"
            cv2.imwrite(str(out_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            metadata_rows.append({
                "file":          str(out_path.relative_to(processed_dir)),
                "label":         label,
                "quality":       quality_name,
                "fruit":         fruit_type,
                "augmented":     0,
                "aug_index":     -1,
                **size_info,
            })

            # ── Data Augmentation
            if n_augments > 0:
                augmented_imgs = augment_image(img_resized)
                for aug_idx, aug_img in enumerate(augmented_imgs[:n_augments]):
                    aug_path = out_dir / f"{stem}_aug{aug_idx}.jpg"
                    cv2.imwrite(str(aug_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    aug_size_info = estimate_size(aug_img)
                    metadata_rows.append({
                        "file":      str(aug_path.relative_to(processed_dir)),
                        "label":     label,
                        "quality":   quality_name,
                        "fruit":     fruit_type,
                        "augmented": 1,
                        "aug_index": aug_idx,
                        **aug_size_info,
                    })

        except Exception as exc:
            tqdm.write(f"  [ERROR] {img_path.name}: {exc}")
            errors += 1

    # ── Guardar metadata
    if metadata_rows:
        fieldnames = [
            "file", "label", "quality", "fruit",
            "augmented", "aug_index",
            "size_category", "area_ratio", "diameter_norm",
        ]
        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_rows)

    # ── Resumen
    originals  = sum(1 for r in metadata_rows if r["augmented"] == 0)
    augmented  = sum(1 for r in metadata_rows if r["augmented"] == 1)
    by_quality = {}
    for r in metadata_rows:
        if r["augmented"] == 0:
            by_quality[r["quality"]] = by_quality.get(r["quality"], 0) + 1

    print(f"\n{'='*60}")
    print(f"   Preprocesamiento completado")
    print(f"{'='*60}")
    print(f"  Imágenes originales procesadas : {originals}")
    print(f"  Imágenes aumentadas generadas  : {augmented}")
    print(f"  Total en dataset final          : {originals + augmented}")
    print(f"  Errores                         : {errors}")
    print(f"\n  Distribución de clases (originales):")
    max_count = max(by_quality.values()) if by_quality else 1
    for quality in CLASS_ORDER:
        count = by_quality.get(quality, 0)
        bar = "█" * (count // max(1, max_count // 20))
        print(f"    {quality:10s}: {count:5d}  {bar}")

    missing_final = [q for q in CLASS_ORDER if by_quality.get(q, 0) == 0]
    if missing_final:
        print(f"\n  [WARN] Dataset procesado incompleto. Faltan clases: {missing_final}")
    print(f"\n  Metadata guardada en: {metadata_path}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de imágenes para Fruit Quality Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
        help="Directorio con imágenes crudas (data/raw/)"
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR,
        help="Directorio de salida (data/processed/)"
    )
    parser.add_argument(
        "--size", type=int, default=DEFAULT_IMG_SIZE,
        help="Tamaño de salida en píxeles (NxN)"
    )
    parser.add_argument(
        "--augment", type=int, default=DEFAULT_AUGMENTS,
        help="Número de aumentos por imagen (0 = desactivado)"
    )
    parser.add_argument(
        "--no-bg-remove", action="store_true",
        help="Omite la remoción de fondo (útil para debug rápido)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        raw_dir       = args.raw_dir,
        processed_dir = args.processed_dir,
        img_size      = args.size,
        n_augments    = args.augment,
        remove_bg     = not args.no_bg_remove,
    )
