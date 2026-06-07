# Clasificador de Calidad y Tamaño de Frutas
**Proyecto Final — Algoritmos y Programación III · ICESI · Semestre 2026-1**

Equipo **sosaSlaves**: Angy Hurtado · Hideki Tamura · David Vergara

---

## Descripción

Sistema de clasificación multi-salida que predice simultáneamente:
- **Calidad**: Mala · Regular · Buena
- **Tamaño**: Pequeño · Mediano · Grande

Implementado con metodología **CRISP-DM**, dos familias de modelos (ML tradicional y CNN) y una interfaz web Streamlit (`app.py`).

---

## Estructura del proyecto

```
sosaSlaves/
├── app.py                                      # Interfaz Streamlit
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                                    # Imágenes originales (Kaggle + fotos propias)
│   └── processed/                              # Imágenes preprocesadas + metadata.csv
├── docs/
│   └── figures/                                # Figuras del EDA (fig1–fig7.pdf) y matrices de confusión
├── notebooks/
│   ├── 01_analisis_exploratorio_datos.ipynb
│   ├── 02_experimentos_ml_tradicional.ipynb
│   └── 03_experimentos_deep_learning.ipynb
├── saved_models/                               # Modelos entrenados exportados
│   ├── best_rf_multioutput.pkl
│   ├── random_forest_multioutput.pkl
│   ├── svm_lineal_multioutput.pkl
│   ├── xgboost_multioutput.pkl
│   └── cnn_multioutput.keras
└── src/
    ├── data_processing/
    │   └── preprocess.py
    ├── evaluation/
    │   └── evaluate.py
    └── utils/
        └── helpers.py
```

---

## Requisitos

- Python 3.10+
- CUDA opcional (para acelerar CNN en entrenamiento)

---

## Instalación

```bash
# 1. Clonar o descargar el repositorio
git clone https://github.com/LissaAN1/fruit-quality-classification.git
cd fruit-quality-classification

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución del pipeline completo

### Paso 1 — Preprocesar imágenes

Coloca las imágenes originales en `data/raw/` con la siguiente estructura:

```
data/raw/
├── Bad Quality_Fruits/
│   ├── Apple_Bad/
│   └── ...
├── Good Quality_Fruits/
│   └── ...
└── Regular Qualit_Fruits/
    └── ...
```

Luego ejecuta desde la **raíz del proyecto**:

```bash
python src/data_processing/preprocess.py
# Con opciones personalizadas:
python src/data_processing/preprocess.py --size 224 --augment 4
```

Esto genera `data/processed/` con las imágenes y `data/processed/metadata.csv`.

### Paso 2 — Análisis exploratorio (EDA)

```bash
cd notebooks
jupyter notebook 01_analisis_exploratorio_datos.ipynb
```

### Paso 3 — Entrenar modelos de ML tradicional

```bash
jupyter notebook 02_experimentos_ml_tradicional.ipynb
```

Entrena Random Forest, SVM lineal y XGBoost. Guarda los modelos en `saved_models/`.

### Paso 4 — Entrenar CNN (GPU recomendada)

```bash
jupyter notebook 03_experimentos_deep_learning.ipynb
```

Guarda el modelo en `saved_models/cnn_multioutput.keras`.

### Paso 5 — Ejecutar la aplicación web

```bash
# Desde la raíz del proyecto
streamlit run app.py
```

Abre `http://localhost:8501` en el navegador.

---

## Uso de la aplicación

1. Selecciona el modelo en el panel lateral (**ML Tradicional** o **CNN**).
2. Sube una imagen de fruta o toma una foto con la cámara.
3. Haz clic en **"Predecir Simultáneamente"**.
4. La app muestra la calidad y el tamaño estimados con porcentaje de confianza (CNN).

---

## Dataset

> ⚠️ **La carpeta `data/raw/` no está incluida en este repositorio** por el tamaño del dataset.
> Puedes descargarla desde el siguiente enlace de Google Drive:
>
> 📁 [Descargar data/raw/ — Google Drive](https://drive.google.com/drive/folders/1RINyjn96rrDTddl0RKnpiz-JC0aGIW8R?usp=sharing)
>
> Una vez descargada, coloca el contenido dentro de `data/raw/` respetando la estructura indicada en el Paso 1.

- **Fuente base**: [Fruit Quality Classification — Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification)
- **Clase Regular**: construida manualmente desde `mix quality_fruits` + fotos propias del equipo
- **Frutas**: Apple · Banana · Guava · Lime · Orange · Pomegranate
- **Total imágenes originales**: 28 395 (Bad: 9 142 · Good: 14 599 · Regular: 4 654)
- **Aumentadas (×4)**: 113 580

---

## Notas técnicas

- La normalización usa **z-score de ImageNet** (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]) en todos los módulos (`preprocess.py`, `helpers.py`, notebooks).
- El split train/test usa **solo imágenes originales** (`augmented == 0`) para el conjunto de prueba, evitando contaminación de datos.
- Se aplica `class_weight='balanced'` en todos los modelos para compensar el desbalanceo (IR ≈ 3.14 entre Good y Regular).
- El mejor modelo tradicional fue **XGBoost** (F1-macro promedio más alto).
- La CNN alcanzó **78% accuracy en calidad** y **93% en tamaño** tras 41 épocas con Early Stopping.
