# Clasificador de Calidad y Tamaño de Frutas

**Proyecto Final — Algoritmos y Programación III · ICESI · Semestre 2026-1**

Equipo **sosaSlaves**: Angy Hurtado · Hideki Tamura · David Vergara

---

## Descripción

Sistema de clasificación multi-salida que predice simultáneamente:

- **Calidad**: Mala · Regular · Buena
- **Tamaño**: Pequeño · Mediano · Grande

El proyecto usa metodología **CRISP-DM**, modelos de **Machine Learning tradicional** y **Deep Learning**, además de una interfaz web desarrollada con **Streamlit** (`app.py`).

---

## Estructura del proyecto

```text
sosaSlaves/
├── app.py                                      # Interfaz Streamlit
├── requirements.txt                           # Dependencias del proyecto
├── README.md                                  # Instrucciones de instalación y ejecución
├── data/
│   ├── raw/                                   # Imágenes originales, no incluidas en Git
│   └── processed/                             # Imágenes preprocesadas + metadata.csv, no incluidas en Git
├── docs/
│   └── figures/                               # Figuras del EDA y matrices de confusión
├── notebooks/
│   ├── 01_analisis_exploratorio_datos.ipynb
│   ├── 02_experimentos_ml_tradicional.ipynb
│   └── 03_experimentos_deep_learning.ipynb
├── saved_models/                              # Modelos entrenados exportados
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

## Requisitos recomendados

- **Python 3.10 o 3.11**
- **Windows, Linux o macOS**
- **16 GB de RAM recomendado** para entrenamiento completo
- **GPU opcional** para entrenar la CNN

> En Windows nativo, TensorFlow moderno puede no usar GPU aunque CUDA esté instalado. Para entrenar CNN con GPU NVIDIA se recomienda usar **Google Colab**, **WSL2 con Ubuntu** o un equipo Linux con GPU configurada.

---

## Instalación recomendada

Se recomienda usar un entorno virtual para evitar conflictos con paquetes globales de Python.

### 1. Clonar o descargar el repositorio

```bash
git clone https://github.com/LissaAN1/fruit-quality-classification.git
cd fruit-quality-classification
```


### 2. Crear entorno virtual

En **Windows PowerShell**:

```powershell
python -m venv .venv
```

Activar el entorno:

```powershell
.\.venv\Scripts\activate
```

Si PowerShell bloquea la activación, ejecutar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\activate
```

En **Linux/macOS**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Actualizar herramientas base

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4. Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

---

## Ejecutar la aplicación web

Los modelos entrenados deben estar en la carpeta `saved_models/`.

Desde la raíz del proyecto, ejecutar:

```bash
python -m streamlit run app.py
```

Luego abrir en el navegador:

```text
http://localhost:8501
```

> Se recomienda usar `python -m streamlit run app.py` en lugar de `streamlit run app.py`, porque en Windows a veces el ejecutable `streamlit.exe` no queda agregado al PATH.

> Tambien puedes ejecutar la app con más detalle;

```bash
python -m streamlit run app.py --logger.level=debug
```

---

## Uso de la aplicación

1. Seleccionar el modelo en el panel lateral:
   - **ML Tradicional**
   - **CNN**
2. Subir una imagen de fruta o tomar una foto con la cámara.
3. Presionar **"Predecir Simultáneamente"**.
4. La aplicación muestra:
   - Calidad estimada
   - Tamaño estimado
   - Porcentaje de confianza, cuando aplique

---

## Dataset

> La carpeta `data/raw/` no está incluida en este repositorio por el tamaño del dataset.

Descargar el dataset desde Google Drive:

```text
https://drive.google.com/drive/folders/1RINyjn96rrDTddl0RKnpiz-JC0aGIW8R?usp=sharing
```

Colocar las imágenes dentro de `data/raw/` respetando esta estructura:

```text
data/raw/
├── Bad Quality_Fruits/
│   ├── Apple_Bad/
│   └── ...
├── Good Quality_Fruits/
│   └── ...
└── Regular Qualit_Fruits/
    └── ...
```

Información del dataset:

- **Fuente base**: Fruit Quality Classification — Kaggle
- **Clase Regular**: construida manualmente desde `mix quality_fruits` + fotos propias del equipo
- **Frutas**: Apple · Banana · Guava · Lime · Orange · Pomegranate
- **Total imágenes originales**: 28 395
  - Bad: 9 142
  - Good: 14 599
  - Regular: 4 654
- **Imágenes aumentadas**: 113 580

---

## Reproducir el entrenamiento completo

Este paso es opcional. Solo debe hacerse si se quiere reentrenar los modelos desde cero.

### Paso 1 — Preprocesar imágenes

Desde la raíz del proyecto:

```bash
python src/data_processing/preprocess.py
```

Con opciones personalizadas:

```bash
python src/data_processing/preprocess.py --size 224 --augment 4
```

Esto genera:

```text
data/processed/
data/processed/metadata.csv
```

### Paso 2 — Ejecutar análisis exploratorio

```bash
cd notebooks
jupyter notebook 01_analisis_exploratorio_datos.ipynb
```

### Paso 3 — Entrenar modelos tradicionales

```bash
jupyter notebook 02_experimentos_ml_tradicional.ipynb
```

Este notebook entrena y compara:

- Random Forest
- SVM lineal
- XGBoost

Los modelos se guardan en:

```text
saved_models/
```

### Paso 4 — Entrenar CNN

```bash
jupyter notebook 03_experimentos_deep_learning.ipynb
```

La CNN es el entrenamiento más pesado. Para ejecutarla completa se recomienda:

- Equipo con suficiente espacio libre
- 16 GB de RAM o más
- GPU o Google Colab

Si se ejecuta en CPU, puede tardar mucho o reiniciar el kernel por falta de memoria.

---

## Notas técnicas

- La normalización de imágenes usa **z-score de ImageNet**:
  - Media: `[0.485, 0.456, 0.406]`
  - Desviación estándar: `[0.229, 0.224, 0.225]`
- El split train/test usa únicamente imágenes originales (`augmented == 0`) para el conjunto de prueba.
- Esto evita contaminación entre imágenes originales y aumentadas.
- Para el desbalance de clases se usa:
  - `class_weight='balanced'` en Random Forest y SVM
  - `sample_weight` o pesos equivalentes en XGBoost/CNN
- Los modelos se exportan en `saved_models/`.
- Las imágenes y archivos pesados no deben subirse a Git.

---
