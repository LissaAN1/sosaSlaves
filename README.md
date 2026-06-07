# Clasificador de Calidad y Tamaño de Frutas

**Proyecto Final — Algoritmos y Programación III · Universidad ICESI · Semestre 2026-1**

Equipo **sosaSlaves**: Angy Hurtado · Hideki Tamura · David Vergara

---

## Descripción

Sistema de clasificación multi-salida que predice simultáneamente:

- **Calidad**: Mala · Buena · Regular
- **Tamaño**: Pequeño · Mediano · Grande

El proyecto usa visión por computadora, modelos tradicionales de machine learning, una red neuronal convolucional multi-salida y una interfaz web desarrollada con **Streamlit**.

---

## Estructura esperada del proyecto

```text
fruit-quality-classification/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── informe_final.pdf
│   └── figures/
├── notebooks/
│   ├── 01_analisis_exploratorio_datos.ipynb
│   ├── 02_experimentos_ml_tradicional.ipynb
│   └── 03_experimentos_deep_learning.ipynb
├── saved_models/
│   ├── random_forest_multioutput.pkl
│   ├── svm_lineal_multioutput.pkl
│   ├── xgboost_multioutput.pkl
│   ├── best_traditional_multioutput.pkl
│   ├── best_rf_multioutput.pkl
│   ├── traditional_model_metadata.json
│   ├── traditional_models_results.csv
│   ├── cnn_best_checkpoint.keras
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

## Modelos y artefactos guardados

Los modelos deben ubicarse en la carpeta `saved_models/`.

### Artefactos generados por el notebook 02 — ML tradicional

| Archivo | Descripción | Uso |
|---|---|---|
| `random_forest_multioutput.pkl` | Random Forest base | Modelo comparativo tradicional |
| `svm_lineal_multioutput.pkl` | SVM lineal | Modelo comparativo tradicional |
| `xgboost_multioutput.pkl` | XGBoost | Modelo tradicional principal |
| `best_traditional_multioutput.pkl` | Mejor modelo tradicional de los experimentos | Corresponde al XGBoost ganador por F1-macro promedio multi-salida |
| `best_rf_multioutput.pkl` | Copia de Random Forest | Compatibilidad con versiones previas |
| `traditional_model_metadata.json` | Metadatos del pipeline tradicional | Etiquetas, dimensión de features y configuración |
| `traditional_models_results.csv` | Tabla de resultados finales | Evidencia reproducible de la comparación entre modelos tradicionales |

> Nota: el notebook 02 puede incluir una sección opcional para ajuste de hiperparámetros de Random Forest. Sin embargo, la entrega final no depende de `random_forest_tuned_multioutput.pkl`, ya que la comparación principal se realiza entre Random Forest, SVM lineal, XGBoost y CNN.

### Artefactos generados por el notebook 03 — CNN

| Archivo | Descripción | Uso |
|---|---|---|
| `cnn_best_checkpoint.keras` | Checkpoint del mejor epoch durante entrenamiento | Respaldo del entrenamiento |
| `cnn_multioutput.keras` | Modelo final cargado desde el checkpoint | Modelo CNN usado por `app.py` |


---

## Instalación

Se recomienda usar Python 3.10 o 3.11.

### 1. Crear entorno virtual

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

En Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Actualizar herramientas base

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3. Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

---

## Ejecutar la aplicación

Desde la raíz del proyecto:

```bash
python -m streamlit run app.py
```

Luego abrir en el navegador:

```text
http://localhost:8501
```

---

## Uso de la aplicación

1. Seleccionar el modelo en el panel lateral.
2. Subir una imagen o tomar una foto con la cámara.
3. Presionar **Predecir simultáneamente**.
4. La aplicación muestra:
   - calidad estimada;
   - tamaño estimado;
   - confianza, cuando el modelo permite calcularla.

La aplicación también muestra un panel de verificación de artefactos esperados para comprobar qué archivos están disponibles y cuáles faltan.

---

## Dataset

El dataset final contiene imágenes de seis frutas:

- Apple
- Banana
- Guava
- Lime
- Orange
- Pomegranate

Clases de calidad:

- Bad
- Good
- Regular

Distribución reportada en el proyecto:

| Clase | Imágenes | Porcentaje |
|---|---:|---:|
| Bad | 9,142 | 32.2% |
| Good | 14,599 | 51.4% |
| Regular | 4,654 | 16.4% |

Total de imágenes originales: **28,395**.

---

## Pipeline de inferencia tradicional

Para los modelos tradicionales, `app.py` usa la función:

```python
process_image_from_streamlit(uploaded_file, model_type="ml")
```

El vector esperado tiene dimensión:

```text
513 features = 512 bins HSV + 1 area_ratio
```

El archivo `traditional_model_metadata.json` documenta esta dimensión y el orden de etiquetas.

Ejemplo de metadata esperado:

```json
{
  "best_model_name": "XGBoost",
  "quality_labels": [0, 1, 2],
  "quality_names": ["Mala", "Buena", "Regular"],
  "size_labels": [0, 1, 2],
  "size_names": ["Pequeño", "Mediano", "Grande"],
  "feature_dim": 513,
  "test_size": 0.2,
  "random_state": 42
}
```

---

## Pipeline de inferencia CNN

Para la CNN, `app.py` usa la función:

```python
process_image_from_streamlit(uploaded_file, model_type="dl")
```

La entrada esperada es un tensor:

```text
(1, 224, 224, 3)
```

La normalización debe coincidir con la usada durante el entrenamiento del notebook 03.

---

## Reproducir entrenamiento

### 1. Preprocesamiento

```bash
python src/data_processing/preprocess.py
```

### 2. EDA

```bash
jupyter notebook notebooks/01_analisis_exploratorio_datos.ipynb
```

### 3. Modelos tradicionales

```bash
jupyter notebook notebooks/02_experimentos_ml_tradicional.ipynb
```

Este notebook debe generar:

```text
saved_models/random_forest_multioutput.pkl
saved_models/svm_lineal_multioutput.pkl
saved_models/xgboost_multioutput.pkl
saved_models/best_traditional_multioutput.pkl
saved_models/best_rf_multioutput.pkl
saved_models/traditional_model_metadata.json
saved_models/traditional_models_results.csv
```

### 4. CNN

```bash
jupyter notebook notebooks/03_experimentos_deep_learning.ipynb
```

Este notebook debe generar:

```text
saved_models/cnn_best_checkpoint.keras
saved_models/cnn_multioutput.keras
```

