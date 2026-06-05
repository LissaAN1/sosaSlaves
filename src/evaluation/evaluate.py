import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

def evaluate_multioutput_model(y_true_quality, y_pred_quality, y_true_size, y_pred_size, quality_classes=None, size_classes=None):
    """
    Evalúa el rendimiento de un modelo con múltiples salidas (Calidad y Tamaño).
    
    Args:
        y_true_quality: Etiquetas reales de calidad.
        y_pred_quality: Predicciones de calidad del modelo.
        y_true_size: Etiquetas reales de tamaño.
        y_pred_size: Predicciones de tamaño del modelo.
        quality_classes: Lista de nombres de las clases de calidad (ej. ['mala', 'regular', 'buena']).
        size_classes: Lista de nombres de las clases de tamaño (ej. ['pequeño', 'mediano', 'grande']).
    """
    print("="*60)
    print("REPORTE DE CLASIFICACIÓN: CALIDAD")
    print("="*60)
    print(classification_report(y_true_quality, y_pred_quality, target_names=quality_classes))
    
    print("="*60)
    print("REPORTE DE CLASIFICACIÓN: TAMAÑO")
    print("="*60)
    print(classification_report(y_true_size, y_pred_size, target_names=size_classes))
    
    # Matrices de confusión
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Matriz para Calidad
    cm_quality = confusion_matrix(y_true_quality, y_pred_quality)
    sns.heatmap(cm_quality, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=quality_classes, yticklabels=quality_classes)
    axes[0].set_title('Matriz de Confusión - Calidad')
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Predicho')
    
    # Matriz para Tamaño
    cm_size = confusion_matrix(y_true_size, y_pred_size)
    sns.heatmap(cm_size, annot=True, fmt='d', cmap='Greens', ax=axes[1], 
                xticklabels=size_classes, yticklabels=size_classes)
    axes[1].set_title('Matriz de Confusión - Tamaño')
    axes[1].set_ylabel('Real')
    axes[1].set_xlabel('Predicho')
    
    plt.tight_layout()
    plt.show()

def plot_roc_multiclass(y_true, y_pred_proba, classes, title="Curva ROC multiclase"):
    """
    Plotea la curva ROC para un problema multiclase. 
    Requiere que el modelo provea probabilidades de predicción.
    """
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = y_true_bin.shape[1]
    
    if n_classes <= 1:
        print("La curva ROC requiere al menos 2 clases.")
        return
        
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve de la clase {classes[i]} (área = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
