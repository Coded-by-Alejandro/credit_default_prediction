# Credit Default Prediction 

Predicción de incumplimiento de pagos de tarjetas de crédito mediante técnicas de Machine Learning.

## Descripción

Este proyecto implementa un flujo completo de análisis de datos y modelado predictivo para anticipar el incumplimiento crediticio (default) de clientes de tarjetas de crédito. Se utilizan modelos de clasificación supervisada para identificar patrones de riesgo crediticio.

## Objetivo

Predecir la probabilidad de que un cliente incumpla con el pago de su tarjeta de crédito el próximo mes (`default_payment_next_month`), utilizando información histórica de pagos, datos demográficos y comportamiento financiero.

## Dataset

- **Fuente**: Default of Credit Card Clients Dataset
- **Tamaño**: 30,000 registros
- **Features**: 23 variables predictoras
- **Target**: `default_payment_next_month` (binaria: 0 = No default, 1 = Default)
- **Desbalance de clases**: 77.88% no default, 22.12% default

### Variables principales

| Categoría | Variables |
|-----------|-----------|
| **Historial de pagos** | `pay_0` a `pay_6` (estado de pago en los últimos 6 meses) |
| **Montos de facturas** | `bill_amt1` a `bill_amt6` |
| **Montos pagados** | `pay_amt1` a `pay_amt6` |
| **Información del cliente** | `limit_bal`, `age`, `sex`, `education`, `marriage` |

## Estructura del proyecto

```
credit_default_prediction/
│
├── data/
│   ├── raw/                                    # Datos originales
│   │   └── default_of_credit_card_clients.xls
│   └── processed/                              # Datos limpios
│       └── default_of_credit_card_clients_clean.csv
│
├── notebooks/
│   └── Credit_Default_Prediction.ipynb         # Notebook principal
│
├── reports/
│   ├── figures/                                # Visualizaciones
│   ├── EDA_report.md                          # Reporte de análisis exploratorio
│   ├── numeric_summary.csv                     # Resumen estadístico
│   ├── target_counts.csv                       # Conteo de clases
│   └── target_proportions.csv                  # Proporciones de clases
│
├── requirements.txt                            # Dependencias
└── README.md                                   # Este archivo
```

## Instalación

### Requisitos previos

- Python 3.8+
- pip

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### Librerías principales

```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
openpyxl>=3.0.0
plotly>=5.0.0
```

## Uso

### 1. Carga y limpieza de datos

```python
import pandas as pd
from pathlib import Path

# Cargar dataset original
raw_path = Path("data/raw/default_of_credit_card_clients.xls")
df = pd.read_excel(raw_path, header=1)

# Normalizar nombres de columnas
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Eliminar duplicados y valores nulos
df = df.drop_duplicates()
df = df.dropna()
```

### 2. Ejecutar análisis completo

```bash
jupyter notebook notebooks/Credit_Default_Prediction.ipynb
```

## Pipeline de modelado

### **Análisis Exploratorio de Datos (EDA)**

- Análisis de distribución de la variable objetivo
- Estadísticos descriptivos de variables numéricas
- Análisis de variables categóricas
- Matrices de correlación (Pearson, Spearman, Kendall)
- Mutual Information para selección de features

### **División del dataset**

- **Train**: 60% (18,000 registros)
- **Validation**: 20% (6,000 registros)
- **Test**: 20% (6,000 registros)
- Estratificación para mantener proporciones de clases

### **Modelos implementados**

#### Regresión Logística (Baseline)

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)
log_reg.fit(X_train_scaled, y_train)
```

#### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight='balanced'
)
rf_clf.fit(X_train, y_train)
```

## Resultados

### Comparación de modelos (Test Set)

| Modelo | AUC-ROC | F1-Score | Precision | Recall |
|--------|---------|----------|-----------|--------|
| **Regresión Logística** | 0.7273 | 0.4801 | 0.3856 | 0.6360 |
| **Random Forest** | 0.7695 | 0.4532 | 0.6681 | 0.3429 |

### Insights clave

 **Random Forest** obtiene mejor AUC-ROC y precision  
 **Regresión Logística** tiene mejor recall (detecta más incumplimientos)  
 Variables más importantes: `pay_0`, `limit_bal`, `age`, `bill_amt1`  
 El historial de pagos recientes es el predictor más fuerte  

### Matriz de confusión - Random Forest (Test)

```
                 Predicción
              No Default  Default
Real    
No Default      4447       226
Default          872       455
```

##  Variables más relevantes

Según Random Forest (feature importance):

1. **pay_0** (0.100) - Estado de pago más reciente
2. **limit_bal** (0.063) - Límite de crédito
3. **age** (0.062) - Edad del cliente
4. **bill_amt1** (0.061) - Monto de factura más reciente
5. **bill_amt2** (0.054) - Monto de factura mes anterior

##  Conclusiones

1. **Desbalance de clases**: Solo el 22% de los clientes incumple, requiere técnicas especiales (class_weight='balanced')

2. **Modelos no lineales mejoran la predicción**: Random Forest captura interacciones y patrones complejos mejor que regresión logística

3. **Trade-off Precision vs Recall**: 
   - Random Forest: Mayor precisión pero detecta menos incumplimientos
   - Regresión Logística: Detecta más incumplimientos pero con más falsos positivos

4. **Variables clave**: El historial de pagos (últimos 6 meses) es el predictor más importante

5. **Próximos pasos sugeridos**:
   - Probar Gradient Boosting (XGBoost, LightGBM)
   - Ajuste de hiperparámetros con GridSearchCV
   - Optimización del umbral de decisión según costo de negocio
   - Ingeniería de features (interacciones, ratios)



## Autor

**Alejandro Isaias Montalvo Lupercio**

## Contacto

Para preguntas o sugerencias, por favor abre un issue en el repositorio.

---

 Si este proyecto te fue útil, considera darle una estrella en GitHub!
