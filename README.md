# 📊 Analizador de Minería de Datos – Unidad 2

Aplicación desarrollada en Python para el curso de **Minería de Datos – Unidad 2**, enfocada en:

- Partición de datos y control de Data Leakage
- Modelo Baseline
- Clustering (K-Means y Jerárquico)
- Clasificación (Árbol de Decisión y Random Forest)
- Evaluación comparativa de modelos
- Curva ROC
- Matriz de Confusión
- Interpretación gerencial de resultados

La aplicación cuenta con una interfaz gráfica moderna desarrollada con **Tkinter + ttk**, gráficos con **Matplotlib/Seaborn** y procesamiento con **Scikit-Learn**.

---

# 🖼️ Vista General del Proyecto

## Funcionalidades principales

✅ Carga automática de datasets CSV  
✅ Detección automática de encoding  
✅ Pipeline completo de minería de datos  
✅ Prevención de Data Leakage  
✅ Segmentación de clientes  
✅ Clasificación predictiva  
✅ Métricas Accuracy, F1 y AUC  
✅ Curvas ROC  
✅ Matriz de confusión  
✅ Interpretación automática de resultados  
✅ Dashboard visual moderno  

---

# 🛠️ Tecnologías Utilizadas

| Tecnología | Uso |
|---|---|
| Python 3.12 | Lenguaje principal |
| Tkinter | Interfaz gráfica |
| Pandas | Manipulación de datos |
| NumPy | Procesamiento numérico |
| Scikit-Learn | Machine Learning |
| Matplotlib | Visualización |
| Seaborn | Gráficos estadísticos |
| SciPy | Clustering jerárquico |
| Chardet | Detección de encoding |

---

# ⚙️ MANUAL DE DESPLIEGUE

## 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/Patricia-Apaza/MD-U2-Modelado-.git
```

---

## 2️⃣ Ingresar a la carpeta del proyecto

```bash
cd MD-U2-Modelado-
```

---

## 4️⃣ Instalar dependencias

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy chardet
```

O usando requirements:

```bash
pip install -r requirements.txt
```

---

## 5️⃣ Ejecutar el proyecto

```bash
python MD EU2.py
```

---

# 🧪 Requisitos del Sistema

| Requisito | Versión mínima |
|---|---|
| Python | 3.10+ |
| RAM | 4 GB |
| Sistema Operativo | Windows / Linux / Mac |

---

# 👤 MANUAL DE USUARIO

# 🚀 Inicio de la Aplicación

Al ejecutar el programa aparecerá la interfaz principal del sistema.

La interfaz está dividida en:

- Barra superior
- Panel de control
- Registro de ejecución
- Métricas rápidas
- Tabla comparativa
- Panel de gráficos y análisis

---

# 📂 Paso 1: Cargar Dataset

Presionar el botón:

```text
📂 Cargar CSV
```

Luego seleccionar un archivo `.csv`.

El sistema:

✅ Detectará automáticamente el encoding  
✅ Cargará el dataset  
✅ Mostrará filas y columnas  
✅ Habilitará el análisis  

---

# 🎯 Paso 2: Seleccionar Variable Objetivo

En el combo:

```text
Variable objetivo
```

Seleccionar la columna que se desea predecir.

Ejemplos:

- Churn
- Respuesta
- Compra
- EstadoCliente

---

# ▶️ Paso 3: Ejecutar Análisis

Presionar:

```text
▶ Ejecutar Análisis
```

El sistema ejecutará automáticamente:

## ✔ Preparación de Datos

- Limpieza
- Imputación
- Escalado
- Codificación

---

## ✔ Partición de Datos

División:

| Conjunto | Porcentaje |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

---

## ✔ Prevención de Data Leakage

El sistema evita Data Leakage aplicando:

- `fit_transform()` solo en entrenamiento
- `transform()` en validación y prueba

Esto garantiza resultados reales y confiables.

---

## ✔ Baseline

Se utiliza:

```python
DummyClassifier(strategy="most_frequent")
```

como referencia mínima de comparación.

---

# 🔵 Segmentación de Clientes

Se aplican:

## ✔ K-Means

Agrupa clientes similares usando centroides.

---

## ✔ Clustering Jerárquico

Agrupa clientes mediante relaciones jerárquicas.

---

## ✔ Índice Silhouette

Evalúa calidad de clusters:

| Valor | Interpretación |
|---|---|
| Cerca de 1 | Excelente |
| Cerca de 0 | Solapado |
| Negativo | Malo |

---

# 🌳 Clasificación Predictiva

El sistema entrena:

## ✔ Árbol de Decisión

Modelo interpretable basado en reglas.

---

## ✔ Random Forest

Conjunto de árboles para mejorar precisión y estabilidad.

---

# 📈 Métricas Evaluadas

| Métrica | Descripción |
|---|---|
| Accuracy | Exactitud general |
| F1-Score | Balance entre precisión y recall |
| AUC | Calidad de clasificación |

---

# 📊 Evaluación Comparativa

El sistema genera:

✅ Curva ROC  
✅ Tabla comparativa  
✅ Barras comparativas  
✅ Métricas automáticas  

---

# 🔲 Matriz de Confusión

Se visualizan:

| Métrica | Significado |
|---|---|
| VP | Verdaderos Positivos |
| VN | Verdaderos Negativos |
| FP | Falsos Positivos |
| FN | Falsos Negativos |

También se calculan:

- Precision
- Recall
- Accuracy
- F1
- AUC
- Especificidad

---

# 💬 Interpretación Gerencial

La pestaña:

```text
💬 Interpretación
```

explica automáticamente:

✅ Resultados técnicos  
✅ Impacto empresarial  
✅ Interpretación de métricas  
✅ Perfiles de clientes  
✅ Recomendaciones gerenciales  

---

# 📸 Resultados Esperados

El sistema mostrará:

- Dashboard visual
- Gráficos interactivos
- Curvas ROC
- Clusters
- Matrices
- Métricas
- Interpretaciones automáticas

---

# ⚠️ Posibles Errores

## Error: Librería no instalada

```bash
ModuleNotFoundError
```

Solución:

```bash
pip install nombre_libreria
```

---

## Error: Archivo CSV inválido

Verificar:

- Separador correcto
- Dataset no corrupto
- Encodings compatibles

---

# 📚 Conceptos Aplicados

- Data Mining
- CRISP-DM
- Machine Learning
- Clustering
- Clasificación
- Evaluación de Modelos
- Data Leakage
- Segmentación de Clientes