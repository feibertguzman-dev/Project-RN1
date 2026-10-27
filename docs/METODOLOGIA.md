# Metodología CRISP-ML Aplicada al Proyecto

## ¿Qué es CRISP-ML?
CRISP-ML (Cross-Industry Standard Process for Machine Learning) es una metodología sistemática que guía el desarrollo de proyectos de Machine Learning desde la comprensión del negocio hasta el despliegue, con énfasis en calidad, reproducibilidad y documentación.

## Aplicación al Proyecto: Clasificación Feibert vs Koku

### 1. Business Understanding (Comprensión del Negocio)
- **Problema**: Distinguir entre dos personajes (Feibert y Koku) mediante clasificación de imágenes.
- **Objetivo General**: Desarrollar un clasificador basado en Redes Neuronales con Transfer Learning para diferenciar entre Feibert y Koku.
- **Objetivos Específicos**:
  - Utilizar el modelo exportado desde Teachable Machine
  - Aplicar metodología CRISP-ML
  - Implementar pipeline de inferencia eficiente
  - Desplegar aplicación interactiva con Streamlit
  - Documentar todo el proceso

### 2. Data Understanding (Comprensión de los Datos)
- **Origen**: Imágenes recolectadas directamente en Teachable Machine para ambas clases
- **Clases**: Feibert (clase 0), Koku (clase 1) según labels.txt
- **Tipo de dato**: Imágenes RGB
- **Proceso**: Captura, etiquetado y organización por clase en Teachable Machine

### 3. Data Preparation (Preparación de los Datos)
- **Recolección**: Realizada en Teachable Machine
- **Preprocesamiento para inferencia**:
  1. Cargar imagen (PIL)
  2. Redimensionar al tamaño de entrada del modelo
  3. Convertir a array NumPy
  4. Normalizar (típicamente / 255.0)
  5. Añadir dimensión batch (1, H, W, C)

### 4. Modeling (Modelado)
- **Técnica**: Transfer Learning con modelo exportado de Teachable Machine
- **Arquitectura**: Modelo base pre-entrenado + capa clasificadora adaptada a 2 clases
- **Entrenamiento**: Realizado en Teachable Machine (abstracto al usuario)
- **Exportación**: Formato Keras HDF5 (.h5)
- **Carga**: Mediante 	ensorflow.keras.models.load_model()

### 5. Evaluation (Evaluación)
- **Métricas**: Accuracy/Loss monitoreadas en Teachable Machine
- **Validación práctica**: Pruebas con imágenes nuevas de ambos personajes
- **Análisis de confianza**: Evaluación de probabilidades softmax
- **Criterio**: Modelo funcional para distinguir ambas clases

### 6. Deployment (Despliegue)
- **Aplicación**: Streamlit (app.py)
- **Pipeline**: Carga modelo → Preprocesar → Inferir → Mostrar resultado
- **Optimización**: @st.cache_resource para carga única del modelo
- **Documentación**: README, notebook, landing page
- **Distribución**: Repositorio completo documentado
