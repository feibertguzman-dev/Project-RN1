# Clasificador Feibert vs Koku - Redes Neuronales (UCN)

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000.svg)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243.svg)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-10.x-green.svg)](https://python-pillow.org/)
[![License: Academic](https://img.shields.io/badge/License-Académico-lightgrey.svg)](LICENSE)

## Descripción

Proyecto de clasificación binaria de imágenes desarrollado para la **Materia de Redes Neuronales** de la **Fundación Universitaria Católica del Norte (UCN)**. El objetivo es diferenciar entre dos personajes: **Feibert** y **Koku**, utilizando un modelo de Red Neuronal con **Transfer Learning** exportado desde **Google Teachable Machine**.

## Objetivo

Desarrollar, documentar y desplegar un clasificador de imágenes Feibert vs Koku aplicando la metodología **CRISP-ML**, con énfasis en el proceso de Redes Neuronales, pipeline de inferencia eficiente y aplicación interactiva en **Streamlit**.

## Metodología

Este proyecto sigue el estándar **CRISP-ML (Cross-Industry Standard Process for Machine Learning)**:

1. **Business Understanding** - Comprensión del problema y objetivos
2. **Data Understanding** - Análisis de las clases y datos
3. **Data Preparation** - Preprocesamiento para inferencia
4. **Modeling** - Transfer Learning con modelo Teachable Machine
5. **Evaluation** - Validación con pruebas prácticas
6. **Deployment** - Despliegue con Streamlit

Ver [docs/METODOLOGIA.md](docs/METODOLOGIA.md) para mayor detalle.

## Estructura del Proyecto

`
Project-RN1/
├── app.py                        # Aplicación Streamlit
├── requirements.txt              # Dependencias
├── README.md                     # Documentación principal
├── index.html                    # Landing page con documentación interactiva
├── assets/                       # Recursos gráficos
├── docs/                         # Documentación técnica
│   ├── GLOSARIO.md               # Glosario de términos de Redes Neuronales
│   ├── METODOLOGIA.md            # Metodología CRISP-ML
│   └── ARQUITECTURA.md           # Arquitectura del pipeline
├── models/                       # Modelo organizado
│   ├── keras_model.h5            # Modelo Keras
│   └── labels.txt                # Etiquetas de clases
├── notebooks/
│   └── 01_Clasificacion_Feibert_Koku_CRISP_ML.ipynb  # Cuaderno documentado
└── converted_keras (Feibert-Koku)/ # Soportes originales de Teachable Machine
    ├── labels.txt
    └── que .h5
`

## Tecnologías Utilizadas

| Tecnología | Uso |
|-------------|-----|
| **Python** | Lenguaje principal |
| **TensorFlow/Keras** | Carga y ejecución del modelo |
| **Teachable Machine** | Entrenamiento con Transfer Learning |
| **Streamlit** | Aplicación web interactiva |
| **NumPy** | Procesamiento numérico |
| **Pillow** | Manipulación de imágenes |
| **Jupyter Notebook** | Documentación interactiva |

## Instalación

### 1. Clonar o descargar el proyecto
`ash
# Si es un repositorio git
git clone <url-repositorio>
cd Project-RN1
`

### 2. Crear entorno virtual (recomendado)
`ash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
`

### 3. Instalar dependencias
`ash
pip install -r requirements.txt
`

## Ejecución

### Aplicación Streamlit
`ash
streamlit run app.py
`
La aplicación se abrirá en http://localhost:8501

### Cuaderno Jupyter
`ash
jupyter notebook notebooks/01_Clasificacion_Feibert_Koku_CRISP_ML.ipynb
`

### Landing Page
Abrir index.html directamente en el navegador web.

## Uso de la Aplicación

1. Abrir la app en Streamlit
2. Seleccionar una imagen (JPG, PNG, JPEG) o usar la cámara
3. El modelo preprocesará la imagen automáticamente
4. Se mostrará la predicción con clase y porcentaje de confianza

## Transfer Learning en este Proyecto

El modelo fue creado con **Google Teachable Machine** usando **Transfer Learning**. Esto significa que se parte de una red neuronal pre-entrenada (con conocimiento general sobre imágenes) y se entrena únicamente la capa clasificadora para distinguir entre Feibert y Koku, requiriendo menos datos y obteniendo buenos resultados.

## Documentación

- [Glosario de Redes Neuronales](docs/GLOSARIO.md) - Términos clave con aplicación al proyecto
- [Metodología CRISP-ML](docs/METODOLOGIA.md) - Proceso paso a paso
- [Arquitectura](docs/ARQUITECTURA.md) - Pipeline técnico detallado
- [Notebook Completo](notebooks/01_Clasificacion_Feibert_Koku_CRISP_ML.ipynb) - Desarrollo documentado

## Autor

**Feibert**  
Materia: Redes Neuronales  
Fundación Universitaria Católica del Norte (UCN)

## Referencias

- [Google Teachable Machine](https://teachablemachine.withgoogle.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [CRISP-ML(Q) - ML Workflow](https://ml-ops.org/content/crisp-ml)
