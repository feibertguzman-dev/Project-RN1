# Glosario de Redes Neuronales

## Conceptos Fundamentales

| Término | Definición | Aplicación en el Proyecto |
|---------|------------|---------------------------|
| **Red Neuronal Artificial (RNA)** | Sistema computacional inspirado en el cerebro biológico, compuesto por neuronas artificiales interconectadas. | Base teórica para la clasificación de imágenes Feibert vs Koku. |
| **Aprendizaje por Transferencia (Transfer Learning)** | Técnica que reutiliza un modelo pre-entrenado en una gran cantidad de datos para resolver una nueva tarea con menor cantidad de datos. | El modelo exportado de Teachable Machine usa Transfer Learning (base pre-entrenada + cabeza clasificador). |
| **Aprendizaje Supervisado** | Tipo de aprendizaje donde el modelo aprende a partir de pares (entrada, etiqueta) conocidos. | Las clases Feibert y Koku fueron etiquetadas durante el entrenamiento en Teachable Machine. |
| **Clasificación Binaria** | Problema donde el modelo debe asignar una entrada a una de dos clases posibles. | Clasificar si la imagen es de Feibert o de Koku. |
| **Conjunto de Entrenamiento/Validación/Prueba** | Divisiones del dataset para entrenar, ajustar hiperparámetros y evaluar el modelo. | Gestionado internamente por Teachable Machine durante el proceso de entrenamiento. |

## Arquitectura y Componentes

| Término | Definición | Aplicación en el Proyecto |
|---------|------------|---------------------------|
| **Capa (Layer)** | Unidad computacional que transforma los datos. Puede ser convolucional, densa, de activación, etc. | El modelo .h5 contiene múltiples capas que extraen características y clasifican. |
| **Convolución** | Operación que aplica filtros para detectar características locales (bordes, texturas, formas). | Extrae patrones visuales distintivos entre Feibert y Koku. |
| **Peso (Weight)** | Parámetro ajustable durante el entrenamiento que determina la fuerza de conexión entre neuronas. | Almacenados en el archivo .h5 exportado de Teachable Machine. |
| **Sesgo (Bias)** | Término adicional que permite desplazar la función de activación. | Parte de los parámetros aprendidos del modelo. |
| **Función de Activación** | Función no lineal que permite a la red aprender relaciones complejas. | Usadas en las capas del modelo para introducir no-linealidad. |
| **Softmax** | Función que convierte las salidas en probabilidades que suman 1. | Usada en la capa de salida para obtener la confianza por clase (Feibert/Koku). |

## Preprocesamiento e Inferencia

| Término | Definición | Aplicación en el Proyecto |
|---------|------------|---------------------------|
| **Preprocesamiento** | Transformaciones aplicadas a los datos antes de alimentar el modelo. | Redimensionar imagen, convertir a array, normalizar valores y añadir dimensión batch. |
| **Normalización** | Escalado de los valores de píxeles (comúnmente a [0,1] dividiendo por 255). | Asegura que la imagen de entrada tenga el mismo rango que durante el entrenamiento. |
| **Tamaño de Entrada (Input Shape)** | Dimensiones esperadas por el modelo (altura, ancho, canales). | Detectado del modelo; típico en Teachable Machine 224x224x3. |
| **Batch** | Conjunto de muestras procesadas juntas. Para inferencia individual se usa batch de tamaño 1. | En inferencia añadimos dimensión: (1, height, width, 3). |
| **Inferencia** | Proceso de usar el modelo ya entrenado para hacer predicciones sobre nuevos datos. | Lo que realiza app.py al subir/capturar una imagen. |
| **Predicción** | Clase asignada por el modelo junto con su nivel de confianza. | Se muestra la clase (Feibert o Koku) con porcentaje de confianza. |

## Métricas y Evaluación

| Término | Definición | Aplicación en el Proyecto |
|---------|------------|---------------------------|
| **Exactitud (Accuracy)** | Proporción de predicciones correctas sobre el total. | Métrica principal observada durante el entrenamiento en Teachable Machine. |
| **Función de Pérdida (Loss)** | Medida de qué tan lejos están las predicciones de las etiquetas correctas. | Se minimizó durante el entrenamiento para mejorar el modelo. |
| **Confianza** | Probabilidad asignada a la clase predicha (salida softmax). | Mostrada al usuario en Streamlit para evaluar la fiabilidad. |
| **Sobreajuste (Overfitting)** | Cuando el modelo memoriza el entrenamiento y generaliza mal. | Se busca evitarlo con buen dataset balanceado y arquitectura adecuada. |

## Herramientas Utilizadas

| Término | Definición | Aplicación en el Proyecto |
|---------|------------|---------------------------|
| **Teachable Machine** | Plataforma web de Google para crear modelos de ML sin código, con exportación a TensorFlow/Keras. | Se usó para recolectar imágenes, entrenar y exportar el modelo Feibert vs Koku. |
| **TensorFlow/Keras** | Framework de Deep Learning para crear, entrenar y desplegar redes neuronales. | Usado para cargar el .h5 y realizar inferencia. |
| **Streamlit** | Framework para crear aplicaciones web interactivas de Data Science/ML. | Usado para crear app.py con interfaz de carga/captura de imagen. |
| **CRISP-ML(Q)** | Metodología para desarrollar proyectos ML de forma sistemática y orientada a negocio/calidad. | Guía estructurada del proyecto desde problema hasta despliegue. |
