# Arquitectura del Proyecto

## Diagrama Conceptual del Pipeline

`
Teachable Machine (Entrenamiento)
    ↓ (Exportar modelo Keras)
converted_keras (Feibert-Koku)/
    ├── labels.txt
    └── que .h5
    ↓ (Copiar y organizar)
models/
    ├── keras_model.h5
    └── labels.txt
    ↓
Pipeline de Inferencia
    ├── Carga de modelo (cached)
    ├── Carga de etiquetas
    ├── Preprocesamiento de imagen
    ├── Predicción (Forward Pass)
    └── Postprocesamiento (Softmax + Mapeo)
    ↓
Aplicación Streamlit (app.py)
`

## Componentes

### 1. Modelo
- **Archivo**: models/keras_model.h5
- **Tipo**: Modelo Keras completo (arquitectura + pesos)
- **Clases**: 2 (Feibert, Koku)
- **Entrada**: Imagen RGB redimensionada
- **Salida**: Probabilidades por clase

### 2. Etiquetas
- **Archivo**: models/labels.txt y converted_keras.../labels.txt
- **Formato**: índice nombre_clase
- **Mapeo**: 0->Feibert, 1->Koku

### 3. Preprocesamiento
`python
# Pasos típicos
img = Image.open(uploaded_file).convert('RGB')
img_resized = img.resize((input_width, input_height))
img_array = np.array(img_resized)
img_array = img_array.astype(np.float32) / 255.0  # Normalización
img_batch = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
`

### 4. Inferencia
`python
predictions = model.predict(img_batch)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class] * 100
class_name = labels[predicted_class]
`

## Estructura de Archivos

Ver README.md para estructura completa detallada.

## Consideraciones Técnicas
- **Carga única**: Uso de @st.cache_resource evita recargar modelo en cada interacción
- **Rutas relativas**: Código portable entre entornos
- **Compatibilidad**: TensorFlow 2.x
- **CPU-friendly**: Funciona sin GPU
- **Formato headless**: opencv-python-headless para despliegue
