import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Configuración de la página
st.set_page_config(
    page_title='Clasificador Feibert vs Koku',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded'
)

# CSS personalizado
st.markdown('''
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
        margin: 10px 0;
    }
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_model_and_labels():
    '''Carga el modelo y las etiquetas con cache'''
    model_path = 'models/keras_model.h5'
    labels_path = 'models/labels.txt'
    
    # Intentar rutas alternativas
    if not os.path.exists(model_path):
        model_path = 'converted_keras (Feibert-Koku)/que .h5'
    if not os.path.exists(labels_path):
        labels_path = 'converted_keras (Feibert-Koku)/labels.txt'
    
    try:
        model = tf.keras.models.load_model(model_path)
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = [line.strip().split(' ', 1)[1] if ' ' in line.strip() else line.strip() 
                      for line in f.readlines()]
        return model, labels
    except Exception as e:
        st.error(f'Error al cargar modelo: {e}')
        return None, None

def preprocess_image(image, target_size):
    '''Preprocesa la imagen para el modelo'''
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.markdown('<h1 class=\"main-header\">🧠 Clasificador Feibert vs Koku</h1>', unsafe_allow_html=True)
    st.markdown('<p class=\"sub-header\">Redes Neuronales - UCN | Transfer Learning con Teachable Machine</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.image('https://www.tensorflow.org/images/tf_logo_social.png', width=150)
        st.title('Acerca del Proyecto')
        st.info('''
        **Metodología:** CRISP-ML
        
        **Técnica:** Transfer Learning
        
        **Origen:** Google Teachable Machine
        
        **Clases:** Feibert, Koku
        ''')
        st.divider()
        st.markdown('**Desarrollado para:** Redes Neuronales - UCN')
    
    # Cargar modelo
    model, labels = load_model_and_labels()
    
    if model is None or labels is None:
        st.error('No se pudo cargar el modelo. Verifica los archivos en models/ o converted_keras/')
        return
    
    # Obtener tamaño de entrada
    input_shape = model.input_shape
    target_size = (input_shape[1], input_shape[2]) if len(input_shape) == 4 else (224, 224)
    
    st.success(f'✅ Modelo cargado correctamente | Entrada: {target_size} | Clases: {labels}')
    
    # Pestañas
    tab1, tab2, tab3 = st.tabs(['🔍 Clasificar', '📊 Información del Modelo', '📚 Documentación'])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader('Subir Imagen')
            uploaded_file = st.file_uploader(
                'Selecciona una imagen (JPG, JPEG, PNG)',
                type=['jpg', 'jpeg', 'png']
            )
            
            st.subheader('O usar Cámara')
            camera_image = st.camera_input('Capturar foto')
            
            image = None
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
            elif camera_image is not None:
                image = Image.open(camera_image)
        
        with col2:
            st.subheader('Resultado de Clasificación')
            if image is not None:
                st.image(image, caption='Imagen a clasificar', use_container_width=True)
                
                with st.spinner('Analizando imagen...'):
                    img_array = preprocess_image(image, target_size)
                    predictions = model.predict(img_array, verbose=0)
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class_idx] * 100
                    predicted_class = labels[predicted_class_idx] if predicted_class_idx < len(labels) else f'Clase {predicted_class_idx}'
                
                st.markdown(f'''
                <div class=\"prediction-box\">
                    <h2>🎯 Predicción: {predicted_class}</h2>
                    <h3>Confianza: {confidence:.2f}%</h3>
                </div>
                ''', unsafe_allow_html=True)
                
                st.subheader('Probabilidades por Clase')
                for i, label in enumerate(labels):
                    prob = predictions[0][i] * 100
                    st.progress(int(prob), text=f'{label}: {prob:.2f}%')
            else:
                st.info('👆 Sube una imagen o usa la cámara para comenzar la clasificación')
    
    with tab2:
        st.subheader('Información del Modelo')
        col1, col2 = st.columns(2)
        with col1:
            st.metric('Forma de Entrada', str(input_shape))
            st.metric('Forma de Salida', str(model.output_shape))
            st.metric('Número de Capas', len(model.layers))
        with col2:
            st.metric('Clases', len(labels))
            st.metric('Etiquetas', ', '.join(labels))
        
        with st.expander('Ver Resumen del Modelo'):
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            st.text('\n'.join(stringlist))
    
    with tab3:
        st.subheader('Documentación del Proyecto')
        st.markdown('''
        - [README.md](README.md) - Documentación principal
        - [Glosario de Redes Neuronales](docs/GLOSARIO.md)
        - [Metodología CRISP-ML](docs/METODOLOGIA.md)
        - [Arquitectura del Pipeline](docs/ARQUITECTURA.md)
        - [Notebook Jupyter](notebooks/01_Clasificacion_Feibert_Koku_CRISP_ML.ipynb)
        - [Landing Page](index.html)
        ''')
        st.info('''💡 **Tip:** Este proyecto fue desarrollado aplicando metodología CRISP-ML 
        con énfasis en Transfer Learning usando un modelo exportado de Teachable Machine.''')

if __name__ == '__main__':
    main()
