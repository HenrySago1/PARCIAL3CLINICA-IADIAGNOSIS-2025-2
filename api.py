from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging

# --- CONFIGURACIÓN ---
# Asegúrate de que este sea el nombre de tu archivo .h5 en GitHub
MODEL_PATH = 'modelo_aprecia_pro.h5' 
IMG_SIZE = 224
CLASES = ['cataract', 'glaucoma', 'normal'] 

# Configurar logging para ver errores en Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) # Permite que Laravel se conecte desde cualquier lado

# --- CARGAR EL MODELO AL INICIO ---
try:
    logger.info("Cargando modelo... esto puede tardar unos segundos.")
    # Cargamos el modelo .h5 que es compatible universalmente
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"¡Modelo cargado exitosamente! Clases: {CLASES}")
except Exception as e:
    logger.error(f"❌ ERROR CRÍTICO cargando modelo: {e}")
    model = None

def preparar_imagen(image_bytes):
    """Transforma los bytes de la imagen para que la IA la entienda"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    return img_array

# --- RUTA 1: LA PREDICCIÓN (Para Laravel) ---
@app.route('/api/prediccion', methods=['POST'])
def predecir():
    if model is None:
        return jsonify({'error': 'El modelo de IA no está disponible'}), 500

    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió el campo "imagen"'}), 400
    
    file = request.files['imagen']
    if file.filename == '':
        return jsonify({'error': 'Archivo vacío'}), 400

    try:
        logger.info(f"Procesando imagen: {file.filename}")
        
        # Procesar y Predecir
        img_array = preparar_imagen(file.read())
        predictions = model.predict(img_array)
        score = predictions[0]
        
        # Calcular resultado
        indice_ganador = np.argmax(score)
        clase_ganadora = CLASES[indice_ganador]
        confianza = float(np.max(score) * 100)
        
        logger.info(f"Resultado: {clase_ganadora.upper()} ({confianza:.2f}%)")

        return jsonify({
            'resultado': clase_ganadora,
            'probabilidad': confianza,
            'mensaje': 'Exito'
        })

    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        return jsonify({'error': str(e)}), 500

# --- RUTA 2: SALUD (Para saber si está vivo) ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online', 
        'model_loaded': model is not None,
        'message': 'La IA esta despierta y lista 🤖'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')