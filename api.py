from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# --- CONFIGURACIÓN ---
MODELO_PATH = 'modelo_aprecia_pro.keras' # Asegúrate de usar tu mejor modelo
CLASES = ['cataract', 'glaucoma', 'normal'] # Tus 3 clases finales

print("Cargando modelo en memoria... espera un momento...")
model = tf.keras.models.load_model(MODELO_PATH)
print("¡Modelo cargado y listo para recibir pacientes! 🚀")

def preparar_imagen(image_bytes):
    """Transforma los bytes que llegan de internet en algo que la IA entienda"""
    # 1. Abrir la imagen desde los bytes
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # 2. Redimensionar a 224x224 (Lo que pide MobileNetV2)
    img = img.resize((224, 224))
    # 3. Convertir a array de números
    img_array = tf.keras.utils.img_to_array(img)
    # 4. Expandir dimensiones (de (224,224,3) a (1,224,224,3))
    img_array = tf.expand_dims(img_array, 0)
    return img_array

@app.route('/api/prediccion', methods=['POST'])
def predecir():
    if model is None:
        return jsonify({'error': 'El modelo de IA no está disponible'}), 500

    # 1. DEBUG: Ver qué está llegando
    print("--- NUEVA PETICIÓN RECIBIDA ---")
    print(f"Archivos recibidos: {request.files.keys()}")
    
    # 2. Validación del campo 'imagen'
    if 'imagen' not in request.files:
        print("❌ ERROR 400: No encontré la clave 'imagen' en el POST.")
        return jsonify({'error': 'No se envió el campo "imagen"'}), 400
    
    file = request.files['imagen']
    print(f"Nombre del archivo: '{file.filename}'")

    # 3. Validación del nombre de archivo
    if file.filename == '':
        print("❌ ERROR 400: El nombre del archivo está vacío.")
        return jsonify({'error': 'Archivo vacío'}), 400

    try:
        # 4. Preprocesamiento
        img_array = preparar_imagen(file.read())
        
        # 5. Predicción
        predictions = model.predict(img_array)
        score = predictions[0]
        
        indice_ganador = np.argmax(score)
        clase_ganadora = CLASES[indice_ganador]
        confianza = float(np.max(score) * 100) 
        
        print(f"✅ ÉXITO: {clase_ganadora} ({confianza:.2f}%)")

        return jsonify({
            'resultado': clase_ganadora,
            'probabilidad': confianza,
            'mensaje': 'Análisis completado exitosamente'
        })

    except Exception as e:
        print(f"❌ ERROR INTERNO (500): {e}")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    # Ejecutar servidor en el puerto 5000
    app.run(debug=True, host='0.0.0.0', port=5000)