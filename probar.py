import tensorflow as tf
import numpy as np
import os

# --- CONFIGURACIÓN ---
MODELO_PATH = 'modelo_aprecia_pro.keras'
IMAGEN_A_PROBAR = 'prueba.jpg' 

# Asegúrate que este orden coincida con el alfabético de tus carpetas
CLASES = ['cataract', 'glaucoma', 'normal']

def cargar_y_probar():
    if not os.path.exists(IMAGEN_A_PROBAR):
        print(f"❌ ERROR: No encuentro '{IMAGEN_A_PROBAR}'")
        return

    print("Cargando modelo...")
    model = tf.keras.models.load_model(MODELO_PATH)

    # Cargar imagen
    img = tf.keras.utils.load_img(IMAGEN_A_PROBAR, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    print("Analizando...")
    predictions = model.predict(img_array)
    
    # --- CORRECCIÓN AQUÍ ---
    # El modelo YA devuelve probabilidades, no aplicamos softmax otra vez.
    score = predictions[0] 
    # -----------------------

    clase_ganadora = CLASES[np.argmax(score)]
    confianza = 100 * np.max(score)

    print("\n" + "="*30)
    print(f" DIAGNÓSTICO CORREGIDO")
    print("="*30)
    print(f"Diagnóstico: {clase_ganadora.upper()}")
    # Color verde si es alta confianza, rojo si es baja
    color = "🟢" if confianza > 70 else "🔴"
    print(f"Certeza:     {confianza:.2f}% {color}")
    print("-" * 30)
    
    print("\nDesglose:")
    for i, clase in enumerate(CLASES):
        prob = 100 * score[i]
        barra = "█" * int(prob / 5)
        print(f"{clase.ljust(20)}: {prob:.2f}%  {barra}")

if __name__ == "__main__":
    cargar_y_probar()