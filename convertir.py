import tensorflow as tf

# 1. Cargar el modelo actual (formato nuevo)
print("Cargando modelo...")
model = tf.keras.models.load_model('modelo_aprecia_pro.keras')

# 2. Guardarlo en formato clásico (H5)
# save_format='h5' es la clave para la compatibilidad
print("Guardando en formato compatible .h5...")
model.save('modelo_aprecia_pro.h5', save_format='h5')

print("¡Listo! Sube 'modelo_aprecia_pro.h5' a GitHub.")