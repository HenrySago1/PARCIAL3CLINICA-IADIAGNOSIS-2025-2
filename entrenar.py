import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
DATASET_PATH = 'dataset'  # El nombre exacto de tu carpeta de datos
IMG_HEIGHT = 224          # Tamaño estándar para IAs de imágenes
IMG_WIDTH = 224
BATCH_SIZE = 32           # Cuantas imágenes procesa de golpe
EPOCHS = 10               # Cuántas veces repasará todo el temario (puedes subirlo a 20 si tienes tiempo)

print("1. Cargando las imágenes y separando datos...")

# Cargar datos de ENTRENAMIENTO (80% de las fotos)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,     # Reservamos 20% para examen
    subset="training",
    seed=123,                 # Semilla para que el azar sea reproducible
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Cargar datos de VALIDACIÓN/PRUEBA (20% de las fotos)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Ver qué clases encontró (para asegurar que leyó las carpetas)
class_names = train_ds.class_names
print(f"Clases encontradas: {class_names}")

# Optimizar la carga de datos para que no sea lenta
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("\n2. Construyendo el Cerebro de la IA (Modelo)...")

# Usamos MobileNetV2 (Pre-entrenada, ligera y potente)
# include_top=False significa: "Quítale la parte final que clasifica perros y gatos"
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False # Congelamos lo que ya sabe para no dañarlo

# Añadimos nuestra propia "capa de oftalmología" al final
model = models.Sequential([
    layers.Rescaling(1./127.5, offset=-1, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)), # Preprocesamiento
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2), # Apaga neuronas al azar para evitar memorización
    layers.Dense(len(class_names), activation='softmax') # La capa final que decide (Catarata, Glaucoma, etc.)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\n3. ¡Comenzando el Entrenamiento! (Esto puede tardar unos minutos)...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

print("\n4. Guardando el modelo...")
# Guardamos el modelo en formato nuevo de Keras
model.save('modelo_aprecia_v1.keras') 
print("¡Éxito! El archivo 'modelo_aprecia_v1.keras' ha sido creado.")