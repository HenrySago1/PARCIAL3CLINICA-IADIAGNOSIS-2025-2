import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os

# --- CONFIGURACIÓN ---
DATASET_PATH = 'dataset'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS_INICIALES = 10
EPOCHS_FINETUNING = 10  # Segunda ronda de estudio intensivo

print("1. Cargando datos con configuración avanzada...")

# Definir técnica de Aumento de Datos (Data Augmentation)
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
  layers.RandomContrast(0.2),
])

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
AUTOTUNE = tf.data.AUTOTUNE
# Aplicamos el Data Augmentation SÓLO al entrenamiento
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                        num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("\n2. Construyendo Modelo V2 (MobileNetV2)...")

base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False # Empezamos con el modelo congelado

model = models.Sequential([
    layers.Rescaling(1./127.5, offset=-1, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3), # Subimos un poco el dropout para evitar memorización
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n3. Fase 1: Entrenamiento Básico...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_INICIALES)

# --- AQUÍ OCURRE LA MAGIA (FINE TUNING) ---
print("\n4. Fase 2: Descongelando capas para especialización (Fine-Tuning)...")

base_model.trainable = True # Descongelamos el cerebro base

# Solo re-entrenamos las últimas capas (las más complejas)
# MobileNetV2 tiene 154 capas. Congelamos hasta la 100.
for layer in base_model.layers[:100]:
    layer.trainable = False

# IMPORTANTE: Usamos un learning_rate MUY bajo para no romper lo aprendido
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Iniciando re-entrenamiento fino...")
history_fine = model.fit(train_ds, 
                         validation_data=val_ds, 
                         epochs=EPOCHS_FINETUNING,
                         initial_epoch=history.epoch[-1])

print("\n5. Guardando Modelo Final...")
model.save('modelo_aprecia_pro.keras')
print("¡Listo! Usa 'modelo_aprecia_pro.keras' en tu prueba.")