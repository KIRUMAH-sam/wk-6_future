# train_model.py
import tensorflow as tf
from tensorflow import keras
import os

layers = tf.keras.layers
models = tf.keras.models

DATA_DIR = "recyclables_dataset"
IMG_SIZE = (128,128)
BATCH_SIZE = 16
EPOCHS = 10

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=123,
    validation_split=0.2,
    subset="training"
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=123,
    validation_split=0.2,
    subset="validation"
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE+(3,), include_top=False, weights="imagenet")
base.trainable = False  # freeze base for faster training

inputs = tf.keras.Input(shape=IMG_SIZE+(3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)

num_classes = 3  
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save("recyclables_mobilenetv2.keras", save_format="keras")
# Print final val accuracy
val_loss, val_acc = model.evaluate(val_ds)
print("Validation accuracy:", val_acc)
