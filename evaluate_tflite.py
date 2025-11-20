# evaluate_tflite.py
import tensorflow as tf
import numpy as np
import os
from PIL import Image

TFLITE_MODEL = "recyclables_model.tflite"
DATA_DIR = "recyclables_dataset"
IMG_SIZE = (128,128)

# load class names
classes = sorted(os.listdir(DATA_DIR))

# load tflite
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr

# simple evaluate over dataset
total = 0
correct = 0
for i,cls in enumerate(classes):
    folder = os.path.join(DATA_DIR, cls)
    for fn in os.listdir(folder):
        if not fn.lower().endswith(".png"): continue
        path = os.path.join(folder, fn)
        img = preprocess_image(path)
        inp = np.expand_dims(img, axis=0)
        interpreter.set_tensor(input_index, inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_index)
        pred = np.argmax(out, axis=1)[0]
        total += 1
        if pred == i: correct += 1

accuracy = correct / total
print(f"TFLite model accuracy on dataset: {accuracy:.3f} ({correct}/{total})")
