# convert_tflite.py
import tensorflow as tf

MODEL_KERAS = "recyclables_mobilenetv2.keras"
TFLITE_FP = "recyclables_model.tflite"
TFLITE_INT8 = "recyclables_model_int8.tflite"

# Load Keras native model
model = tf.keras.models.load_model(MODEL_KERAS)

# 1) float32 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(TFLITE_FP, "wb") as f:
    f.write(tflite_model)
print("Saved float32 TFLite:", TFLITE_FP)

# 2) post-training dynamic-range quantization (smaller)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant = converter.convert()
with open(TFLITE_INT8, "wb") as f:
    f.write(tflite_quant)
print("Saved quantized TFLite:", TFLITE_INT8)
