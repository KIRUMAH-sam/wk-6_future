# raspi_infer.py
import argparse
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="recyclables_model.tflite")
parser.add_argument("--image", default=None, help="Path to test image (optional). If not, use webcam if available.")
args = parser.parse_args()

interpreter = tf.lite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()
input_idx = interpreter.get_input_details()[0]["index"]
output_idx = interpreter.get_output_details()[0]["index"]

def preprocess(img):
    img = cv2.resize(img, (128,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img.astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def infer_image(path):
    img = cv2.imread(path)
    inp = preprocess(img)
    interpreter.set_tensor(input_idx, inp)
    start=time.time()
    interpreter.invoke()
    dt = time.time()-start
    out = interpreter.get_tensor(output_idx)
    pred = out[0].argmax()
    print("Prediction:", pred, "in", dt*1000, "ms")

if args.image:
    infer_image(args.image)
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            inp = preprocess(frame)
            interpreter.set_tensor(input_idx, inp)
            start=time.time()
            interpreter.invoke()
            dt = time.time()-start
            out = interpreter.get_tensor(output_idx)
            pred = out[0].argmax()
            cv2.putText(frame, f"Pred: {pred} {int(dt*1000)}ms", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("edge infer", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cap.release()
        cv2.destroyAllWindows()
