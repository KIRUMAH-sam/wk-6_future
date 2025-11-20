# Recyclables Classification with MobileNetV2

This project implements an image classification system to identify different types of recyclables using **TensorFlow** and **MobileNetV2**. The trained model is also converted to **TFLite** for deployment on edge devices such as Raspberry Pi.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Training the Model](#training-the-model)
- [Converting to TFLite](#converting-to-tflite)
- [Evaluating TFLite Model](#evaluating-tflite-model)
- [Running on Raspberry Pi / Edge Device](#running-on-raspberry-pi--edge-device)
- [Dependencies](#dependencies)

---

## Project Overview

The goal is to classify images of recyclables into three categories using a **transfer learning approach** with MobileNetV2. The workflow includes:

1. **Data loading**: Load images from folders with automatic labeling.
2. **Model training**: Train a MobileNetV2-based classifier.
3. **Model conversion**: Convert the trained model to TFLite (float32 and quantized versions).
4. **Evaluation**: Evaluate TFLite models on the dataset.
5. **Deployment**: Ready for edge deployment.

---

## Dataset

- Folder structure:
recyclables_dataset/
plastic/
metal/
paper/

- Images are resized to **128x128** for training and evaluation.
- Each folder corresponds to a class.

> **Note**: Do not include large datasets in GitHub. Keep them locally.

---

## Project Structure
wk-6/
│
├─ train_model.py # Train MobileNetV2 model
├─ convert_tflite.py # Convert Keras model to TFLite
├─ evaluate_tflite.py # Evaluate TFLite model accuracy
├─ raspi_infer.py 
├─ recyclables_dataset/ # Dataset folder (ignored in git)
├─ tf_env/ # Python virtual environment (ignored in git)
└─ README.md


---

## Environment Setup
git clone https://github.com/KIRUMAH-sam/wk-6_future.git
cd wk-6_future

1. Create and activate virtual environment:
```bash
python -m venv tf_env
# Windows
.\tf_env\Scripts\activate
# Mac/Linux
source tf_env/bin/activate
pip install -r requirements.txt
Ensure dataset folder recyclables_dataset exists with proper subfolders.

Training the Mode
Run train_model.py:
python train_model.py
convert:
python convert_tflite.py
evaluate:
python evaluate_tflite.py
inference:
python raspi_infer.py
```
