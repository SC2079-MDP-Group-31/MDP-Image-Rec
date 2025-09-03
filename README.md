# Introduction

## Backend Technology Stack
- 🐍 **Language**: Python

- 🧠 **AI Model Integration**: TensorFlow / PyTorch (image rec)

---

## Image Rec Backend Responsibilities
1. Accept image uploads

2. Resize → normalize → run inference on model

3. Return Image With Detected Output

---

## AI Model (Ingredient Detection)

### YOLOV11s-seg (Ultralytics)

**Type**: Object Detection (Instance Segmentation)
**Framework**: PyTorch

#### Pros:

- Very fast and lightweight (ideal for real-time or API-based inference)
- Easy to train and fine-tune on custom datasets
- Well-documented and active community

---

### Data Augmentation
Applied to prevent overfitting of data, prepare model for real world variations in images and improve model generalization to unseen data

- `Hue Shift Factor` : Randomly adjusts colour tone of image within a specified factor to account for variations in food ripeness, produce colour tone.
- `Saturation Jitter` : Randomly adjusts vividness of colour within a specified factor to account for different lighting settings when taking image.
- `Brightness Jitter` : Randomly makes images brighter/darker within a specified factor to account for different lighting settings when taking image.
---

### Required Libraries & Interpreters
- `python v3.11.0` : Python interpreter version for compatibility with installed libraries
- `ultralytics` : Python library to use YOLO object detection models
- `torch` : Deep learning framework that handles tensor operations, GPU accelerations & model layers/parameters
- `torchvision` : Pytorch companion library focused on computer vision tasks
- `torchaudio` : Pytorch companion library focused on audio processing tasks
- `opencv-python` : To load/resize/preprocess images
- `numpy` : Tensor operations
- `python-dotenv` : load `.env` config file

---

### Environment Setup
**Setup Virtual Environment**
```bash
py -3.10 -m venv venv
```

**Activate Virtual Environment**
```bash
.\venv\Scripts\Activate.ps1
```

**Install Required Libraries:**
```bash
pip install -r requirements.txt
```

**Install Torch With CUDA Enabled**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Install Model Weights & Yaml File**
- Obtain Model Weights From The Provided GDrive Link
- Place Model Weights Under `models/` directory

**Add Required Directories**
```bash
mkdir -p models inputs outputs
```

**Run Inference**
```bash
python src/infer.py
```

---

### Data Mapping
| Yaml ID | Name |
|---|---|
| 11 | **1** |
| 12 | **2** |
| 13 | **3** |
| 14 | **4** |
| 15 | **5** |
| 16 | **6** |
| 17 | **7** |
| 18 | **8** |
| 19 | **9** |
| 20 | **A** |
| 21 | **B** |
| 22 | **C** |
| 23 | **D** |
| 24 | **E** |
| 25 | **F** |
| 26 | **G** |
| 27 | **H** |
| 28 | **S** |
| 29 | **T** |
| 30 | **U** |
| 31 | **V** |
| 32 | **W** |
| 33 | **X** |
| 34 | **Y** |
| 35 | **Z** |
| 36 | **Up Arrow** |
| 37 | **Down Arrow** |
| 38 | **Right Arrow** |
| 39 | **Left Arrow** |
| 40 | **Stop** |

Serve Model : uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
Test API at (http://localhost:8000/docs)