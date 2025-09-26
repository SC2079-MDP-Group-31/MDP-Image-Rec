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
- Obtain Model Weights From The Provided GDrive Link : https://drive.google.com/drive/folders/1gYvWR5__JVK6osUoonCY5rmd46knZXbt?usp=drive_link
- Place Model Weight Folders Under `models/` directory

**Add Required Directories**
```bash
mkdir -p models inputs outputs
```

**Run Inference**
```bash
python src/infer.py
```

---

### API Endpoints

**Accessing End Points**
`http://{PC_IP_ADDRESS}:5000/{ENDPT}`

**Pathing Algorithm Endpoint**
`http://{PC_IP_ADDRESS}:5000/path`

- Use Case: Call to obtain the machine pathing instructions

- Request Parameters : (String)

- Response : `{predictions_path : COMMANDS }`

Example:

*INPUT*:
`"ALG:11,0,E,0;19,4,W,1;19,10,W,2;7,7,W,3;7,17,S,4;0,14,S,5;14,18,W,6;11,9,E,7;"`

*OUTPUT:*
```py
{
  "predictions_path": [
    "SF040",
    "RF090",
    "SB010",
    "SCAN_3",
    "SF010",
    "RB090",
    "SF050",
    "SCAN_5",
    "SB020",
    "RF090",
    "SF010",
    "LF090",
    "SCAN_4",
    "SF010",
    "RF090",
    "RF090",
    "SB020",
    "RB090",
    "SCAN_6",
    "RF090",
    "SF030",
    "LF090",
    "SCAN_2",
    "SB010",
    "RF090",
    "SF060",
    "RB090",
    "SF010",
    "SCAN_1",
    "SB030",
    "LF090",
    "SB030",
    "RB090",
    "SF010",
    "SCAN_0",
    "SB010",
    "RF090",
    "SF080",
    "RB090",
    "SF010",
    "SCAN_7"
  ]
}
```

**Image Recognition Endpoint**
`http://{PC_IP_ADDRESS}:5000/image`

- Use Case: Call to obtain class prediction for a captured image

- Request Parameters : (Img File)

- Response : `{predictions : PREDICTIONS }`

Example:

*INPUT*:

<img src="assets/10_jpg.rf.a69c59fd2fe4d8d72a021ebba2a9acd4.jpg" alt="Input Image" width="400"/>

*OUTPUT:*
```py
{
  "predictions": [
    {
      "class_id": 10,
      "image_id": "21", # This Maps To The Detected Value
      "confidence": 0.894066572189331
    }
  ]
}
```

**Image Stitching Endpoint**
`http://{PC_IP_ADDRESS}:5000/stitch`

- Use Case: Call to display stitched image on PC consisting of currently detected images

- Request Parameters : N/A

- Response : `{"message": "Stitched image displayed."}`

---

### Serving Model On LocalHost

**Start Up Uvicorn Server On Local Machine**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 5000
```

Test API at (http://localhost:5000/docs)

- `0.0.0.0:5000` : Means that the server can be accessed by any machine that can reach your machine over the network *(same Wi-Fi, LAN)*
- Access Via `http://<server_machine_local_ip>:5000`
---

### Local To Public Server Hosting
Ngrok exposes the local server to the public via a secure tunnel

**Installing Ngrok**
```bash
pip install pyngrok
```

**Inputting Ngrok API Key**
```bash
ngrok config add-authtoken $YOUR_AUTHTOKEN
```

**Start Up Uvicorn Server On Local Machine**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 5000
```

**Start Up Ngrok Tunneling**
```bash
ngrok http 5000
```

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

---
