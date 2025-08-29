from ultralytics import YOLO
import yaml
import sys
import os

def load_cfg():
    # load config
    with open("src/config.yaml", "r") as f:   # adjust path if config.yaml is elsewhere
        cfg = yaml.safe_load(f)
    
    return cfg

def main():
    # --- load config.yaml ---
    cfg = load_cfg()

    # --- adjust these paths ---
    WEIGHTS = cfg["weights"]   # path to your YOLOv11s-seg weights
    SOURCE = cfg["source"]
    SAVE_DIR = cfg["save_dir"]
    IMGSZ = cfg.get("imgsz", 640)
    CONF = cfg.get("conf", 0.25)
    IOU = cfg.get("iou", 0.6)
    DEVICE = cfg.get("device", None)

    # allow overriding from command line: python infer.py path/to/image.jpg
    if len(sys.argv) > 1:
        SOURCE = sys.argv[1]

    # make sure save dir exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # load model
    model = YOLO(WEIGHTS)

    # run prediction
    results = model.predict(
        source=SOURCE,
        imgsz=640,
        conf=CONF,
        iou=0.6,
        save=True,
        save_txt=True,
        save_conf=True,
        project=SAVE_DIR,
        name="pred",
        exist_ok=True
    )

    print(f"✅ Done! Results saved in: {os.path.join(SAVE_DIR, 'pred')}")

if __name__ == "__main__":
    main()