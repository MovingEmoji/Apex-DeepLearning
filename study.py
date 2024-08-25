import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("myyolov8n.yaml")
    model = YOLO("myyolov8n.pt")

    results = model.train(data="mycoco.yaml", epochs=200, batch=64, imgsz=416)

    results = model.val()