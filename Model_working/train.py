from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    workers=0,
    device=0,
    name="VisDrone_YOLO26n"
) 

