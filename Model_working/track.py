from ultralytics import YOLO

model = YOLO(
    r"E:\Models\MultiObjectDetection\runs\detect\VisDrone_YOLO26n-3\weights\best.pt"
)
results = model.track(
    source=r"G:\My Drive\YOLODATASET\yt2.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    conf=0.25,
    save=True,
    show=True
)