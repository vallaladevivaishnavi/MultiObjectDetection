import os
from PIL import Image

# Classes to keep
class_map = {
    1: 0,   # pedestrian
    2: 1,   # person
    4: 2,   # car
    5: 3,   # van
    6: 4,   # truck
    9: 5    # bus
}


def convert(annotation_dir, image_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)

    for file in os.listdir(annotation_dir):

        annotation_path = os.path.join(annotation_dir, file)
        image_path = os.path.join(image_dir, file.replace('.txt', '.jpg'))

        if not os.path.exists(image_path):
            continue

        img = Image.open(image_path)
        img_w, img_h = img.size

        yolo_labels = []

        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = line.strip().split(',')

            x = float(data[0])
            y = float(data[1])
            w = float(data[2])
            h = float(data[3])
            cls = int(data[5])

            if cls not in class_map:
                continue

            cls = class_map[cls]

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w /= img_w
            h /= img_h

            yolo_labels.append(
                f"{cls} {x_center} {y_center} {w} {h}"
            )

        output_file = os.path.join(label_dir, file)

        with open(output_file, 'w') as f:
            f.write('\n'.join(yolo_labels))


# Convert train dataset
convert(
    r"E:\Models\MultiObjectDetection\Datasets\ObjectDetectiondataset\VisDrone2019-DET-train\annotations",
    r"E:\Models\MultiObjectDetection\Datasets\ObjectDetectiondataset\VisDrone2019-DET-train\images",
    r"E:\Models\MultiObjectDetection\Datasets\ObjectDetectiondataset\labels\train"
)

# Convert validation dataset
convert(
    r"E:\Models\MultiObjectDetection\Datasets\ObjectDetectiondataset\VisDrone2019-DET-val\annotations",
    r"E:\Models\MultiObjectDetection\Datasets\ObjectDetectiondataset\VisDrone2019-DET-val\images",
    r"E:\Models\MultiObjectDetection\Datasets\ObjectDetectiondataset\labels\val"
)

print("Conversion completed successfully!")