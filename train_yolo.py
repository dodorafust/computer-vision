from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('models/yolo11n.pt')
    model.train(data='dataset.yaml', epochs=50, imgsz=640, device=0, workers=4)