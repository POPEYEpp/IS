from ultralytics import YOLO

if __name__ == "__main__":
    # โหลดโมเดล YOLOv8
    model = YOLO("yolov8n.pt")  # หรือใช้ yolov8s.pt, yolov8m.pt ตามต้องการ

    # เทรนโมเดล
    results = model.train(data="C:/Users/User/Desktop/IS2/pants-model-5/data.yaml", epochs=50, imgsz=640,batch=32, optimizer="AdamW",lr0=0.001,lrf=0.01)