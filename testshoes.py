from ultralytics import YOLO

# โหลดโมเดลที่เทรนเสร็จแล้ว
model = YOLO("runs/detect/train/weights/bestshoes.pt")
model.conf = 0.5
# ใส่ path ของภาพที่ต้องการทดสอบ
# results = model("shoes-model-1/valid/images/760691518_506348_mp4-0184_jpg.rf.973427827a0876c924b667a4ce4ac869.jpg", save=True, show=True)
# results = model("shoes-model-1/valid/images/760691262_959204_mp4-0012_jpg.rf.3d158cbf8d3783f148b0b90126b14f75.jpg", save=True, show=True)
# results = model("shoes-model-1/train/images/760691184_466676_mp4-0049_jpg.rf.c33328fdd9ffefb5c4b0c9b3b78e6190.jpg", save=True, show=True)
# results = model("PT/resized_1.jpg", save=True, show=True)
# results = model("PT/resized_2.jpg", save=True, show=True)
results = model("PT/IMG_5878.mp4", save=True, show=True, conf=0.6)