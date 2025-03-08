from ultralytics import YOLO

# โหลดโมเดลที่เทรนเสร็จแล้ว
model = YOLO("runs/detect/train3/weights/bestpants.pt")

# ใส่ path ของภาพที่ต้องการทดสอบ และกำหนดค่า conf=0.1
# results = model("pants-model-5/test/images/760691262_959204_mp4-0078_jpg.rf.8756822be266fe223e214fd3ddc0aa84.jpg", save=True, show=True, conf=0.6)
# results = model("pants-model-5/valid/images/760691184_466676_mp4-0137_jpg.rf.7a4059a8eb83d2ab8250dcc4dd4f9469.jpg", save=True, show=True, conf=0.6)
# results = model("pants-model-5/train/images/760691184_466676_mp4-0017_jpg.rf.33acc2a6445ccb7fad63310c160bda57.jpg", save=True, show=True, conf=0.6)
# results = model("PT/1.jpg", save=True, show=True, conf=0.6)
# results = model("PT/2.jpg", save=True, show=True, conf=0.6)
results = model("PT/IMG_5878.mp4", save=True, show=True, conf=0.6)
