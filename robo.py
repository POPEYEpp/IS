from roboflow import Roboflow
rf = Roboflow(api_key="MqGYxcrvqJZA6sdqPror")
project = rf.workspace("exscan").project("shoes-model-ey9vd")
version = project.version(1)
dataset = version.download("yolov8")
                