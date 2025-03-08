from roboflow import Roboflow
rf = Roboflow(api_key="MqGYxcrvqJZA6sdqPror")
project = rf.workspace("exscan").project("pants-model")
version = project.version(5)
dataset = version.download("yolov8")
                