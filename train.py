from ultralytics import YOLO

# model = YOLO() # to download the model
model = YOLO('Models/yolov8n.pt')  # if already downloaded
model.train(data='DataSet/SplitData/data.yaml', epochs=10)


