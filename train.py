from ultralytics import YOLO

model = YOLO('Models/yolov8n.pt')
model.train(data='DataSet/SplitData/data.yaml', epochs=10)


