from ultralytics import YOLO

# Baixa o modelo oficial YOLOv8 Large (pode trocar por yolov8m, yolov8n, etc)
model = YOLO("models/yolov8x.pt")

# Exporta para formato ONNX compatível com OpenCV
model.export(format="onnx", opset=13, simplify=True, dynamic=False)