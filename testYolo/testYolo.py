from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model('test2.jpg')

for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    result.save(filename='result2.jpg')