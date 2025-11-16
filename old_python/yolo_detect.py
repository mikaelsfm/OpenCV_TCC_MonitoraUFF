from ultralytics import YOLO
import torch
import cv2
import numpy as np
import random
import sys
import json


def detect_pack(boxes, class_ids, model):
    # Filtra apenas cães
    dog_indices = [i for i, cls in enumerate(class_ids) if model.names[cls] == "dog"]

    if len(dog_indices) < 5:
        return False, []

    # Pega os centros das caixas dos cães
    centers = []
    for i in dog_indices:
        # Extrai as coordenadas da caixa
        x1, y1, x2, y2 = boxes[i].xyxy[0].cpu().numpy()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centers.append((cx, cy))

    # Verifica proximidade entre câes
    near = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            if dist < 150:
                near.append((centers[i], centers[j]))

    if len(near) > 0:
        return True, near
    return False, []

def main():
    # Verifica se CUDA está disponível
    if torch.cuda.is_available():
        print(f"CUDA disponível: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("CUDA não disponível, rodando na CPU")
        device = "cpu"


    # Carrega o modelo YOLO
    model = YOLO("models/yolo11l")
    model.to(device)

    image_path = 'assets/test.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem em: {image_path}")
        return

    print(f"Analisando a imagem: {image_path}")
    results = model(image, device=device)


    all_detections_json = []
    all_boxes_for_pack_detection = []
    all_class_ids_for_pack_detection = []


    result = results[0]
    annotated_frame = result.plot()
    
    if result.boxes is not None:
        for box in result.boxes:
            confidence = box.conf[0].cpu().numpy()
            # if confidence < 0.5:
            #     continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]
            
            all_detections_json.append({
                "class": class_name,
                "confidence": float(confidence),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
            
            all_boxes_for_pack_detection.append(box)
            all_class_ids_for_pack_detection.append(class_id)
    
    is_pack, _ = detect_pack(all_boxes_for_pack_detection, all_class_ids_for_pack_detection, model)

    if is_pack:
        print("\n------------------------------")
        print(">>> ALERTA: Matilha detectada na imagem! <<<")
        print("------------------------------\n")

    if all_detections_json:
        print("\n--- Resultado Final (JSON) ---")
        print(json.dumps({"detections": all_detections_json}, ensure_ascii=False, indent=2))
        print("------------------------------\n")

if __name__ == "__main__":
    main()