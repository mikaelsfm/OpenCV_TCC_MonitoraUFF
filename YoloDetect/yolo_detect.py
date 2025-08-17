from ultralytics import YOLO
import torch
import cv2
import numpy as np
import random

def detect_pack(boxes, class_ids, model):
    # Filtra apenas cães
    dog_indices = [i for i, cls in enumerate(class_ids) if model.names[cls] == "dog"]

    if len(dog_indices) < 2:
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
        for j in range(i+1, len(centers)):
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
    model = YOLO("/models/yolo11l.pt")

    # # Inicializa a captura da webcam
    # cap = cv2.VideoCapture(2)  # 0 para webcam padrão

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Inicializa a captura da webcam
    cap = cv2.VideoCapture('assets/dogs.mp4')  # 0 para webcam padrão

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # Verifica se a webcam foi aberta corretamente
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam")
        return

    print("Webcam iniciada. Pressione 'q' para sair.")

    while True:
        # Captura um frame da webcam
        ret, frame = cap.read()

        if not ret:
            print("Erro: Não foi possível capturar frame da webcam")
            break

        # Aplica o modelo YOLO no frame, forçando uso na GPU
        results = model(frame, device=device)

        # Processa os resultados
        for result in results:

            boxes = []
            class_ids = []
            # Desenha as detecções no frame
            result = results[0]
            # Separado para fazer só um desenho por frame
            annotated_frame = result.plot()

            # Exibe informações das detecções no console
            if result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    # Obtém as coordenadas da caixa
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Obtém a confiança
                    confidence = box.conf[0].cpu().numpy()
                    # Ignora detecções fracas
                    if confidence < 0.5: 
                        continue
                    # Obtém a classe
                    class_id = int(box.cls[0].cpu().numpy())
                    class_ids.append(class_id)
                    class_name = model.names[class_id]

                    print(f"Detectado: {class_name} - Confiança: {confidence:.2f}")

                is_pack, near_dogs = detect_pack(boxes, class_ids, model)
                if is_pack:
                    # Faz a tela piscar em vermelho para alertar sobre matilha
                    number = random.randint(0,100)
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0, 0, 255), -1)
                    if number % 2 == 0:
                        alpha = 0.3  # transparência
                    else:
                        alpha = 0.0
                    annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)

                    # Renderiza texto de alerta na exibição da câmera
                    cv2.putText(
                        annotated_frame,
                        "MATILHA DETECTADA!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA
                    )
                   # Desenhar linhas entre cães próximos
                for (c1, c2) in near_dogs:
                    cv2.line(annotated_frame, c1, c2, (0, 0, 255), 2)

        # Exibe o frame com as detecções
        cv2.imshow("YOLO Webcam Detection", annotated_frame)

        # Aguarda por uma tecla (1ms)
        key = cv2.waitKey(10) & 0xFF

        # Se 'q' for pressionado, sai do loop
        if key == ord("q"):
            break

    # Libera recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam fechada.")


if __name__ == "__main__":
    main()
