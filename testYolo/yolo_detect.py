from ultralytics import YOLO
import torch
import cv2
import numpy as np


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

    # Inicializa a captura da webcam
    cap = cv2.VideoCapture(2)  # 0 para webcam padrão

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
                    # if confidence < 0.5: 
                    #     continue
                    # Obtém a classe
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]

                    print(f"Detectado: {class_name} - Confiança: {confidence:.2f}")

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
