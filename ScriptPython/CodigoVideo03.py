import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import object_detector
from mediapipe.tasks.python.vision.object_detector import ObjectDetector

def visualize(image, detection_result) -> np.ndarray:

    COLORS = {
        'person': (0, 255, 0),
        'dog': (0, 0, 255),
        'cat': (255, 0, 255),
        'cell phone': (0, 255, 255),
        'book': (255, 255, 0),
    }

    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)

        if probability < 0.7:
            continue

        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

        color = COLORS.get(category_name, (255, 0, 0))  # azul padrão

        cv2.rectangle(image, start_point, end_point, color, 3)

        result_text = f'{category_name} ({probability})'
        text_location = (bbox.origin_x + 10, bbox.origin_y + 30)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    2, color, 2)
    return image

base_options = python.BaseOptions(model_asset_path='../Modelos/ssd_mobilenet_v2.tflite')
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.75
)

detector = vision.ObjectDetector.create_from_options(options)

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Câmera 2 não encontrada. Tentando a câmera padrão (0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Nenhuma câmera foi encontrada.")
        exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler o quadro da câmera.")
        break

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Cria um objeto mp.Image a partir do array numpy do quadro
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detecta os objetos na imagem
    detection_result = detector.detect(mp_image)

    # Desenha os resultados na imagem original (BGR)
    annotated_image = visualize(frame, detection_result)

    # Mostra o resultado
    cv2.imshow("MediaPipe - Detecção de Objetos (Webcam)", annotated_image)

    # Encerra o loop se 'q' for pressionado
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

detector.close()

cap.release()
cv2.destroyAllWindows()
