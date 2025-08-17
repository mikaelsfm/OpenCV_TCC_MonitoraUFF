import numpy as np
import cv2
import time
import os

# Caminho para a pasta com os ficheiros do YOLO.
# Garanta que esta pasta 'yoloDados' está no mesmo diretório que o seu script.
YOLO_PATH = 'yoloDados'
MODEL_NAMES_PATH = os.path.join(YOLO_PATH, 'YoloNames.names')
CONFIG_PATH = os.path.join(YOLO_PATH, 'yolov3.cfg')
WEIGHTS_PATH = os.path.join(YOLO_PATH, 'yolov3.weights')

# Definições de confiança e limiar para a deteção
PROBABILITY_MINIMUM = 0.5
THRESHOLD = 0.3

with open(MODEL_NAMES_PATH) as f:
    labels = [line.strip() for line in f]


colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
network = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)

# try:
#     print("A verificar a disponibilidade da GPU...")
#     network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#     network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#     print("Backend configurado para usar a GPU (CUDA).")
# except cv2.error as e:
#     print(f"Não foi possível configurar o backend para CUDA: {e}")
#     print("A usar a CPU.")


layers_names_all = network.getLayerNames()
try:
    layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]
except IndexError:
    layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]


camera = cv2.VideoCapture("rtsp://192.168.15.10:8085/live")
if not camera.isOpened():
    print("Câmara 2 não encontrada. A tentar a câmara padrão (0)...")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Erro: Nenhuma câmara foi encontrada.")
        exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
h, w = None, None

cv2.namedWindow('YOLO v3 WebCamera', cv2.WINDOW_NORMAL)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Obtém as dimensões do frame apenas uma vez
    if w is None or h is None:
        h, w = frame.shape[:2]

    # Cria um blob a partir do frame para alimentar a rede
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Define o blob como entrada para a rede e faz a previsão
    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Listas para guardar as informações das deteções
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Itera sobre os resultados da rede
    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current > PROBABILITY_MINIMUM:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    # Aplica a Supressão Não-Máxima para eliminar caixas sobrepostas e redundantes
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, PROBABILITY_MINIMUM, THRESHOLD)

    # Desenha os resultados finais na imagem
    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Obtém a cor para a classe atual
            colour_box_current = [int(c) for c in colours[class_numbers[i]]]

            # Desenha o retângulo
            cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current, 2)

            # Prepara o texto com o nome do objeto e a confiança
            text_box_current = f'{labels[class_numbers[i]]}: {confidences[i]:.2f}'

            # Coloca o texto na imagem
            cv2.putText(frame, text_box_current, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour_box_current,
                        2)

    cv2.imshow('YOLO v3 WebCamera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("A encerrar o programa.")
camera.release()
cv2.destroyAllWindows()
