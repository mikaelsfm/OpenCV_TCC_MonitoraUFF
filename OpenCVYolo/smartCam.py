import numpy as np
import cv2
import time
import os
from pyzm.api import ZMApi

ZM_HOST = 'rtsp://192.168.15.10:8085/live'
ZM_USER = 'none'
ZM_PASS = 'none'
MONITOR_ID = 2
YOLO_PATH = 'OpenCVYolo/yoloDados'
PROBABILITY_MINIMUM = 0.5
THRESHOLD = 0.3
INTERESTING_OBJECTS = ['person', 'car', 'motorcycle', 'dog', 'sofa', 'cell phone']
MODEL_NAMES_PATH = os.path.join(YOLO_PATH, 'YoloNames.names')
CONFIG_PATH = os.path.join(YOLO_PATH, 'yolov3.cfg')
WEIGHTS_PATH = os.path.join(YOLO_PATH, 'yolov3.weights')

def conectar_api_zm():
    print(f"INFO: Conectando à API do ZoneMinder em {ZM_HOST}...")
    try:
        zm = ZMApi(options={'apiurl': f'{ZM_HOST}/api'})
        monitor = zm.get_monitor(MONITOR_ID)
        print("Conectado à API do ZoneMinder com sucesso.")
        return zm, monitor
    except Exception as e:
        print(f"ERRO ao conectar API do ZoneMinder: {e}")
        return None, None

def abrir_stream_video():
    if ZM_USER and ZM_PASS:
        stream_url = f"http://{ZM_USER}:{ZM_PASS}@{ZM_HOST.replace('http://','')}/cgi-bin/nph-zms?monitor={MONITOR_ID}&scale=100"
    else:
        stream_url = f"{ZM_HOST}/cgi-bin/nph-zms?monitor={MONITOR_ID}&scale=100"
    print(f"INFO: Tentando abrir stream de vídeo: {stream_url}")
    cam = cv2.VideoCapture(stream_url)
    if not cam.isOpened():
        print("ERRO: Não foi possível abrir o stream de vídeo.")
        return None
    print("Stream de vídeo conectado com sucesso.")
    return cam

print("INFO: Carregando nomes dos objetos YOLO...")
with open(MODEL_NAMES_PATH) as f:
    labels = [line.strip() for line in f]
print("INFO: Carregando modelo YOLOv3...")
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
network = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
layers_names_all = network.getLayerNames()
try:
    layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]
except IndexError:
    layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
print("YOLO carregado com sucesso.")

zm, monitor = conectar_api_zm()
camera = abrir_stream_video()
if camera is None:
    print("ERRO FATAL: Não é possível continuar sem vídeo.")
    exit()

h, w = None, None
cv2.namedWindow('YOLOv3 - ZoneMinder', cv2.WINDOW_NORMAL)
print("\nINFO: Loop de detecção iniciado. Pressione 'q' para sair.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("AVISO: Frame não recebido. Tentando reconectar em 5 segundos...")
        time.sleep(5)
        camera.release()
        camera = abrir_stream_video()
        if camera is None:
            print("ERRO FATAL: Reconexão falhou. Saindo...")
            break
        continue
    if w is None or h is None:
        h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    network.setInput(blob)
    output_from_network = network.forward(layers_names_output)
    bounding_boxes = []
    confidences = []
    class_numbers = []
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
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, PROBABILITY_MINIMUM, THRESHOLD)
    if len(results) > 0:
        for i in results.flatten():
            detected_label = labels[class_numbers[i]]
            if detected_label in INTERESTING_OBJECTS and zm and monitor:
                try:
                    print(f"Objeto '{detected_label}' detectado. Acionando alarme...")
                    monitor.force_alarm()
                except Exception as e:
                    print(f"ERRO ao acionar alarme: {e}")
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box_current = [int(c) for c in colours[class_numbers[i]]]
            cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current, 2)
            text_box_current = f'{detected_label}: {confidences[i]:.2f}'
            cv2.putText(frame, text_box_current, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour_box_current, 2)
    cv2.imshow('YOLOv3 - ZoneMinder', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("\nINFO: Encerrando...")
camera.release()
cv2.destroyAllWindows()
print("Recursos liberados. Programa finalizado.")
