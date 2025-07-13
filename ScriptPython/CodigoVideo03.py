import cv2
import numpy as np

webcamera = cv2.VideoCapture(2, cv2.CAP_DSHOW)

webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not webcamera.isOpened():
    print("Erro ao abrir a câmera")
    exit()

net = cv2.dnn.readNetFromCaffe(
    'Modelos/deploy.prototxt',
    'Modelos/res10_300x300_ssd_iter_140000.caffemodel'
)

classificadorOlho = cv2.CascadeClassifier("Haarcascade/haarcascade_eye.xml")

while True:
    ret, frame = webcamera.read()

    if not ret or frame is None:
        print("Erro ao capturar frame")
        continue

    altura, largura = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))

    net.setInput(blob)
    detections = net.forward()

    # Percorre as detecções de faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([largura, altura, largura, altura])
            (x1, y1, x2, y2) = box.astype("int")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Região da face para detectar olhos
            pegaOlho = frame[y1:y2, x1:x2]
            if pegaOlho.size > 0:
                OlhoCinza = cv2.cvtColor(pegaOlho, cv2.COLOR_BGR2GRAY)
                localizaOlho = classificadorOlho.detectMultiScale(OlhoCinza)
                for (ox, oy, ol, oa) in localizaOlho:
                    cv2.rectangle(pegaOlho, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

    cv2.imshow("Video WebCamera (DNN)", frame)

    if cv2.waitKey(1) == ord('q'):
        break

webcamera.release()
cv2.destroyAllWindows()
