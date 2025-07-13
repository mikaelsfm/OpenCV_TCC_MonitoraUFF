import cv2
import numpy as np

webcamera = cv2.VideoCapture(2, cv2.CAP_DSHOW)

webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

net = cv2.dnn.readNetFromCaffe(
    'Modelos/deploy.prototxt',
    'Modelos/res10_300x300_ssd_iter_140000.caffemodel'
)

while True:
    camera, frame = webcamera.read()
    
    if frame is not None:
        altura, largura = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([largura, altura, largura, altura])
                (x1, y1, x2, y2) = box.astype("int")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{confidence*100:.1f}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Video WebCamera (DNN)", frame)

    if cv2.waitKey(1) == ord('f'):
        break

webcamera.release()
cv2.destroyAllWindows()
