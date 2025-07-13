import cv2
import numpy as np

# Carregando a imagem
imagem = cv2.imread('Fotos/imagem5.jpg', cv2.IMREAD_COLOR)

if imagem is not None:
    # Pré-processamento
    altura, largura = imagem.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(imagem, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    # Carrega a rede treinada para faces
    net = cv2.dnn.readNetFromCaffe(
        'Modelos/deploy.prototxt',
        'Modelos/res10_300x300_ssd_iter_140000.caffemodel'
    )

    # Carrega classificador Haar Cascade para olhos
    carregaOlho = cv2.CascadeClassifier('Haarcascade/haarcascade_eye.xml')

    net.setInput(blob)
    detections = net.forward()

    # Percorre as detecções de faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([largura, altura, largura, altura])
            (x1, y1, x2, y2) = box.astype("int")

            cv2.rectangle(imagem, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            localOlho = imagem[y1:y2, x1:x2]
            if localOlho.size > 0:
                localOlhoCinza = cv2.cvtColor(localOlho, cv2.COLOR_BGR2GRAY)
                detectado = carregaOlho.detectMultiScale(localOlhoCinza)

                for(ox, oy, ol, oa) in detectado:
                    cv2.rectangle(localOlho, (ox, oy), (ox + ol, oy + oa), (0, 0, 255), 2)

    cv2.imshow("Detecta Face e os Olhos (DNN)", imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Erro: Não foi possível carregar a imagem")