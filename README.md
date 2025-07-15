# Aplicação do OpenCV no Trabalho de Conclusão de Curso (UFF)

Este repositório reúne diversos exemplos e aplicações de Visão Computacional utilizando OpenCV, MediaPipe e modelos de Deep Learning, voltados para detecção de faces, objetos e comparação de imagens. O projeto foi desenvolvido como parte do Trabalho de Conclusão de Curso na UFF.

## Principais Funcionalidades

### 1. Detecção de Faces com Haarcascade
- **Script:** `Opencv_Contador_faces.py`
- **Descrição:** Utiliza o classificador Haarcascade para detectar e contar faces em tempo real via webcam.
- **Como usar:**  
  ```bash
  python Opencv_Contador_faces.py
  ```
- **Requisito:** Pasta `Haarcascade` com os arquivos `.xml` (especialmente `haarcascade_frontalface_default.xml`).

---

### 2. Detecção de Objetos com YOLOv3
- **Script:** `OpenCVYolo/YoloWebCamera.py`
- **Descrição:** Detecta múltiplos objetos em tempo real usando YOLOv3 e exibe as classes detectadas na webcam.
- **Como usar:**  
  ```bash
  cd OpenCVYolo
  python YoloWebCamera.py
  ```
- **Requisito:**  
  - Arquivos `yolov3.weights`, `yolov3.cfg` e `YoloNames.names` na pasta `OpenCVYolo/yoloDados/`.
  - O arquivo `yolov3.weights` não está no repositório devido ao tamanho. Baixe em: https://pjreddie.com/media/files/yolov3.weights

---

### 3. Detecção de Faces com DNN (Caffe)
- **Scripts:**  
  - `ScriptPython/Codigo_imagem01.py`  
  - `ScriptPython/CodigoImagem02.py`  
  - `ScriptPython/CodigoVideo01.py`  
  - `ScriptPython/CodigoVideo02.py`
- **Descrição:** Utilizam modelos Caffe para detectar faces em imagens e vídeo (webcam).
- **Como usar:**  
  ```bash
  python ScriptPython/Codigo_imagem01.py
  python ScriptPython/CodigoImagem02.py
  python ScriptPython/CodigoVideo01.py
  python ScriptPython/CodigoVideo02.py
  ```
- **Requisito:**  
  - Modelos `deploy.prototxt` e `res10_300x300_ssd_iter_140000.caffemodel` na pasta `Modelos/`.

---

### 4. Detecção Facial Avançada com MediaPipe
- **Script:** `ScriptPython/CodigoImagem03.py`
- **Descrição:** Detecta múltiplos rostos, contornos faciais e íris em imagens usando MediaPipe.
- **Como usar:**  
  ```bash
  python ScriptPython/CodigoImagem03.py
  ```
- **Requisito:**  
  - Biblioteca `mediapipe` instalada.
  - Imagem de entrada na pasta `Fotos/`.

---

### 5. Detecção de Objetos com MediaPipe (TFLite)
- **Script:** `ScriptPython/CodigoVideo03.py`
- **Descrição:** Detecta objetos em tempo real na webcam usando modelos TensorFlow Lite e MediaPipe.
- **Como usar:**  
  ```bash
  python ScriptPython/CodigoVideo03.py
  ```
- **Requisito:**  
  - Modelos `.tflite` na pasta `Modelos/`.

---

### 6. Comparação de Similaridade entre Imagens
- **Script:** `ImageComparisonSimilarity.py`
- **Descrição:** Compara duas imagens usando o algoritmo ORB do OpenCV e retorna o percentual de similaridade.
- **Como usar:**  
  - Ajuste os caminhos das imagens no script.
  - Execute:
    ```bash
    python ImageComparisonSimilarity.py
    ```

---

## Instalação

### Dependências Principais

- Python 3.x
- OpenCV (`pip install opencv-python`)
- Numpy (`pip install numpy`)
- MediaPipe (`pip install mediapipe`)
- (Para YOLO) Baixe o arquivo `yolov3.weights` conforme instrução acima.

### Instalação do OpenCV a partir do código-fonte (Linux)

Veja o arquivo `Instalação OpenCV Source` para instruções detalhadas de instalação no Linux, incluindo dependências do sistema e compilação do OpenCV com módulos extras.

---

## Créditos e Referências

- Modelos e arquivos de configuração obtidos de fontes oficiais do OpenCV, YOLO e MediaPipe.
- Projeto desenvolvido para fins acadêmicos na UFF.

---

## Observações

- Para rodar scripts de webcam, certifique-se de que a câmera está conectada e disponível.
- Ajuste o índice da câmera (`cv2.VideoCapture(0)` ou `cv2.VideoCapture(2)`) conforme necessário.
- O projeto contém exemplos didáticos e pode ser expandido para outras aplicações de visão computacional.

---

Se precisar de mais detalhes ou exemplos de execução, consulte os comentários nos próprios scripts!