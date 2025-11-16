## Sistema de Detecção com YOLO — TCC (UFF)

Este projeto implementa um sistema inteligente de **detecção de objetos** utilizando YOLO (You Only Look Once) com a biblioteca Ultralytics.  
O sistema é capaz de identificar múltiplos objetos em tempo real, exibindo as detecções com bounding boxes e informações de confiança.  
Como extensão, o projeto inclui uma funcionalidade específica para **detecção de matilhas de cães**, que identifica quando eles estão próximos o suficiente para formar um grupo.

---

## Funcionalidades Principais

### Detecção em Tempo Real (implementação atual)
- Implementação principal em C++ usando OpenCV DNN e modelos ONNX.
- Arquivos C++ realizam pré-processamento, inferência e pós-processamento (NMS, desenho de bounding boxes).
- Suporte a aceleração por CUDA quando disponível (via backend/target do OpenCV DNN).

### Detecção de Matilhas (feature adicional)
- Lógica que agrupa detecções da classe `dog` por proximidade e sinaliza quando há um agrupamento (matilha).
- Implementação no código C++ na função `detect_pack` (pode ser ajustada conforme necessidade).


---

## Estrutura do Projeto

```
YoloDetect/
├── yolo_detect.cpp         # Implementações C++ por imagem
├── yolo_detect_video.cpp   # Detector para webcam/arquivo
├── yolo_detect_rtsp.cpp    # Variante para entrada RTSP
├── exportYolo.py           # Script Python (Ultralytics) para exportar modelos para ONNX (opcional)
├── server.py               # Gateway Flask — recebe upload ou RTSP e executa os binários C++
├── models/                 # Modelos YOLO (.pt / .onnx) e coco.names
├── assets/                 # Vídeos/imagens de teste
└── build/                  # Saída do cmake / executáveis
```

---

## Instalação

### Pré-requisitos
- Sistema com compilador C++ (g++/clang) e CMake
- OpenCV (com módulo dnn). Para aceleração por GPU, use uma build do OpenCV com suporte a CUDA
- Python 3.8+ (apenas para o `server.py` e para o utilitário `exportYolo.py`, se for usar)
- Webcam ou arquivo de vídeo para teste

### Dependências Python (para server/export — opcionais)
```bash
python3 -m venv venv
source venv/bin/activate
pip install ultralytics flask werkzeug
# se quiser manipular/rodar inferência em Python (opcional):
pip install torch torchvision opencv-python numpy
```

## Construir os binários C++ (OpenCV + ONNX)

Os exemplos C++ usam CMake. No diretório `YoloDetect/` execute:

```bash
cd YoloDetect
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

Isso produz executáveis (por exemplo `yolo_detect_video`) em `YoloDetect/build/`.

Observações:
- Os exemplos esperam encontrar o modelo ONNX em `YoloDetect/models/` (por exemplo `yolov8l.onnx` ou `yolo11l.onnx`). Ajuste o caminho no código se necessário.
- O C++ tenta usar backend CUDA/target CUDA quando disponível; caso contrário usa CPU.

## Rodando o detector (binário C++)

Exemplo para webcam (executável `yolo_detect_video`):

```bash
./build/yolo_detect_video
```

Exemplo para arquivo de vídeo (quando o executável aceitar caminho):

```bash
./build/yolo_detect_video ../assets/dogs.mp4
```

O programa desenha bounding boxes, calcula uma métrica simples para "matilha" (detecta grupos de cães) e salva resultados periodicamente em `result_video.json`.

## Gateway HTTP (Flask)

O arquivo `YoloDetect/server.py` oferece um endpoint simples para enviar vídeos ou uma URL RTSP para processamento.

Endpoints principais:
- POST `/run_yolo` — aceita um campo `rtsp` (URL) ou um `file` multipart para upload.

Exemplo (upload de arquivo):

```bash
curl -F "file=@/caminho/para/video.mp4" http://localhost:5005/run_yolo
```

Exemplo (usar RTSP):

```bash
curl -X POST -F "rtsp=rtsp://usuario:senha@ip:porta/stream" http://localhost:5005/run_yolo
```

O servidor executa o binário (definido na variável `YOLO_BIN` em `server.py`) e tenta ler um arquivo `result.json` gerado pelo processo para devolver as detecções no corpo da resposta.

Para rodar o servidor:

```bash
python3 YoloDetect/server.py
```

## Exportar modelos com Ultralytics (script)

O `YoloDetect/exportYolo.py` é um exemplo mínimo que carrega um modelo `.pt` com a API Ultralytics e exporta para ONNX:

```python
from ultralytics import YOLO
model = YOLO("models/yolov8x.pt")
model.export(format="onnx", opset=13, simplify=True, dynamic=False)
```

Use esse script para gerar os `.onnx` necessários pelos exemplos C++.

## Modelos incluídos

Em `YoloDetect/models/` há exemplos como:

- `yolov8l.pt`, `yolov8x.pt`, `yolov8l.onnx`, `yolov8x.onnx`
- `yolo11l.pt`, `yolo11n.pt`, `yolo11l.onnx`, etc.
- `coco.names` — classes padrão

Substitua ou force o caminho do modelo no código conforme sua preferência.

## Detecção de "matilha" (pack detection)

Uma rotina simples nos exemplos calcula o centro de cada detecção de classe `dog`, mede distâncias e decide quando vários cães estão próximos (parâmetros estão hard-coded no código C++ — por exemplo proximidade = 0.5 * min(width,height) do frame). Você pode ajustar esse critério no código fonte (função `detect_pack`).

## Boas práticas e dicas

- Para melhor performance use OpenCV com suporte CUDA e um ONNX otimizado para sua GPU.
- Se usar o gateway Flask em produção, proteja o endpoint e use um gerenciador WSGI (gunicorn/uvicorn) e limites de tempo/recursos.
- Ao gerar ONNX com `exportYolo.py`, confirme o `opset` e execute uma inferência rápida para validar a compatibilidade com sua versão do OpenCV.

## Como contribuir

1. Fork
2. Crie uma branch descriptiva
3. Abra um Pull Request com mudanças pequenas e documentadas

Sugestões de melhorias:
- adicionar suporte direto a modelos PyTorch em Python
- endpoint autenticado e fila de jobs para processamentos longos
- painel web com visualização de detecções em tempo real

## Licença

Projeto acadêmico (TCC UFF). Consulte o autor/maintainer para detalhes sobre uso e redistribuição.

---

**Desenvolvido para o TCC da UFF**
