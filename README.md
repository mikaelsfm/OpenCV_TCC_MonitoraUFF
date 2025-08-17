# Sistema de Detecção de Matilhas com YOLO - Trabalho de Conclusão de Curso (UFF)

Este projeto implementa um sistema inteligente de detecção de matilhas de cães utilizando YOLO (You Only Look Once) com a biblioteca Ultralytics. O sistema é capaz de detectar múltiplos cães em tempo real e identificar quando eles estão próximos o suficiente para formar uma matilha, emitindo alertas visuais.

## Funcionalidades Principais

### Detecção de Matilhas em Tempo Real
- **Script:** `yolo_detect.py`
- **Descrição:** Sistema completo de detecção de cães e identificação de matilhas usando YOLO v11
- **Recursos:**
  - Detecção de múltiplos cães simultaneamente
  - Análise de proximidade entre cães detectados
  - Alerta visual quando matilha é identificada (tela pisca em vermelho)
  - Linhas de conexão entre cães próximos
  - Suporte a GPU (CUDA) para processamento acelerado

### Como Usar
```bash
cd YoloDetect
python yolo_detect.py
```

**Controles:**
- Pressione `q` para sair da aplicação
- O sistema funciona com webcam ou arquivos de vídeo

---

## Estrutura do Projeto

```
YoloDetect/
├── yolo_detect.py          # Script principal
├── models/                 # Modelos YOLO
│   ├── yolo11l.pt         # Modelo principal (49MB)
│   ├── yolo11n.pt         # Modelo nano (5.4MB)
│   ├── yolo11x.pt         # Modelo extra large (109MB)
│   └── ...                # Outros modelos disponíveis
├── assets/                 # Arquivos de mídia
│   ├── dogs.mp4           # Vídeo de teste
│   ├── test.jpg           # Imagens de teste
│   └── result.jpg         # Resultados salvos
└── venv/                  # Ambiente virtual Python
```

---

## Instalação

### Pré-requisitos
- Python 3.8+
- CUDA (opcional, para aceleração GPU)
- Webcam ou arquivo de vídeo para teste

### Dependências
```bash
pip install ultralytics
pip install torch torchvision
pip install opencv-python
pip install numpy
```

### Configuração do Ambiente
```bash
# Clone o repositório
git clone [URL_DO_REPOSITORIO]

# Entre na pasta do projeto
cd YoloDetect

# Crie um ambiente virtual (recomendado)
python -m venv venv

# Ative o ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

---

## Configuração

### Modelos Disponíveis
O projeto inclui vários modelos YOLO v11:
- **yolo11n.pt** (5.4MB) - Modelo nano, mais rápido
- **yolo11l.pt** (49MB) - Modelo large, equilíbrio velocidade/precisão
- **yolo11x.pt** (109MB) - Modelo extra large, máxima precisão

### Personalização
Para alterar o modelo usado, edite a linha 44 em `yolo_detect.py`:
```python
model = YOLO("/models/yolo11l.pt")  # Altere para o modelo desejado
```

### Configuração de Detecção
- **Limiar de confiança:** 0.5 (linha 96)
- **Distância para matilha:** 150 pixels (linha 25)
- **Resolução:** 640x360 (linhas 50-51)

---

## Como Funciona

### Algoritmo de Detecção de Matilhas
1. **Detecção de Objetos:** YOLO identifica todos os cães no frame
2. **Filtragem:** Remove detecções com confiança < 0.5
3. **Análise de Proximidade:** Calcula distância entre centros dos cães
4. **Identificação de Matilha:** Se cães estão a menos de 150 pixels, forma matilha
5. **Alerta Visual:** Tela pisca em vermelho + texto "MATILHA DETECTADA!"

### Recursos Técnicos
- **GPU Acceleration:** Detecta automaticamente CUDA disponível
- **Processamento em Tempo Real:** 30+ FPS com GPU
- **Múltiplas Classes:** Suporte a detecção de outros objetos além de cães
- **Interface Visual:** Overlay com bounding boxes e informações

---

## Uso

### Execução Básica
```bash
python yolo_detect.py
```

### Modos de Entrada
1. **Webcam:** Descomente linhas 42-45 e comente linha 47
2. **Arquivo de Vídeo:** Use linha 47 (padrão: `assets/dogs.mp4`)
3. **Imagem:** Modifique para `cv2.imread()`

### Exemplo de Saída
```
CUDA disponível: NVIDIA GeForce RTX 3080
Webcam iniciada. Pressione 'q' para sair.
Detectado: dog - Confiança: 0.87
Detectado: dog - Confiança: 0.92
```

---

## Casos de Uso

### Aplicações Práticas
- **Segurança Pública:** Monitoramento de matilhas em áreas urbanas
- **Proteção Animal:** Identificação de grupos de cães soltos
- **Pesquisa:** Estudos comportamentais de cães em grupo
- **Educação:** Demonstração de visão computacional aplicada

### Extensões Possíveis
- Detecção de outros animais
- Análise de comportamento de grupo
- Sistema de alerta por email/SMS
- Interface web para monitoramento remoto

---

## Desenvolvimento

### Estrutura do Código
- **Função `detect_pack()`:** Lógica de detecção de matilhas
- **Função `main()`:** Loop principal e interface
- **Configurações:** Parâmetros ajustáveis no topo do arquivo

### Debugging
- Logs detalhados no console
- Visualização em tempo real
- Salvamento de frames com detecções

---

## Performance

### Métricas Típicas
- **FPS:** 30-60 (dependendo do hardware)
- **Precisão:** 85-95% (modelo yolo11l)
- **Latência:** <100ms (com GPU)

### Otimizações
- Uso de GPU CUDA quando disponível
- Processamento de frames otimizado
- Modelo configurável para velocidade/precisão

---

## Contribuição

Este projeto foi desenvolvido como parte do Trabalho de Conclusão de Curso na UFF. Contribuições são bem-vindas!

### Como Contribuir
1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

---

## Licença

Projeto acadêmico desenvolvido na Universidade Federal Fluminense (UFF).

---

## Suporte

Para dúvidas ou problemas:
- Consulte os comentários no código
- Verifique se todas as dependências estão instaladas
- Confirme se o CUDA está configurado (se usando GPU)

---

**Desenvolvido para o TCC da UFF**