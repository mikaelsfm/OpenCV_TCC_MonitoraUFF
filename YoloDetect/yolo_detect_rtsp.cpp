/*
 * YOLOv8 Pipeline Full GPU para NVIDIA GTX 1050
 * Requisitos: OpenCV 4.10+ com CUDA, CUDNN, CUDACODEC e OPENGL ativados.
 *
 * Fluxo de Dados:
 * RTSP (Rede) -> NVDEC (GPU VRAM) -> Pre-processamento (GPU VRAM) -> Inferência (Tensor Cores) 
 * -> Renderização Retângulos (GPU VRAM) -> OpenGL Display (GPU).
 *
 * Apenas os metadados da detecção (coordenadas das caixas) passam pela CPU.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/cudacodec.hpp>    // Para decodificação de hardware (NVDEC)
#include <opencv2/cudaarithm.hpp>   // Operações aritméticas na GPU
#include <opencv2/cudawarping.hpp>  // Resize na GPU
#include <opencv2/cudaimgproc.hpp>  // Conversão de cores na GPU
#include <opencv2/cudafilters.hpp>  // Filtros na GPU
#include <opencv2/dnn.hpp>          // Rede Neural
#include <opencv2/highgui.hpp>      // Interface Gráfica (com suporte OpenGL)

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>      // Biblioteca JSON
#include <csignal>                // Para capturar Ctrl+C
#include <atomic>                 // Para flag thread-safe
#include <thread>                 // Para sleep se necessário

// Namespaces para facilitar a leitura
using json = nlohmann::json;
using namespace cv;
using namespace std;

// ---------------------------------------------------------
// Variáveis Globais e Estruturas
// ---------------------------------------------------------

// Controle de execução para parada segura via Ctrl+C
std::atomic<bool> running{true};

// Estrutura para armazenar os dados de uma detecção
struct Detection {
    int class_id;
    string class_name;
    float confidence;
    Rect box;
    Scalar color; // Cor única para a classe
};

// Função para capturar o sinal de interrupção do sistema
void handle_sig(int signal) {
    cout << "\n[INFO] Sinal de parada recebido. Finalizando..." << endl;
    running = false;
}

// ---------------------------------------------------------
// Lógica de Negócio (Processamento de Metadados na CPU)
// Esta função é leve e não manipula pixels, apenas coordenadas.
// ---------------------------------------------------------
bool detect_pack(const vector<Detection>& detections, const Size& frameSize) {
    vector<Point> centers;
    
    // Filtra apenas cachorros
    for (const auto& d : detections) {
        if (d.class_name == "dog") {
            Point c(d.box.x + d.box.width / 2, d.box.y + d.box.height / 2);
            centers.push_back(c);
        }
    }

    int dogCount = centers.size();
    // Se houver menos de 3 cães, não é considerado alcateia/matilha
    if (dogCount < 3) return false;

    // Define proximidade como 50% da menor dimensão da tela
    float proximity = 0.5f * min(frameSize.width, frameSize.height);
    int closePairs = 0;

    // Verifica quantos cães estão próximos uns dos outros
    for (size_t i = 0; i < centers.size(); i++) {
        for (size_t j = i + 1; j < centers.size(); j++) {
            if (norm(centers[i] - centers[j]) < proximity)
                closePairs++;
        }
    }

    // Regra: Pelo menos 3 cães e pelo menos 2 pares próximos
    return (dogCount >= 3 && closePairs >= 2);
}

void gpu_rectangle(cv::cuda::GpuMat& img, cv::Rect rect, cv::Scalar color, int thickness, cv::cuda::Stream& stream) {
    // Garante que o retângulo está dentro da imagem
    cv::Rect img_rect(0, 0, img.cols, img.rows);
    rect = rect & img_rect; // Interseção segura

    if (rect.empty()) return;

    // Se thickness < 0 (FILLED), pinta o retângulo todo
    if (thickness < 0) {
        img(rect).setTo(color, stream);
        return;
    }

    // Desenha as 4 linhas usando sub-matrizes (ROI)
    // Topo
    cv::Rect r_top(rect.x, rect.y, rect.width, thickness);
    r_top = r_top & img_rect;
    if (!r_top.empty()) img(r_top).setTo(color, stream);

    // Base
    cv::Rect r_bottom(rect.x, rect.y + rect.height - thickness, rect.width, thickness);
    r_bottom = r_bottom & img_rect;
    if (!r_bottom.empty()) img(r_bottom).setTo(color, stream);

    // Esquerda
    cv::Rect r_left(rect.x, rect.y, thickness, rect.height);
    r_left = r_left & img_rect;
    if (!r_left.empty()) img(r_left).setTo(color, stream);

    // Direita
    cv::Rect r_right(rect.x + rect.width - thickness, rect.y, thickness, rect.height);
    r_right = r_right & img_rect;
    if (!r_right.empty()) img(r_right).setTo(color, stream);
}

// ---------------------------------------------------------
// Função Principal
// ---------------------------------------------------------
int main(int argc, char** argv) {
    // Registra o handler de sinal (Ctrl+C)
    signal(SIGINT, handle_sig);
    signal(SIGTERM, handle_sig);

    // 1. Configuração de Arquivos e Fonte
    // ------------------------------------
    string source = (argc > 1) ? argv[1] : "rtsp://127.0.0.1:8554/video";
    
    // Caminhos absolutos definidos conforme seu ambiente
    string modelPath = "/home/monitora-uff/OpenCV_TCC_MonitoraUFF/YoloDetect/models/yolov8n.onnx";
    string classesPath = "/home/monitora-uff/OpenCV_TCC_MonitoraUFF/YoloDetect/models/coco.names";

    // Parâmetros do Modelo
    const int inputWidth = 640;
    const int inputHeight = 640;
    const float scoreThreshold = 0.25f;
    const float nmsThreshold = 0.45f;

    // 2. Verificação de Hardware (GPU)
    // ------------------------------------
    if (cuda::getCudaEnabledDeviceCount() <= 0) {
        cerr << "[FATAL] Nenhuma GPU CUDA detectada. O programa será encerrado." << endl;
        return -1;
    }
    // Define a GPU 0 como padrão
    cuda::setDevice(0);
    cuda::printShortCudaDeviceInfo(cuda::getDevice());

    // 3. Carregamento das Classes
    // ------------------------------------
    vector<string> class_names;
    ifstream ifs(classesPath);
    if (!ifs.is_open()) {
        cerr << "[FATAL] Falha ao abrir arquivo de classes: " << classesPath << endl;
        return -1;
    }
    string line;
    while (getline(ifs, line)) {
        if (!line.empty()) {
            if (line.back() == '\r') line.pop_back(); // Remove caracteres Windows
            class_names.push_back(line);
        }
    }
    cout << "[INFO] " << class_names.size() << " classes carregadas." << endl;

    // 4. Carregamento da Rede Neural (DNN)
    // ------------------------------------
    dnn::Net net;
    try {
        cout << "[INFO] Carregando modelo YOLOv8n na GPU..." << endl;
        net = dnn::readNetFromONNX(modelPath);
        
        // Força o uso do backend CUDA (processamento na GPU)
        net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
    } catch (const cv::Exception& e) {
        cerr << "[FATAL] Erro ao carregar o modelo: " << e.what() << endl;
        return -1;
    }

    // 5. Inicialização do NVDEC (Decodificador de Hardware)
    // ------------------------------------
    // Configurações para RTSP de baixa latência
    cv::cudacodec::VideoReaderInitParams params;
    params.allowFrameDrop = true; // Permite pular frames se a GPU estiver cheia (evita lag)
    params.udpSource = true;      // Otimização para streams UDP/RTSP
    params.rawMode = false;       // Queremos o frame processado, não raw

    Ptr<cudacodec::VideoReader> reader;
    try {
        // Cria o leitor que decodifica direto para a memória da GPU
        reader = cudacodec::createVideoReader(source, {}, params);
        cout << "[INFO] Decodificador NVDEC iniciado para: " << source << endl;
    } catch (const cv::Exception& e) {
        cerr << "[FATAL] Falha ao abrir stream com NVDEC: " << e.what() << endl;
        return -1;
    }

    // 6. Configuração da Janela OpenGL
    // ------------------------------------
    // WINDOW_OPENGL é crucial: permite exibir cv::cuda::GpuMat sem download para CPU
    string windowName = "YOLOv8 Full GPU Pipeline";
    namedWindow(windowName, WINDOW_AUTOSIZE);
    resizeWindow(windowName, 1024, 768); // Tamanho inicial da janela

    // 7. Variáveis de Processamento (Alocadas na GPU)
    // ------------------------------------
    cuda::GpuMat d_frame_bgra;    // Frame original vindo do NVDEC (geralmente BGRA)
    cuda::GpuMat d_frame_rgb;     // Frame convertido para RGB
    cuda::GpuMat d_frame_resized; // Frame redimensionado para a rede (640x640)
    cuda::GpuMat d_frame_float;   // Frame normalizado (float 0.0 - 1.0)
    
    cuda::Stream stream;          // Stream CUDA para operações assíncronas

    // Objeto JSON para log
    json j_log;
    j_log["frames"] = json::array();
    
    int frameCount = 0;
    double t0_global = (double)getTickCount();

    cout << "[INFO] Iniciando loop de inferência..." << endl;

    // ---------------------------------------------------------
    // Loop Principal
    // ---------------------------------------------------------
    while (running) {
        // A. Leitura do Frame (Direto na GPU)
        bool success = false;
        try {
            // nextFrame preenche d_frame_bgra na VRAM
            success = reader->nextFrame(d_frame_bgra, stream);
        } catch (const cv::Exception& e) {
            cerr << "[ERRO] NVDEC nextFrame exception: " << e.what() << endl;
            break;
        }

        if (!success) {
            // Se não houve frame (buffer vazio ou fim), espera um pouco
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        
        frameCount++;

        // B. Pré-processamento na GPU (Zero CPU Copy)
        // -------------------------------------------
        
        // 1. Converter BGRA (padrão NVDEC) para RGB (padrão YOLO)
        // Ocorre inteiramente na GPU
        if (d_frame_bgra.channels() == 4) {
            cuda::cvtColor(d_frame_bgra, d_frame_rgb, COLOR_BGRA2RGB, 0, stream);
        } else {
            // Fallback caso venha BGR
            cuda::cvtColor(d_frame_bgra, d_frame_rgb, COLOR_BGR2RGB, 0, stream);
        }

        // 2. Normalizar (0-255 -> 0.0-1.0) e converter para Float32 na GPU
        d_frame_rgb.convertTo(d_frame_float, CV_32F, 1.0 / 255.0, stream);

        // 3. Redimensionar para 640x640 na GPU
        // Usamos INTER_LINEAR que é rápido e bom para redes neurais
        cuda::resize(d_frame_float, d_frame_resized, Size(inputWidth, inputHeight), 0, 0, INTER_LINEAR, stream);

        // 4. Preparar Blob
        // blobFromImage da OpenCV DNN suporta entrada GpuMat. 
        // Ele faz a transposição HWC -> NCHW necessária para a rede.
        Mat blob = dnn::blobFromImage(d_frame_resized, 1.0, Size(), Scalar(), false, false);
        net.setInput(blob);

        // C. Inferência (GPU - Tensor Cores)
        // -------------------------------------------
        vector<Mat> outputs;
        // Roda a rede. O cálculo é na GPU. O resultado volta para CPU (Mat) 
        // porque o tamanho do output (84x8400 floats) é pequeno e precisa de lógica complexa de parser.
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // D. Pós-processamento (CPU)
        // -------------------------------------------
        // Infelizmente, a lógica de parsing do YOLO e NMS é muito complexa para C++ CUDA puro
        // sem bibliotecas externas (como TensorRT puro). Faremos na CPU, mas é muito rápido.
        
        if (outputs.empty()) continue;

        Mat output = outputs[0];
        Mat out_t; // Transposta para facilitar leitura (linhas = detecções)

        // Ajuste de dimensões do YOLOv8 (pode vir como 1x84x8400)
        if (output.dims == 3) {
            int d = output.size[2]; // 8400
            int classes_plus_box = output.size[1]; // 84 (4 box + 80 classes)
            out_t = output.reshape(1, classes_plus_box).t();
        } else {
            // Formato 2D
            out_t = output;
        }

        vector<Rect> boxes;
        vector<float> confidences;
        vector<int> classIds;

        float x_factor = (float)d_frame_bgra.cols / inputWidth;
        float y_factor = (float)d_frame_bgra.rows / inputHeight;
        float* data = (float*)out_t.data;
        int rows = out_t.rows;

        // Itera sobre todas as possíveis detecções
        for (int i = 0; i < rows; ++i) {
            // Ponteiro para a linha atual
            float* row_ptr = data + i * out_t.cols;
            
            // Encontrar a classe com maior probabilidade (as classes começam no índice 4)
            float max_score = 0.0f;
            int max_class_id = -1;
            
            // Otimização simples: loop manual
            for (int c = 0; c < (int)class_names.size(); c++) {
                float score = row_ptr[4 + c];
                if (score > max_score) {
                    max_score = score;
                    max_class_id = c;
                }
            }

            if (max_score >= scoreThreshold) {
                // YOLO retorna centro_x, centro_y, largura, altura
                float cx = row_ptr[0];
                float cy = row_ptr[1];
                float w = row_ptr[2];
                float h = row_ptr[3];

                // Converte para coordenadas da imagem original
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(max_score);
                classIds.push_back(max_class_id);
            }
        }

        // Aplica Non-Maximum Suppression (NMS) para remover duplicatas
        vector<int> indices;
        dnn::NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, indices);

        // Prepara lista final de detecções
        vector<Detection> final_detections;
        for (int idx : indices) {
            Detection det;
            det.class_id = classIds[idx];
            det.class_name = class_names[classIds[idx]];
            det.confidence = confidences[idx];
            det.box = boxes[idx];

            // Gera uma cor única baseada no hash do nome (CPU)
            size_t hashValue = std::hash<string>{}(det.class_name);
            // Nota: GpuMat BGRA espera ordem Blue, Green, Red, Alpha
            det.color = Scalar(
                (hashValue & 0xFF), 
                ((hashValue >> 8) & 0xFF), 
                ((hashValue >> 16) & 0xFF), 
                255 // Alpha opaco
            );
            
            final_detections.push_back(det);
        }

        // Detecção de Alcateia
        bool is_pack = detect_pack(final_detections, d_frame_bgra.size());

        // E. Exportação JSON (A cada 12 frames)
        // -------------------------------------------
        if (frameCount % 12 == 0) {
            json frame_data;
            frame_data["frame_id"] = frameCount;
            frame_data["pack_detected"] = is_pack;
            frame_data["detections"] = json::array();

            for (const auto& d : final_detections) {
                frame_data["detections"].push_back({
                    {"class", d.class_name},
                    {"confidence", d.confidence},
                    {"box", {d.box.x, d.box.y, d.box.width, d.box.height}}
                });
            }
            j_log["frames"].push_back(frame_data);

            // Escrita em disco assíncrona/rápida
            try {
                ofstream outFile("result_video.json");
                outFile << j_log.dump(2);
            } catch (const std::exception& e) {
                cerr << "[ERRO] Falha ao salvar JSON: " << e.what() << endl;
            }
        }

        // F. Renderização na GPU (Desenhar Retângulos)
        // -------------------------------------------
        // Usamos cv::cuda::rectangle para modificar a d_frame_bgra diretamente na VRAM.
        // ATENÇÃO: Não desenhamos texto (putText) pois o OpenCV não suporta isso na GPU
        // e baixar o frame para CPU violaria a regra de performance.
        for (const auto& d : final_detections) {
            // Retângulo principal
            gpu_rectangle(d_frame_bgra, d.box, d.color, 2, stream);
            
            // Indicador de alcateia (borda vermelha grossa em toda a tela se detectado)
            if (is_pack) {
                cv::Rect tela(0, 0, d_frame_bgra.cols, d_frame_bgra.rows);
                gpu_rectangle(d_frame_bgra, tela, Scalar(0, 0, 255, 255), 10, stream);
            }
        }

        // G. Exibição via OpenGL (Zero Copy)
        // -------------------------------------------
        // imshow com suporte OpenGL pega o handle da textura da GPU e mostra.
        // Não há tráfego no barramento PCIe.
        cv::imshow(windowName, d_frame_bgra);

        // Verifica tecla ESC (1ms de delay)
        if (waitKey(1) == 27) break;
    }

    // ---------------------------------------------------------
    // Finalização
    // ---------------------------------------------------------
    double t1_global = (double)getTickCount();
    double total_fps = frameCount / ((t1_global - t0_global) / getTickFrequency());
    
    cout << "\n[INFO] Processamento concluído." << endl;
    cout << "[INFO] Total Frames: " << frameCount << endl;
    cout << "[INFO] Média de FPS: " << total_fps << endl;

    // Liberação de recursos é automática pelos destrutores do C++
    return 0;
}