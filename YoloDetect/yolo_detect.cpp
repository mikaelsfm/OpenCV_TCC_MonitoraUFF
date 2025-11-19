/*
 * YOLOv8 + OpenCV CUDA - INFERÊNCIA EM IMAGEM ÚNICA
 *
 * Diferença do Pipeline de Vídeo:
 * - Não usa NVDEC (VideoReader), pois é uma imagem estática.
 * - Usa imread (CPU) -> upload (GPU).
 * - Todo o processamento (Resize, Normalize, Inferência) ocorre na GPU.
 * - Desenho e Salvamento ocorrem na CPU (para melhor qualidade visual e I/O de disco).
 */

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/dnn.hpp>

#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <unistd.h> // Para getcwd

using json = nlohmann::json;
using namespace cv;
using namespace std;
namespace cuda = cv::cuda;

// Estrutura de Detecção
struct Detection {
    string class_name;
    float confidence;
    Rect box;
};

// Lógica de Matilha
bool detect_pack(const vector<Detection>& detections) {
    vector<Point> centers;
    for (const auto& d : detections) {
        if (d.class_name == "dog") {
            centers.push_back(Point(d.box.x + d.box.width / 2, d.box.y + d.box.height / 2));
        }
    }
    // Regra: Pelo menos 5 cães (ajustado conforme seu script antigo)
    if (centers.size() < 5) return false;

    // Verifica proximidade
    for (size_t i = 0; i < centers.size(); i++)
        for (size_t j = i + 1; j < centers.size(); j++)
            if (norm(centers[i] - centers[j]) < 150) return true;

    return false;
}

int main(int argc, char** argv) {
    // 1. Verifica Argumentos
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " <caminho_da_imagem>" << endl;
        return -1;
    }

    string imagePath = argv[1];
    string modelPath = "/home/monitora-uff/OpenCV_TCC_MonitoraUFF/YoloDetect/models/yolov8l.onnx";
    string classesPath = "/home/monitora-uff/OpenCV_TCC_MonitoraUFF/YoloDetect/models/coco.names";

    // Parâmetros
    const int inputSize = 640;
    const float scoreThreshold = 0.25f;
    const float nmsThreshold = 0.45f;

    // 2. Configura GPU
    if (cuda::getCudaEnabledDeviceCount() <= 0) {
        cerr << "[FATAL] GPU não encontrada." << endl;
        return -1;
    }
    cuda::setDevice(0);

    // 3. Carrega Classes
    vector<string> class_names;
    ifstream ifs(classesPath);
    if (!ifs.is_open()) {
        cerr << "[ERRO] Não foi possível abrir: " << classesPath << endl;
        return -1;
    }
    string line;
    while (getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) class_names.push_back(line);
    }

    // 4. Carrega Modelo (Backend CUDA)
    dnn::Net net;
    try {
        net = dnn::readNetFromONNX(modelPath);
        net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
        cout << "[INFO] Modelo carregado na GPU (CUDA)." << endl;
    } catch (const cv::Exception& e) {
        cerr << "[ERRO] Falha ao carregar modelo: " << e.what() << endl;
        return -1;
    }

    // 5. Carrega Imagem (IO Disk)
    Mat img_raw = imread(imagePath);
    if (img_raw.empty()) {
        cerr << "[ERRO] Imagem inválida ou não encontrada: " << imagePath << endl;
        return -1;
    }
    cout << "[INFO] Imagem carregada: " << img_raw.cols << "x" << img_raw.rows << endl;

    // ====================================================
    // INÍCIO DO PIPELINE GPU
    // ====================================================
    double t0 = (double)getTickCount();

    // Alocação na GPU
    cuda::GpuMat d_src, d_rgb, d_resized, d_float;
    
    // A. Upload CPU -> GPU
    d_src.upload(img_raw);

    // B. Pré-processamento na GPU (Rápido)
    // Converte BGR -> RGB
    cuda::cvtColor(d_src, d_rgb, COLOR_BGR2RGB);
    
    // Normaliza (0-255 -> 0.0-1.0) e Converte para Float32
    d_rgb.convertTo(d_float, CV_32F, 1.0 / 255.0);

    // Resize para 640x640
    cuda::resize(d_float, d_resized, Size(inputSize, inputSize), 0, 0, INTER_LINEAR);

    // C. Ponte GPU -> CPU para o Blob (Workaround Essencial)
    Mat h_resized_cpu;
    d_resized.download(h_resized_cpu); // Baixa apenas a miniatura processada

    // Cria Blob
    Mat blob = dnn::blobFromImage(h_resized_cpu, 1.0, Size(), Scalar(), false, false);
    net.setInput(blob);

    // D. Inferência na GPU
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // ====================================================
    // PÓS-PROCESSAMENTO (CPU)
    // ====================================================
    
    if (outputs.empty()) {
        cerr << "[ERRO] Saída vazia do modelo." << endl;
        return -1;
    }

    Mat out_t = (outputs[0].dims == 3) ? outputs[0].reshape(1, outputs[0].size[1]).t() : outputs[0];
    
    // Fatores de escala para mapear de volta para a imagem original
    float x_factor = (float)img_raw.cols / inputSize;
    float y_factor = (float)img_raw.rows / inputSize;
    float* data = (float*)out_t.data;

    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> classIds;

    for (int i = 0; i < out_t.rows; ++i) {
        float* row_ptr = data + i * out_t.cols;
        float max_score = 0; 
        int max_class = -1;

        for (int c = 0; c < (int)class_names.size(); c++) {
            if (row_ptr[4 + c] > max_score) {
                max_score = row_ptr[4 + c];
                max_class = c;
            }
        }

        if (max_score >= scoreThreshold) {
            float w = row_ptr[2]; float h = row_ptr[3];
            int left = int((row_ptr[0] - 0.5f * w) * x_factor);
            int top = int((row_ptr[1] - 0.5f * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.emplace_back(left, top, width, height);
            confidences.push_back(max_score);
            classIds.push_back(max_class);
        }
    }

    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, indices);

    vector<Detection> detections;
    for (int idx : indices) {
        Detection d;
        d.class_name = class_names[classIds[idx]];
        d.confidence = confidences[idx];
        d.box = boxes[idx];
        detections.push_back(d);
    }

    double t1 = (double)getTickCount();
    double time_ms = (t1 - t0) * 1000.0 / getTickFrequency();
    cout << "[INFO] Tempo de processamento: " << time_ms << " ms" << endl;

    // ====================================================
    // RENDERIZAÇÃO VISUAL (Estilo Antigo)
    // ====================================================
    
    // Cores vibrantes (Estilo novo/claro)
    vector<Scalar> colors;
    for (size_t i = 0; i < class_names.size(); i++) {
        colors.emplace_back(
            (uchar)(128 + (37 * i % 127)),
            (uchar)(128 + (17 * i % 127)),
            (uchar)(128 + (29 * i % 127))
        );
    }

    for (const auto& d : detections) {
        // Acha a cor certa para a classe
        int classIdx = -1;
        for(size_t k=0; k<class_names.size(); k++) {
            if(class_names[k] == d.class_name) { classIdx = k; break; }
        }
        if(classIdx == -1) classIdx = 0;
        Scalar color = colors[classIdx % colors.size()];

        // Caixa
        rectangle(img_raw, d.box, color, 2);

        // Texto com fundo
        string label = d.class_name + " " + format("%.2f", d.confidence);
        int baseLine = 0;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        int top = max(d.box.y, labelSize.height);
        
        // Fundo da etiqueta
        rectangle(img_raw, 
            Point(d.box.x, top - labelSize.height),
            Point(d.box.x + labelSize.width, top + baseLine),
            color, FILLED);
            
        // Texto (Preto para contraste)
        putText(img_raw, label, 
            Point(d.box.x, top),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }

    // Verifica Matilha
    bool is_pack = detect_pack(detections);
    if (is_pack) {
        cout << "\n>>> ALERTA: Matilha detectada na imagem! <<<\n" << endl;
        // Marca d'água visual na imagem
        putText(img_raw, "ALERTA: MATILHA", Point(20, 50), FONT_HERSHEY_DUPLEX, 1.5, Scalar(0, 0, 255), 2);
    }

    // ====================================================
    // SAÍDA (Arquivo e JSON)
    // ====================================================

    // Salva imagem anotada
    string outputImagePath = "result.jpg";
    imwrite(outputImagePath, img_raw);
    cout << "Imagem anotada salva em: " << outputImagePath << endl;

    // Gera JSON
    json j;
    if (detections.empty()) {
        j["detections"] = json::array();
    } else {
        for (auto& d : detections) {
            j["detections"].push_back({
                {"class", d.class_name},
                {"confidence", d.confidence},
                {"bbox", {d.box.x, d.box.y, d.box.x + d.box.width, d.box.y + d.box.height}}
            });
        }
    }
    j["is_pack"] = is_pack;

    // Dump JSON no terminal
    cout << j.dump(2) << endl;

    // Salva JSON no disco
    ofstream outFile("result.json");
    outFile << j.dump(2);
    outFile.close();
    cout << "JSON salvo em: result.json" << endl;

    return 0;
}