/*
 * YOLOv8l + OpenCV CUDA + NVDEC (GTX 1050) - MODO HEADLESS FINAL
 */

#include <opencv2/opencv.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/dnn.hpp>

// Biblioteca JSON
#include <nlohmann/json.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <string>
#include <cmath>


using json = nlohmann::json;
using namespace cv;
using namespace std;
namespace cuda = cv::cuda;

// --- Variáveis Globais ---
std::atomic<bool> running{true};

void handle_sig(int) {
    cout << "\n[INFO] Sinal de parada recebido. Finalizando..." << endl;
    running = false;
}

struct Detection {
    int class_id;
    string class_name;
    float confidence;
    Rect box;
};

// --- Lógica de Matilha (Processamento leve de metadados na CPU) ---
bool detect_pack(const vector<Detection>& detections, const Size& frameSize) {
    vector<Point> centers;
    for (const auto& d : detections) {
        if (d.class_name == "dog") {
            centers.push_back(Point(d.box.x + d.box.width / 2, d.box.y + d.box.height / 2));
        }
    }
    if (centers.size() < 3) return false;

    float proximity = 0.5f * min(frameSize.width, frameSize.height);
    int closePairs = 0;
    for (size_t i = 0; i < centers.size(); ++i) {
        for (size_t j = i + 1; j < centers.size(); ++j) {
            if (norm(centers[i] - centers[j]) < proximity) ++closePairs;
        }
    }
    return (centers.size() >= 3 && closePairs >= 2);
}

// --- Main ---
int main(int argc, char** argv) {
    signal(SIGINT, handle_sig);
    signal(SIGTERM, handle_sig);

    string source = (argc > 1) ? argv[1] : "rtsp://127.0.0.1:8554/video";
    
    string modelPath = "/home/monitora-uff/OpenCV_TCC_MonitoraUFF/YoloDetect/models/yolov8l.onnx";
    string classesPath = "/home/monitora-uff/OpenCV_TCC_MonitoraUFF/YoloDetect/models/coco.names";

    // Config GPU
    if (cuda::getCudaEnabledDeviceCount() <= 0) {
        cerr << "[FATAL] GPU não encontrada." << endl;
        return -1;
    }
    cuda::setDevice(0);
    
    // Carrega Classes
    vector<string> class_names;
    ifstream ifs(classesPath);
    string line;
    while (getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) class_names.push_back(line);
    }

    // Carrega Modelo (Backend CUDA)
    dnn::Net net = dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(dnn::DNN_TARGET_CUDA);

    // Configura NVDEC
    cudacodec::VideoReaderInitParams params;
    params.allowFrameDrop = false; 
    params.udpSource = false;      
    params.rawMode = false;        

    Ptr<cudacodec::VideoReader> reader;
    try {
        reader = cudacodec::createVideoReader(source, {}, params);
        cout << "[INFO] Decodificador NVDEC iniciado." << endl;
    } catch (const cv::Exception& e) {
        cerr << "[FATAL] Erro ao iniciar NVDEC: " << e.what() << endl;
        return -1;
    }

    // Alocação de Memória (VRAM)
    cuda::GpuMat d_frame_bgra;    // 4 canais (Decoder)
    cuda::GpuMat d_frame_rgb;     // 3 canais (YOLO)
    cuda::GpuMat d_frame_float;   // Normalizado
    cuda::GpuMat d_frame_resized; // 640x640

    // Buffer de CPU (Apenas para a entrada da rede - Pequeno)
    Mat h_frame_resized; 

    cuda::Stream stream;

    json j_log;
    j_log["frames"] = json::array();

    int frameCount = 0;
    int failCount = 0;
    const int inputSize = 640;

    cout << "[INFO] Iniciando inferência HEADLESS (Sem Display)..." << endl;
    double t0_global = (double)getTickCount();

    while (running) {
        bool success = false;
        try {
            // 1. Decodificação (100% GPU)
            success = reader->nextFrame(d_frame_bgra, stream);
        } catch (const cv::Exception& e) {
            break;
        }

        if (!success) {
            if (++failCount > 30) break; 
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        failCount = 0;

        if (d_frame_bgra.empty()) continue;
        frameCount++;

        // 2. Pré-processamento (100% GPU)
        // BGRA -> RGB
        if (d_frame_bgra.channels() == 4)
            cuda::cvtColor(d_frame_bgra, d_frame_rgb, COLOR_BGRA2RGB, 0, stream);
        else
            cuda::cvtColor(d_frame_bgra, d_frame_rgb, COLOR_BGR2RGB, 0, stream);

        // Normaliza e Resize
        d_frame_rgb.convertTo(d_frame_float, CV_32F, 1.0 / 255.0, stream);
        cuda::resize(d_frame_float, d_frame_resized, Size(inputSize, inputSize), 0, 0, INTER_LINEAR, stream);

        // 3. Ponte para Entrada (Download Mínimo)
        // Baixa APENAS o 640x640 para criar o Blob.
        d_frame_resized.download(h_frame_resized, stream);
        stream.waitForCompletion(); 

        // Passamos 'h_frame_resized' (CPU) em vez de 'd_frame_resized' (GPU)
        Mat blob = dnn::blobFromImage(h_frame_resized, 1.0, Size(), Scalar(), false, false);
        net.setInput(blob); // Upload automático para a GPU acontece aqui

        // 4. Inferência (100% GPU)
        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // 5. Parsing de Resultados (CPU - Rápido)
        if (!outputs.empty()) {
            Mat out_t = (outputs[0].dims == 3) ? outputs[0].reshape(1, outputs[0].size[1]).t() : outputs[0];
            
            // Fatores de escala baseados no tamanho original (que está na GPU, mas sabemos o tamanho)
            float x_factor = (float)d_frame_bgra.cols / inputSize;
            float y_factor = (float)d_frame_bgra.rows / inputSize;
            float* data = (float*)out_t.data;

            vector<Rect> boxes;
            vector<float> confidences;
            vector<int> classIds;

            for (int i = 0; i < out_t.rows; ++i) {
                float* row_ptr = data + i * out_t.cols;
                float max_score = 0; int max_class = -1;

                for (int c = 0; c < (int)class_names.size(); c++) {
                    if (row_ptr[4 + c] > max_score) {
                        max_score = row_ptr[4 + c];
                        max_class = c;
                    }
                }

                if (max_score >= 0.25f) {
                    float w = row_ptr[2]; float h = row_ptr[3];
                    int left = int((row_ptr[0] - 0.5f * w) * x_factor);
                    int top = int((row_ptr[1] - 0.5f * h) * y_factor);
                    boxes.emplace_back(left, top, int(w * x_factor), int(h * y_factor));
                    confidences.push_back(max_score);
                    classIds.push_back(max_class);
                }
            }

            vector<int> indices;
            dnn::NMSBoxes(boxes, confidences, 0.25f, 0.45f, indices);

            vector<Detection> final_detections;
            for (int idx : indices) {
                Detection d;
                d.class_id = classIds[idx];
                d.class_name = class_names[d.class_id];
                d.confidence = confidences[idx];
                d.box = boxes[idx];
                final_detections.push_back(d);
            }

            bool is_pack = detect_pack(final_detections, Size(d_frame_bgra.cols, d_frame_bgra.rows));

            // 6. Exportação JSON (Sem desenho, sem display)
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
                
                try {
                    ofstream outFile("result_video.json");
                    outFile << j_log.dump(2);
                } catch (...) {}
                
                // Feedback mínimo no terminal para saber que está vivo
                if (frameCount % 120 == 0) {
                    cout << "[STATUS] Frame: " << frameCount << " | Matilha: " << (is_pack ? "SIM" : "NÃO") << endl;
                }
            }
        }
    }

    double t1_global = (double)getTickCount();
    double total_fps = frameCount / ((t1_global - t0_global) / getTickFrequency());

    cout << "\n[INFO] Finalizado." << endl;
    cout << "[INFO] Frames Totais: " << frameCount << endl;
    cout << "[INFO] FPS Médio: " << total_fps << endl;

    return 0;
}