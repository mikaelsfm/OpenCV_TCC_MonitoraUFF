#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace cv;
using namespace std;

struct Detection {
    string class_name;
    float confidence;
    Rect box;
};

// --- Função para detectar matilha ---
bool detect_pack(const vector<Detection>& detections, const Size& frameSize) {
    vector<Point> centers;
    for (auto& d : detections) {
        if (d.class_name == "dog") {
            Point c(d.box.x + d.box.width / 2, d.box.y + d.box.height / 2);
            centers.push_back(c);
        }
    }

    int dogCount = centers.size();
    if (dogCount < 3) return false;

    float proximity = 0.15f * min(frameSize.width, frameSize.height);
    int closePairs = 0;

    for (size_t i = 0; i < centers.size(); i++) {
        for (size_t j = i + 1; j < centers.size(); j++) {
            if (norm(centers[i] - centers[j]) < proximity)
                closePairs++;
        }
    }

    return (dogCount >= 3 && closePairs >= 2);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Uso: ./YoloDetect <caminho_do_video_ou_rtsp>" << endl;
        return -1;
    }

    string source = argv[1];
    cout << "[INFO] Fonte de vídeo: " << source << endl;

    // --- Caminhos do modelo e classes ---
    string modelPath = "/home/monitora-uff/OpenCV_TCC_MonitoraUFF/YoloDetect/models/yolov8l.onnx";
    string classesPath = "/home/monitora-uff/OpenCV_TCC_MonitoraUFF/YoloDetect/models/coco.names";

    const int inputWidth = 640;
    const int inputHeight = 640;
    const float scoreThreshold = 0.25f;
    const float nmsThreshold = 0.45f;

    // --- Carrega classes ---
    vector<string> class_names;
    ifstream ifs(classesPath);
    if (!ifs.is_open()) {
        cerr << "[ERRO] Falha ao abrir arquivo de classes: " << classesPath << endl;
        return -1;
    }

    string line;
    while (getline(ifs, line)) {
        if (!line.empty()) {
            if (line.back() == '\r') line.pop_back();
            class_names.push_back(line);
        }
    }
    cout << "[INFO] " << class_names.size() << " classes carregadas." << endl;

    // --- Carrega modelo YOLOv8L ---
    dnn::Net net;
    try {
        net = dnn::readNetFromONNX(modelPath);
        net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
        cout << "[INFO] Modelo YOLOv8L carregado na GPU (CUDA)." << endl;
    } catch (const cv::Exception& e) {
        cerr << "[ERRO] Falha ao carregar modelo: " << e.what() << endl;
        return -1;
    }

    // --- Abre vídeo / RTSP ---
    VideoCapture cap(source);
    if (!cap.isOpened()) {
        cerr << "[ERRO] Não foi possível abrir a fonte de vídeo: " << source << endl;
        return -1;
    }

    cout << "[INFO] Streaming iniciado. Resolução: "
         << cap.get(CAP_PROP_FRAME_WIDTH) << "x" << cap.get(CAP_PROP_FRAME_HEIGHT)
         << " | FPS: " << cap.get(CAP_PROP_FPS) << endl;

    json j;
    j["frames"] = json::array();
    int frameCount = 0;
    double t0 = (double)getTickCount();

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        frameCount++;

        Mat blob = dnn::blobFromImage(frame, 1.0 / 255.0, Size(inputWidth, inputHeight), Scalar(), true, false);
        net.setInput(blob);

        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());
        if (outputs.empty()) continue;

        Mat output = outputs[0];
        Mat out;
        if (output.dims == 3) {
            int d1 = output.size[1];
            int d2 = output.size[2];
            if (d1 == 84) out = output.reshape(1, d1).t();
            else if (d2 == 84) out = output.reshape(1, d2);
            else continue;
        } else continue;

        int numDetections = out.rows;
        int numAttributes = out.cols;
        int numClasses = numAttributes - 4;

        vector<Rect> boxes;
        vector<float> confidences;
        vector<int> classIds;

        for (int i = 0; i < numDetections; ++i) {
            const float* data = out.ptr<float>(i);
            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            Mat scores(1, numClasses, CV_32F, (void*)(data + 4));
            Point classIdPoint;
            double classScore;
            minMaxLoc(scores, 0, &classScore, 0, &classIdPoint);
            if (classScore <= scoreThreshold) continue;

            int clsId = classIdPoint.x;
            if (clsId < 0 || clsId >= (int)class_names.size()) continue;

            int x = static_cast<int>((cx - w / 2.0f) * frame.cols);
            int y = static_cast<int>((cy - h / 2.0f) * frame.rows);
            int width = static_cast<int>(w * frame.cols);
            int height = static_cast<int>(h * frame.rows);

            x = max(0, min(x, frame.cols - 1));
            y = max(0, min(y, frame.rows - 1));
            if (width <= 0 || height <= 0) continue;
            if (x + width > frame.cols) width = frame.cols - x;
            if (y + height > frame.rows) height = frame.rows - y;

            boxes.emplace_back(x, y, width, height);
            confidences.push_back((float)classScore);
            classIds.push_back(clsId);
        }

        vector<int> nmsIndices;
        if (!boxes.empty())
            dnn::NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, nmsIndices);

        vector<Detection> detections;
        for (int idx : nmsIndices) {
            Detection d;
            d.class_name = class_names[classIds[idx]];
            d.confidence = confidences[idx];
            d.box = boxes[idx];
            detections.push_back(d);
        }

        bool is_pack = detect_pack(detections, frame.size());
        if (is_pack)
            cout << "[ALERTA] Matilha detectada no frame " << frameCount << endl;

        json frame_data;
        frame_data["frame_id"] = frameCount;
        frame_data["pack_detected"] = is_pack;
        
        json j_detections = json::array();
        
        for (const auto& d : detections) {
            json det;
            det["class"] = d.class_name;
            det["confidence"] = d.confidence;
            det["box"] = {d.box.x, d.box.y, d.box.width, d.box.height};
            j_detections.push_back(det);
        }

        frame_data["detections"] = j_detections;    
        j["frames"].push_back(frame_data);

        for (auto& d : detections) {
            Scalar color = (d.class_name == "dog") ? Scalar(0, 255, 0) : Scalar(255, 255, 0);
            rectangle(frame, d.box, color, 2);
            string label = d.class_name + " " + format("%.2f", d.confidence);
            putText(frame, label, Point(d.box.x, d.box.y - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }

        imshow("YOLOv8L Real-time", frame);
        if (waitKey(1) == 27) break; // ESC para sair
    }

    ofstream outFile("result_video.json");
    outFile << j.dump(2);
    outFile.close();
    cout << "[INFO] Análise concluída. JSON salvo em 'result_video.json'." << endl;

    double t1 = (double)getTickCount();
    double fps = frameCount / ((t1 - t0) / getTickFrequency());
    cout << "[INFO] Processamento concluído. Média de " << fps << " FPS." << endl;

    return 0;
}
