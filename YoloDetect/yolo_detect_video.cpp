// yolo_detect_live_webcam.cpp
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

struct Detection {
    string class_name;
    float confidence;
    Rect box;
};

bool detect_pack(const vector<Detection>& detections) {
    vector<Point> centers;
    for (auto& d : detections) {
        if (d.class_name == "dog") {
            Point c(d.box.x + d.box.width / 2, d.box.y + d.box.height / 2);
            centers.push_back(c);
        }
    }
    if (centers.size() < 5) return false;
    
    for (size_t i = 0; i < centers.size(); i++)
        for (size_t j = i + 1; j < centers.size(); j++)
            if (norm(centers[i] - centers[j]) < 150) return true;
            
    return false;
}

int main() {
    string modelPath = "../models/yolov8l.onnx"; 
    string classesPath = "../models/coco.names";
    
    int videoPath = 0;
    
    const int inputWidth = 640;
    const int inputHeight = 640;
    const float scoreThreshold = 0.25f;
    const float nmsThreshold = 0.45f;

    vector<string> class_names;
    {
        ifstream ifs(classesPath);
        if (!ifs.is_open()) {
            cerr << "Erro: não foi possível abrir classes file: " << classesPath << endl;
            return -1;
        }
        string line;
        while (getline(ifs, line)) {
            if (!line.empty()) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty()) class_names.push_back(line);
            }
        }
    }
    if (class_names.empty()) {
        cerr << "Aviso: lista de classes vazia." << endl;
    }
    cout << "Total de classes carregadas: " << class_names.size() << endl;

    dnn::Net net;
    try {
        net = dnn::readNetFromONNX(modelPath);
    } catch (const cv::Exception& e) {
        cerr << "Erro ao carregar ONNX: " << e.what() << endl;
        return -1;
    }

    try {
        net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
        cout << "CUDA ativo" << endl;
    } catch (...) {
        net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(dnn::DNN_TARGET_CPU);
        cout << "CUDA indisponível, usando CPU" << endl;
    }

    cv::VideoCapture cap;
    cap.open(videoPath); 

    if (!cap.isOpened()) {
        cerr << "Erro: não foi possível abrir a fonte de vídeo (ID " << videoPath << ")." << endl;
        return -1;
    }
    cout << "Webcam (ID " << videoPath << ") aberta com sucesso." << endl;

    cv::namedWindow("YOLO Monitoramento ao Vivo", cv::WINDOW_AUTOSIZE);
    Mat frame;

    while (true) {
        if (!cap.read(frame)) {
            cerr << "Erro: não foi possível ler o frame da webcam. Desconectada?" << endl;
            break;
        }

        if (frame.empty()) {
            cerr << "Frame vazio recebido." << endl;
            continue;
        }

        float xScale = (float)frame.cols / inputWidth;
        float yScale = (float)frame.rows / inputHeight;

        Mat blob = dnn::blobFromImage(frame, 1.0 / 255.0, Size(inputWidth, inputHeight), Scalar(), true, false);
        net.setInput(blob);

        vector<Mat> outputs;
        try {
            net.forward(outputs, net.getUnconnectedOutLayersNames());
        } catch (const cv::Exception& e) {
            cerr << "Erro durante a inferência (net.forward): " << e.what() << endl;
            cerr << "Isso pode acontecer se o modelo (ex: yolo11l) for incompatível." << endl;
            cerr << "Tente usar o yolov8l.onnx." << endl;
            break;
        }
        
        Mat output = outputs[0];
        Mat out;
        if (output.dims == 3) {
            int d1 = output.size[1];
            int d2 = output.size[2];
            if (d1 == 84 && d2 >= 1) {
                Mat tmp = output.reshape(1, d1); 
                out = tmp.t();                   
            } else if (d1 > 1 && d2 == 84) {
                out = output.reshape(1, d1);
            } else {
                continue;
            }
        } else {
            continue;
        }

        int numDetections = out.rows;
        int numClasses = out.cols - 4;

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

            float finalConf = static_cast<float>(classScore);
            int clsId = classIdPoint.x;
            if (clsId < 0 || clsId >= (int)class_names.size()) continue;

            int x = static_cast<int>((cx - w / 2.0f) * inputWidth);
            int y = static_cast<int>((cy - h / 2.0f) * inputHeight);
            int width = static_cast<int>(w * inputWidth);
            int height = static_cast<int>(h * inputHeight);

            int rx = static_cast<int>(round(x * xScale));
            int ry = static_cast<int>(round(y * yScale));
            int rwidth = static_cast<int>(round(width * xScale));
            int rheight = static_cast<int>(round(height * yScale));

            rx = std::max(0, std::min(rx, frame.cols - 1));
            ry = std::max(0, std::min(ry, frame.rows - 1));
            if (rwidth <= 0 || rheight <= 0) continue;
            if (rx + rwidth > frame.cols) rwidth = frame.cols - rx;
            if (ry + rheight > frame.rows) rheight = frame.rows - ry;

            boxes.emplace_back(rx, ry, rwidth, rheight);
            confidences.push_back(finalConf);
            classIds.push_back(clsId);
        }

        vector<int> nmsIndices;
        if (!boxes.empty()) {
            dnn::NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, nmsIndices);
        }

        vector<Detection> detections;
        for (int idx : nmsIndices) {
            Detection d;
            d.class_name = class_names[classIds[idx]];
            d.confidence = confidences[idx];
            d.box = boxes[idx];
            detections.push_back(d);

            rectangle(frame, d.box, Scalar(0, 255, 0), 2);
            
            string label = format("%s: %.2f", d.class_name.c_str(), d.confidence);
            int baseLine;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            
            rectangle(frame, Point(d.box.x, d.box.y - labelSize.height - baseLine),
                      Point(d.box.x + labelSize.width, d.box.y), Scalar(0, 255, 0), FILLED);
            
            putText(frame, label, d.box.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }

        bool is_pack = detect_pack(detections);
        if (is_pack) {
            putText(frame, "ALERTA: Matilha detectada!", Point(50, 50), 
                    FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255), 3);
        }

        cv::imshow("YOLO Monitoramento ao Vivo", frame);

        if (cv::waitKey(1) == 'q') {
            cout << "Saindo..." << endl;
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}