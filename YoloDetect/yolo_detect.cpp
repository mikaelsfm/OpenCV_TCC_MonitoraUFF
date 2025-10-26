// yolo_detect_fixed_v8.cpp
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
    string imagePath = "../assets/test.jpg";

    const int inputWidth = 640;
    const int inputHeight = 640;
    const float scoreThreshold = 0.01f;
    const float nmsThreshold = 0.01f;

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

    Mat image = imread(imagePath);
    cout << "Carregando imagem de: " << imagePath << endl;
    cout << "Dimensões da imagem: " << image.cols << "x" << image.rows << endl;
    if (image.empty()) {
        cerr << "Erro: não foi possível carregar a imagem." << endl;
        return -1;
    }

    Mat blob = dnn::blobFromImage(image, 1.0 / 255.0, Size(inputWidth, inputHeight), Scalar(), true, false);
    net.setInput(blob);

    {
        double minVal = 0, maxVal = 0;
        Mat flatBlob = blob.reshape(1, 1);
        minMaxLoc(flatBlob, &minVal, &maxVal);
        cout << "Blob range: " << minVal << " - " << maxVal << endl;
    }

    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    if (outputs.empty()) {
        cerr << "Erro: saída vazia do modelo." << endl;
        return -1;
    }

    Mat output = outputs[0];
    cout << "Dimensões da saída: ";
    for (int i = 0; i < output.dims; i++) cout << output.size[i] << " ";
    cout << endl;

    Mat out;
    if (output.dims == 3) {
        int d0 = output.size[0];
        int d1 = output.size[1];
        int d2 = output.size[2];
        if (d1 == 84 && d2 >= 1) {
            Mat tmp = output.reshape(1, d1);
            out = tmp.t();
        } else if (d1 > 1 && d2 == 84) {
            out = output.reshape(1, d1);
        } else {
            Mat tmp = output.reshape(1, output.size[1]);
            if (tmp.rows > 0 && tmp.cols > 0) {
                if (tmp.rows == 84) out = tmp.t();
                else out = tmp; // best-effort
            } else {
                cerr << "Formato de saída inesperado. dims: ";
                for (int i = 0; i < output.dims; ++i) cerr << output.size[i] << " ";
                cerr << endl;
                return -1;
            }
        }
    } else {
        cerr << "Formato de saída inesperado (dims != 3)." << endl;
        return -1;
    }

    int numDetections = out.rows;
    int numAttributes = out.cols;
    
    int numClasses = numAttributes - 4;

    cout << "Total de detecções (linhas): " << numDetections << endl;
    cout << "Atributos por detecção (colunas): " << numAttributes << " (classes = " << numClasses << ")" << endl;

    {
        int toPrint = min(10, out.cols);
        cout << "Primeiros valores da primeira linha: ";
        const float* p0 = out.ptr<float>(0);
        for (int i = 0; i < toPrint; ++i) cout << p0[i] << " ";
        cout << "..." << endl;
    }

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

        if (classScore <= scoreThreshold) {
            continue;
        }

        float finalConf = static_cast<float>(classScore);
        int clsId = classIdPoint.x;

        if (clsId < 0 || clsId >= (int)class_names.size()) {
            continue;
        }

        int x = static_cast<int>((cx - w / 2.0f) * inputWidth);
        int y = static_cast<int>((cy - h / 2.0f) * inputHeight);
        int width = static_cast<int>(w * inputWidth);
        int height = static_cast<int>(h * inputHeight);

        float xScale = static_cast<float>(image.cols) / inputWidth;
        float yScale = static_cast<float>(image.rows) / inputHeight;

        int rx = static_cast<int>(round(x * xScale));
        int ry = static_cast<int>(round(y * yScale));
        int rwidth = static_cast<int>(round(width * xScale));
        int rheight = static_cast<int>(round(height * yScale));

        rx = std::max(0, std::min(rx, image.cols - 1));
        ry = std::max(0, std::min(ry, image.rows - 1));
        if (rwidth <= 0 || rheight <= 0) continue;
        if (rx + rwidth > image.cols) rwidth = image.cols - rx;
        if (ry + rheight > image.rows) rheight = image.rows - ry;

        boxes.emplace_back(rx, ry, rwidth, rheight);
        confidences.push_back(finalConf);
        classIds.push_back(clsId);
    }


    cout << "Candidatos antes de NMS: " << boxes.size() << endl;

    vector<int> nmsIndices;
    if (!boxes.empty()) {
        vector<Rect> boxesCopy = boxes; // NMSBoxes espera vector<Rect>
        dnn::NMSBoxes(boxesCopy, confidences, scoreThreshold, nmsThreshold, nmsIndices);
    }

    cout << "Detecções após NMS: " << nmsIndices.size() << endl;

    vector<Detection> detections;
    for (int idx : nmsIndices) {
        Detection d;
        d.class_name = class_names[classIds[idx]];
        d.confidence = confidences[idx];
        d.box = boxes[idx];
        detections.push_back(d);
    }

    bool is_pack = detect_pack(detections);
    if (is_pack) cout << "\n>>> ALERTA: Matilha detectada na imagem! <<<\n" << endl;

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


    cout << j.dump(2) << endl;

    ofstream outFile("result.json");
    outFile << j.dump(2);
    outFile.close();
    cout << "\nJSON salvo em: result.json\n";

    return 0;
}