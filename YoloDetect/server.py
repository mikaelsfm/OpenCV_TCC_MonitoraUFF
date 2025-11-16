# yolo_gateway.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import subprocess
import os
import json

app = Flask(__name__)

YOLO_BIN = "/home/monitora-uff/OpenCV_TCC_MonitoraUFF/YoloDetect/build/yolo_detect"
YOLO_BIN_RTSP = "/home/monitora-uff/OpenCV_TCC_MonitoraUFF/YoloDetect/build/yolo_detect_rtsp"
UPLOAD_DIR = "/tmp/yolo_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/run_yolo", methods=["POST"])
def run_yolo():
    rtsp_url = request.form.get("rtsp")
    print(f"Recebida requisição /run_yolo com RTSP >>>>>>>>>: {rtsp_url}")

    # Caso seja stream RTSP
    if rtsp_url:
        return run_yolo_rtsp(rtsp_url)

    # Caso seja arquivo (como já funciona hoje)
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo ou RTSP enviado"}), 400

    return run_yolo_file(request.files["file"])


def run_yolo_rtsp(rtsp_url):
    try:
        print(f"Executando YOLO com stream RTSP: {rtsp_url}")

        result = subprocess.run(
            [YOLO_BIN_RTSP, rtsp_url],
            text=True,
            timeout=120
        )

        return build_response(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_yolo_file(file):
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    try:
        result = subprocess.run(
            [YOLO_BIN, file_path],
            text=True,
            timeout=60
        )

        return build_response(result)

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def build_response(result):
    stdout = result.stdout.strip() if result.stdout else ""
    stderr = result.stderr.strip() if result.stderr else ""

    result_json_path = os.path.join(os.getcwd(), "result.json")
    detections = {}
    if os.path.exists(result_json_path):
        try:
            with open(result_json_path, "r") as f:
                detections = json.load(f)
        except:
            with open(result_json_path, "r") as f:
                detections = {"raw": f.read()}

    return jsonify({
        "status": "ok" if result.returncode == 0 else "error",
        "detections": detections,
        "stdout": stdout,
        "stderr": stderr
    }), 200 if result.returncode == 0 else 500


if __name__ == "__main__":
    print("Servidor Flask rodando na porta 5005 (YOLO Gateway)...")
    app.run(host="0.0.0.0", port=5005)
