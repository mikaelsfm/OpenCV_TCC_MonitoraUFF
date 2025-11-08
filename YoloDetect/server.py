# yolo_gateway.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import subprocess
import os
import json

app = Flask(__name__)

YOLO_BIN = "/home/monitora-uff/OpenCV_TCC_MonitoraUFF/YoloDetect/build/yolo_detect"
UPLOAD_DIR = "/tmp/yolo_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/run_yolo", methods=["POST"])
def run_yolo():
    # 1. Verifica se o arquivo foi enviado
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    try:
        # 2. Executa o binário YOLO passando o caminho da imagem
        result = subprocess.run(
            [YOLO_BIN, file_path],
            text=True,
            timeout=60
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": "Falha ao executar YOLO",
                "stderr": stderr
            }), 500

        # 3. Tenta carregar o JSON gerado pelo binário (se existir)
        result_json_path = os.path.join(os.getcwd(), "result.json")
        detections = {}

        if os.path.exists(result_json_path):
            try:
                with open(result_json_path, "r") as f:
                    detections = json.load(f)
            except Exception as e:
                with open(result_json_path, "r") as f:
                    detections = {"raw": f.read(), "parse_error": str(e)}

        # 4. Retorna tudo consolidado
        return jsonify({
            "status": "ok",
            "file": file.filename,
            "detections": detections,
            "result_image": "result.jpg",
            "stdout": output
        }), 200

    except subprocess.TimeoutExpired:
        return jsonify({
            "status": "error",
            "message": "Timeout ao executar YOLO"
        }), 504

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        # 5. Limpa arquivo temporário
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    print("Servidor Flask rodando na porta 5005 (YOLO Gateway)...")
    app.run(host="0.0.0.0", port=5005)
