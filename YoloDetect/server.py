from flask import Flask, request, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route("/run_yolo", methods=["POST"])
def run_yolo():
    data = request.get_json()
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "image_path ausente"}), 400

    try:
        # Executa o yolo_detect.py com o caminho da imagem
        result = subprocess.run(
            ["python3", "yolo_detect.py", image_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": result.stderr.strip()
            }), 500

        # Tenta interpretar o output como JSON (se seu script imprimir JSON)
        try:
            detections = json.loads(result.stdout)
        except json.JSONDecodeError:
            detections = {"raw_output": result.stdout.strip()}

        return jsonify({
            "status": "ok",
            "image": image_path,
            "detections": detections
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)