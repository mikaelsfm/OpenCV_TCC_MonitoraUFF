from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route("/analisar", methods=["POST"])
def analisar():
    # Pega info do evento (pode ser um vídeo ou frame)
    evento = request.json.get("evento", "desconhecido")
    print(f"Recebi evento: {evento}")
    
    # Roda YOLO no vídeo/stream
    subprocess.Popen(["python", "meu_yolo.py", evento])
    
    return {"status": "YOLO iniciado"}

app.run(host="0.0.0.0", port=5000)
