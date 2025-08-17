import time
import subprocess
from pyzm.api import ZMApi

# CONFIGURAÇÕES
ZM_HOST = "http://192.168.15.14:8080"  # Seu notebook Linux com ZoneMinder
MONITOR_ID = 2  # Qual câmera você quer monitorar
YOLO_SCRIPT = "yolo_script.py"  # Script YOLO que você já tem

# Inicializa API
zm = ZMApi(options={'apiurl': f'{ZM_HOST}/api'})

# Guarda último evento visto
last_event_id = 0

print("✅ Monitor de eventos iniciado. Aguardando detecções...")

while True:
    try:
        # Busca último evento desse monitor
        events = zm.events.list(
            monitor_id=MONITOR_ID,
            limit=1,
            sort='Id',
            direction='desc'
        )

        if events and events['events']:
            latest = events['events'][0]['Event']
            event_id = int(latest['Id'])
            cause = latest['Cause']
            start_time = latest['StartTime']

            if event_id > last_event_id:
                print(f"\n🚨 Novo evento detectado!")
                print(f"  ID: {event_id}")
                print(f"  Causa: {cause}")
                print(f"  Início: {start_time}")

                # Roda YOLO SOMENTE quando tem evento novo
                subprocess.Popen(["python", YOLO_SCRIPT, "--event", str(event_id)])

                last_event_id = event_id

        time.sleep(5)  # consulta a cada 5 segundos

    except Exception as e:
        print(f"⚠️ Erro ao consultar API: {e}")
        time.sleep(10)
