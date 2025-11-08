import cv2
import numpy as np
import time

# --- Parâmetros do Teste ---
rows, cols = 4000, 4000
# NOVO: Número de vezes que vamos repetir o cálculo na GPU
# Aumente este número se quiser que o teste dure ainda mais.
num_iterations = 200 

print(f"Iniciando teste de multiplicação de matrizes de {rows}x{cols}...")
print(f"O teste de GPU será repetido {num_iterations} vezes para criar uma carga sustentada.")

# --- Etapa 1: Preparar os dados na CPU (Host) ---
mat_a_cpu = np.random.rand(rows, cols).astype(np.float32)
mat_b_cpu = np.random.rand(rows, cols).astype(np.float32)

# =================================================================
# Teste 1: Execução na CPU (uma única vez para referência)
# =================================================================
# print("\n--- EXECUTANDO NA CPU ---")
# start_time_cpu = time.time()
# result_cpu = cv2.gemm(mat_a_cpu, mat_b_cpu, 1, None, 0)
# end_time_cpu = time.time()
# duration_cpu = end_time_cpu - start_time_cpu
# print(f"Cálculo na CPU concluído em: {duration_cpu:.4f} segundos.")

# =================================================================
# Teste 2: Execução na GPU (em um loop para estressar)
# =================================================================
print(f"\n--- EXECUTANDO NA GPU ({num_iterations} iterações) ---")
try:
    # Envia as matrizes para a VRAM APENAS UMA VEZ, antes do loop
    mat_a_gpu = cv2.cuda_GpuMat()
    mat_b_gpu = cv2.cuda_GpuMat()
    mat_a_gpu.upload(mat_a_cpu)
    mat_b_gpu.upload(mat_b_cpu)

    # Inicia o cronômetro antes de começar o loop
    start_time_gpu = time.time()

    # NOVO: Loop que executa o cálculo várias vezes
    for i in range(num_iterations):
        # Executa a multiplicação diretamente na GPU
        result_gpu = cv2.cuda.gemm(mat_a_gpu, mat_b_gpu, 1, None, 0)
        # Imprime o progresso para não parecer que travou
        if (i + 1) % 20 == 0:
            print(f"  Iteração {i + 1}/{num_iterations}...")

    # Para o cronômetro depois que o loop terminar
    end_time_gpu = time.time()
    
    # duration_gpu = end_time_gpu - start_time_gpu
    # print(f"Cálculo de {num_iterations} iterações na GPU concluído em: {duration_gpu:.4f} segundos.")

    # --- RESULTADO ---
    print("\n--- RESULTADO ---")
    print(f"✅ Sucesso! O OpenCV executou {num_iterations} cálculos na GPU de forma sustentada.")
    print("Se você viu um pico estável em 'GPU-Util' no nvidia-smi, sua instalação está funcionando perfeitamente.")

except cv2.error as e:
    print("\n--- ERRO ---")
    print("❌ Falha! O OpenCV não conseguiu executar a operação na GPU.")
    print(f"Detalhe do erro: {e}")