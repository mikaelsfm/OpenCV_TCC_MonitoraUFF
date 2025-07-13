import cv2
import numpy as np
import mediapipe as mp

# --- 1. Inicialização do MediaPipe ---
# Inicializa as soluções do MediaPipe que vamos usar
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- 2. Carregamento da Imagem ---
# O caminho para a sua imagem. Verifique se está correto.
caminho_imagem = '../Fotos/imagem5.jpg'
imagem = cv2.imread(caminho_imagem)

# --- 3. Processamento com MediaPipe ---
# Verifica se a imagem foi carregada corretamente
if imagem is None:
    print(f"Erro: Não foi possível carregar a imagem em '{caminho_imagem}'")
else:
    # O 'with' garante que os recursos do MediaPipe sejam liberados no final
    # Parâmetros:
    #   static_image_mode=True: Otimizado para imagens estáticas (não vídeo).
    #   max_num_faces=5: Detecta até 5 rostos na imagem.
    #   refine_landmarks=True: Adiciona pontos extras para os olhos e lábios, aumentando a precisão.
    #   min_detection_confidence=0.5: Confiança mínima para a detecção de um rosto.
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        # O MediaPipe trabalha com imagens RGB, mas o OpenCV carrega em BGR.
        # Precisamos converter o espaço de cores.
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # Processa a imagem para encontrar os marcos faciais
        results = face_mesh.process(imagem_rgb)

        # --- 4. Desenho dos Resultados ---
        # Verifica se algum rosto foi detectado
        if results.multi_face_landmarks:
            print(f"Encontrado(s) {len(results.multi_face_landmarks)} rosto(s).")
            # Itera sobre cada rosto encontrado
            for face_landmarks in results.multi_face_landmarks:

                # # Desenha a malha completa do rosto (ótimo para visualização)
                # mp_drawing.draw_landmarks(
                #     image=imagem,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_tesselation_style())

                # Desenha os contornos dos olhos, sobrancelhas e lábios com mais destaque
                mp_drawing.draw_landmarks(
                    image=imagem,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())

                # Desenha as íris dos olhos (detecção ocular de alta precisão)
                # Isso só funciona se 'refine_landmarks=True'
                mp_drawing.draw_landmarks(
                    image=imagem,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        else:
            print("Nenhum rosto foi detectado na imagem.")

        # --- 5. Exibição da Imagem ---
        cv2.imshow("Detecta Rosto e Olhos com MediaPipe", imagem)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
