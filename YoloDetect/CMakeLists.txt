cmake_minimum_required(VERSION 3.0)
project(YoloGPU)

# 1. Configura o C++ (C++11 ou superior é necessário para OpenCV 4)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# 2. Encontra o OpenCV (que você acabou de instalar)
find_package(OpenCV REQUIRED)

# 3. Encontra o OpenGL (necessário para sua janela Zero Copy)
find_package(OpenGL REQUIRED)

# 4. Configurações do CUDA (caminhos manuais para garantir)
# Adicionamos o caminho padrão do CUDA para o linker achar as libs
link_directories(/usr/local/cuda/lib64)
include_directories(
    /usr/local/cuda/include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

# 5. Define o executável (nome do app e arquivo fonte)
add_executable(yolo_detect_rtsp yolo_detect_rtsp.cpp)

# 6. Linka todas as bibliotecas necessárias ao executável
target_link_libraries(yolo_detect_rtsp
    ${OpenCV_LIBS}      # Linka todos os módulos do OpenCV (cudacodec, dnn, etc.)
    OpenGL::GL          # Linka libGL (OpenGL)
    OpenGL::GLU         # Linka libGLU (Utilitários OpenGL)
    cudart              # CUDA Runtime
    cublas              # CUDA BLAS (usado pelo DNN)
    cudnn               # CUDA Deep Neural Network
)

# Mensagem de sucesso para conferência
message(STATUS "OpenCV encontrado em: ${OpenCV_DIR}")
message(STATUS "Bibliotecas OpenGL: ${OPENGL_LIBRARIES}")