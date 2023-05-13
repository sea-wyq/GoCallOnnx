cmake_minimum_required(VERSION 3.10)
project(infer_servers)

set(CMAKE_CXX_STANDARD 11)

# 添加第三方库的头文件路径和库文件路径
include_directories(/home/wuyiqiang/onnx/include /usr/local/include/opencv4)
link_directories(/home/wuyiqiang/onnx/lib /usr/local/lib)

# 将 include 文件夹中的所有 CPP 文件添加到一个变量中
file(GLOB SRC_FILES "*.cpp")

# 生成静态库 lib1.a
add_library(onnxinfer STATIC ${SRC_FILES})

# 链接第三方库
target_link_libraries(onnxinfer onnxruntime opencv_core opencv_imgcodecs opencv_dnn)   