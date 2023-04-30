cmake_minimum_required(VERSION 3.10)
project(infer_servers)

# 将 infer_servers 文件夹中的所有 CPP 文件添加到一个变量中
file(GLOB SRC_FILES "*.cpp")

# 生成静态库 lib1.a
add_library(onnxinfer STATIC ${SRC_FILES})