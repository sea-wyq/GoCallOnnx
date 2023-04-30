# 动态库编译
cmake_minimum_required(VERSION 3.15)
 
project(infer) 

set(CMAKE_CXX_STANDARD 14)
 
# 添加第三方库的头文件路径
include_directories(/home/wuyiqiang/onnx/include /usr/local/include/opencv4)
 
# 添加第三方库的库文件路径
link_directories(/home/wuyiqiang/onnx/lib /usr/local/lib)
 
# 编译选项
add_compile_options(-Wall -std=c++11)
 
# 源文件
add_library(onnxinfer SHARED infer.cpp)