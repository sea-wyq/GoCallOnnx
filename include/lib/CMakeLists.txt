cmake_minimum_required(VERSION 3.10)
project(lib)

add_subdirectory(infer_servers)

# 将生成的静态库文件移动到 build 文件夹中
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})