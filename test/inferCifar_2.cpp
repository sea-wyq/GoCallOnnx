#include <infer.h>

int main(){
    inferCifar("image.onnx", "car.png");
}
// gcc -o main main.cpp -I/home/wuyiqiang/testlib -L/home/wuyiqiang/testlib/build -lonnxinfer -I/home/wuyiqiang/onnx/include -L/home/wuyiqiang/onnx/lib -lonnxruntime `pkg-config --cflags --libs opencv4`
// export LD_LIBRARY_PATH=/home/wuyiqiang/onnx/lib:$LD_LIBRARY_PATH 生成动态库的时候需要添加路径，不然执行会出错，找不到onnxinfer.so