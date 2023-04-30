# GoCallOnnx
使用CGO来调用onnx模型的推理流程

## 

g++ -c -o infer.o infer.cpp -I/home/wuyiqiang/onnx/include -L/home/wuyiqiang/onnx/lib -lonnxruntime `pkg-config --cflags --libs opencv4`
ar rcs libonnxinfer.a infer.o

## 执行指令
go build -o infer main.go
./main
