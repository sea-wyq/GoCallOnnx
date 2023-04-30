# GoCallOnnx
使用CGO来调用onnx模型的推理流程

## 前置环境
- openCV
- onnxruntime

## 制作onnx模型推理的静态库onnxinfer
```shell
cd include
mkdir build & cd build
cmake ..
make
```

## 生成可执行程序
```shell
go build -o infer main.go
./infer
```
