# GoCallOnnx
使用CGO来调用onnx模型的推理流程

## 前置环境
- openCV
- onnxruntime 1.9.0
- cuda11.4
- cudnn8.2.4

## 制作onnx模型推理的静态库onnxinfer
```shell
cd include
mkdir build & cd build
cmake ..
make
```
## 设置动态库路径
```shell
export LD_LIBRARY_PATH=/home/wuyiqiang/GoCallOnnx/include/build:$LD_LIBRARY_PATH
```

## 生成可执行程序
```shell
go build -o infer main.go
./infer
```
