package main

//#cgo CXXFLAGS: -std=c++11
//#cgo CFLAGS: -I./include
//#cgo LDFLAGS: -L/home/wuyiqiang/GoCallOnnx/include/build -L/home/wuyiqiang/onnx/lib -L/usr/local/lib  -lonnxinfer -lonnxruntime -lopencv_core -lopencv_imgcodecs -lopencv_dnn -lstdc++
//
//#include "infer.h"
import "C"

func main() {

	modelPath := C.CString("./models/image.onnx")
	imagePath := C.CString("./images/horse.png")
	C.inferCifar(modelPath, imagePath)
}

/*
问题：error while loading shared libraries: libonnxruntime.so.1.7.0: cannot open shared object file: No such file or directory
解决：export LD_LIBRARY_PATH=/home/wuyiqiang/onnx/lib:$LD_LIBRARY_PATH
*/

