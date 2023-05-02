package main

import "infer/server"

func main() {
	s := server.Server{} //可以定义成全局变量
	s.InferCifar("./models/image.onnx", "./images/horse.png")
}
