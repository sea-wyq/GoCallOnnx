package main

import (
	"fmt"
	"infer/server"
)

func main() {
	s := server.Server{} //可以定义成全局变量
	res := s.InferCifar("./models/image.onnx", "./images/horse.png")
	fmt.Println(res)
}
