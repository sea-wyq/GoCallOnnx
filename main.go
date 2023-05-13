package main

import (
	"fmt"
	"infer/server"
)

var s = server.Server{} //可以定义成全局变量

func main() {

	res := s.InferCifar("./models/image.onnx", "./images/horse.png")
	fmt.Println("inferCifar:",res)

	res = s.InferDebias("./models/debias.onnx", "./inferData/ref.txt")
	fmt.Println("inferDebias:"res)

}
