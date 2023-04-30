package infer

/*
#cgo CXXFLAGS: -std=c++11
#cgo LDFLAGS:  -L/home/wuyiqiang/onnx/lib -L/usr/local/lib  -lonnxruntime -lopencv_core -lstdc++
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;

void infer(const char *model_path, const char *image_path){
    vector<const char *> input_names;
    vector<const char *> output_names;

    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "example");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path, session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = session.GetInputName(0, allocator);
    const auto output_name = session.GetOutputName(0, allocator);
    input_names.push_back(input_name);
    output_names.push_back(output_name);

    Ort::TypeInfo input_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo();
    vector<int64_t> input_shape = input_tensor_info.GetShape();
    vector<float> input_values(input_tensor_info.GetElementCount()); // 获取Tensor的元素数量：
    cout << input_tensor_info.GetElementCount() << endl;

    // 设置模型的输入tensor
    for (size_t i = 0; i < input_values.size(); i++)
    {
        input_values[i] = 1.0;
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    vector<Ort::Value> input_tensors;
    input_tensors.push_back(
        Ort::Value(Ort::Value::CreateTensor<float>(memory_info,input_values.data(),
		input_values.size(),input_shape.data(),input_shape.size())));

    Ort::TypeInfo output_info = session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo();
    vector<int64_t> output_shape = output_tensor_info.GetShape();
    vector<float> output_values(input_tensor_info.GetElementCount()); // 获取Tensor的元素数量：

    vector<Ort::Value> output_tensors;
    output_tensors.push_back(
        Ort::Value::CreateTensor<float>(memory_info,output_values.data(),output_values.size(),
                                        output_shape.data(),output_shape.size()));

    session.Run(Ort::RunOptions{nullptr},input_names.data(),input_tensors.data(),
                input_names.size(),output_names.data(),output_tensors.data(),output_names.size());

    float *output_data = output_tensors[0].GetTensorMutableData<float>();
    int predicted_digit = distance(output_data, max_element(output_data, output_data + 10));

    cout << "Predicted digit: " << predicted_digit << endl;
}
*/
import "C"

func main() {
	C.infer("image.onnx", "car.png")
}
