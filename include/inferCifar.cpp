#include <iostream>
#include <vector>
#include <algorithm>
#include "infer.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>

extern "C"
{
    int inferCifar(const char *model_path, const char *image_path)
    {
        
        std::vector<const char *> input_names;
        std::vector<const char *> output_names;

        Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "example"); // ORT_LOGGING_LEVEL_VERBOSE 获取更详细的日志信息
        Ort::SessionOptions session_options;

        OrtCUDAProviderOptions options;
        options.device_id = 0;
        options.arena_extend_strategy = 0;
        options.gpu_mem_limit = (size_t)1 * 1024 * 1024 * 1024; // 1G内存
        options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
        options.do_copy_in_default_stream = 1;
        session_options.AppendExecutionProvider_CUDA(options);

        Ort::Session session{env, model_path, Ort::SessionOptions{nullptr}}; // CPU
        // Ort::Session session{env, model_path, session_options}; //GPU

        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();
        size_t num_output_nodes = session.GetOutputCount();

        // 获得模型输入和输出的名称
        const auto input_name = session.GetInputName(0, allocator);
        const auto output_name = session.GetOutputName(0, allocator);
        input_names.push_back(input_name);
        output_names.push_back(output_name);


        // 获取输入张量的元数据
        Ort::TypeInfo input_info = session.GetInputTypeInfo(0);         // 用于获取 ONNX 模型的第一个输入的类型信息。
        auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo(); // 用于获取ONNX模型中张量的类型和形状信息
        std::vector<int64_t> input_shape = input_tensor_info.GetShape(); // 获取数据形状
        std::vector<float> input_values(input_tensor_info.GetElementCount()); // 获取Tensor的元素数量：

        // 设置模型的输入tensor
        cv::Mat imageBGR = cv::imread(image_path, cv::IMREAD_COLOR);
        cv::Mat scaledImage, preprocessedImage;
        imageBGR.convertTo(scaledImage, CV_32F, 2.0f / 255.0f, -1.0f); // Scale image pixels from [0 255] to [-1, 1]
        cv::dnn::blobFromImage(scaledImage, preprocessedImage); // Convert HWC to CHW
        input_values.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>()); // Assign the input image to the input tensor

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(
            Ort::Value(Ort::Value::CreateTensor<float>(memory_info,
                                                        input_values.data(),input_values.size(),
                                                        input_shape.data(),input_shape.size())));


        // 获取输出张量的元数据
        Ort::TypeInfo output_info = session.GetOutputTypeInfo(0);         // 用于获取 ONNX 模型的第一个输入的类型信息。
        auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo(); // 用于获取ONNX模型中张量的类型和形状信息
        std::vector<int64_t> output_shape = output_tensor_info.GetShape(); // 获取数据形状
        std::vector<float> output_values(input_tensor_info.GetElementCount()); // 获取Tensor的元素数量：

        std::vector<Ort::Value> output_tensors;
        output_tensors.push_back(
            Ort::Value::CreateTensor<float>(memory_info,
                                            output_values.data(),output_values.size(),
                                            output_shape.data(),output_shape.size()));
        // 创建计时器
        auto start_time = std::chrono::high_resolution_clock::now();
        session.Run(Ort::RunOptions{nullptr},
                    input_names.data(),input_tensors.data(),input_names.size(),
                    output_names.data(),output_tensors.data(),output_names.size());

        float *output_data = output_tensors[0].GetTensorMutableData<float>();
        int predicted_digit = std::distance(output_data, std::max_element(output_data, output_data + 10));
        // 结束计时器
        auto end_time = std::chrono::high_resolution_clock::now();
        // 计算推理时间
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Predicted digit: " << predicted_digit << std::endl;
        std::cout << "推理时间：" << duration.count() / 1000000.0 << " 秒" << std::endl;

        return predicted_digit;

        }
}

// g++ -std=c++11 -I/path/to/onnxruntime/include -L/path/to/onnxruntime/lib -lonnxruntime -lcudart -lcudnn -o test test.cpp