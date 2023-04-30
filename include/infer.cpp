#include <iostream>
#include <vector>
#include <algorithm>
#include "infer.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

extern "C"
{
    void inferCifar(const char *model_path, const char *image_path)
    {

        std::vector<const char *> input_names;
        std::vector<const char *> output_names;

        Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "example"); 
        Ort::SessionOptions session_options;
        Ort::Session session(env, model_path, session_options);

        // 获得模型有多少个输入和输出，因为是三输入三输出网络，那么input和output数量都为3
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();
        size_t num_output_nodes = session.GetOutputCount();

        // 获得模型输入和输出的名称
        const auto input_name = session.GetInputName(0, allocator);
        const auto output_name = session.GetOutputName(0, allocator);
        input_names.push_back(input_name);
        output_names.push_back(output_name);
        std::cout << "inputs数量:" << num_input_nodes << " "
                  << "outputs数量:" << num_output_nodes << std::endl;
        std::cout << input_name << " " << output_name << std::endl;

        // 获取输入张量的元数据
        Ort::TypeInfo input_info = session.GetInputTypeInfo(0);          // 用于获取 ONNX 模型的第一个输入的类型信息。
        auto input_tensor_info = input_info.GetTensorTypeAndShapeInfo(); // 用于获取ONNX模型中张量的类型和形状信息

        // 获取数据形状
        std::vector<int64_t> input_shape = input_tensor_info.GetShape();
        // 遍历vector并打印每个元素
        for (size_t i = 0; i < input_shape.size(); i++)
        {
            std::cout << input_shape[i] << " ";
        }

        std::vector<float> input_values(input_tensor_info.GetElementCount()); // 获取Tensor的元素数量：
        std::cout << input_tensor_info.GetElementCount() << std::endl;

        // 设置模型的输入tensor
        cv::Mat imageBGR = cv::imread(image_path, cv::IMREAD_COLOR);
        cv::Mat scaledImage, preprocessedImage;
        imageBGR.convertTo(scaledImage, CV_32F, 2.0f / 255.0f, -1.0f);                         // Scale image pixels from [0 255] to [-1, 1]
        cv::dnn::blobFromImage(scaledImage, preprocessedImage);                                //  Convert HWC to CHW
        input_values.assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>()); // Assign the input image to the input tensor

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        // vector<Ort::Value> input_tensors = {input_tensor};
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(
            Ort::Value(Ort::Value::CreateTensor<float>(memory_info,
                                                       input_values.data(),
                                                       input_values.size(),
                                                       input_shape.data(),
                                                       input_shape.size())));

        printf("haha \n");

        // 获取输出张量的元数据
        Ort::TypeInfo output_info = session.GetOutputTypeInfo(0);          // 用于获取 ONNX 模型的第一个输入的类型信息。
        auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo(); // 用于获取ONNX模型中张量的类型和形状信息

        // 获取数据形状
        std::vector<int64_t> output_shape = output_tensor_info.GetShape();
        std::cout << "output_shape: :";
        for (size_t i = 0; i < output_shape.size(); i++)
        {
            std::cout << output_shape[i] << " ";
        }
        std::cout << std::endl;
        std::vector<float> output_values(input_tensor_info.GetElementCount()); // 获取Tensor的元素数量：

        std::vector<Ort::Value> output_tensors;
        output_tensors.push_back(
            Ort::Value::CreateTensor<float>(memory_info,
                                            output_values.data(),
                                            output_values.size(),
                                            output_shape.data(),
                                            output_shape.size()));

        session.Run(Ort::RunOptions{nullptr},
                    input_names.data(),
                    input_tensors.data(),
                    input_names.size(),
                    output_names.data(),
                    output_tensors.data(),
                    output_names.size());

        float *output_data = output_tensors[0].GetTensorMutableData<float>();
        int predicted_digit = std::distance(output_data, std::max_element(output_data, output_data + 10));

        std::cout << "Predicted digit: " << predicted_digit << std::endl;
    }
}
    