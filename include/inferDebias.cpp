#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
// renwenhao

extern "C"
{
    int inferDebias(const char *model_path, const char *rec_path)
    {
        std::vector<const char *> input_names;
        std::vector<const char *> output_names;
        std::string line;
        std::vector<std::string> lines;
        std::vector<int> user(100);
        std::vector<int> item(100);
        std::ifstream infile;
        infile.open(rec_path);
        // if (!infile)
        // {
        //     std::cout << "无法打开文件!" << std::endl;
        //     exit(1);
        // }

        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "example"); // ORT_LOGGING_LEVEL_VERBOSE 获取更详细的日志信息
        Ort::SessionOptions session_options;

        OrtCUDAProviderOptions options;
        options.device_id = 0;
        options.arena_extend_strategy = 0;
        options.gpu_mem_limit = (size_t)1 * 1024 * 1024 * 1024; // 1G内存
        options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
        options.do_copy_in_default_stream = 1;
        session_options.AppendExecutionProvider_CUDA(options);

        // Ort::Session session{env, model_path, Ort::SessionOptions{nullptr}}; // CPU

        Ort::Session session{env, model_path, session_options}; // GPU
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();
        size_t num_output_nodes = session.GetOutputCount();

        // 获得模型输入和输出的名称
        const auto input_name_1 = session.GetInputName(0, allocator);
        const auto input_name_2 = session.GetInputName(1, allocator);
        const auto output_name = session.GetOutputName(0, allocator);
        input_names.push_back(input_name_1);
        input_names.push_back(input_name_2);
        output_names.push_back(output_name);

        // 获取输入张量的元数据
        Ort::TypeInfo input_info_1 = session.GetInputTypeInfo(0);            // 用于获取 ONNX 模型的第一个输入的类型信息。
        auto input_tensor_info_1 = input_info_1.GetTensorTypeAndShapeInfo(); // 用于获取ONNX模型中张量的类型和形状信息
        std::vector<int64_t> input_shape_1 = input_tensor_info_1.GetShape(); // 获取数据形状
        std::vector<int> input_values_1(input_tensor_info_1.GetElementCount()); // 获取Tensor的元素数量：
        // cout << input_tensor_info_1.GetElementCount() << endl;

        Ort::TypeInfo input_info_2 = session.GetInputTypeInfo(1);            // 用于获取 ONNX 模型的第2个输入的类型信息。
        auto input_tensor_info_2 = input_info_2.GetTensorTypeAndShapeInfo(); // 用于获取ONNX模型中张量的类型和形状信息
        std::vector<int64_t> input_shape_2 = input_tensor_info_2.GetShape(); // 获取数据形状
        std::vector<int> input_values_2(input_tensor_info_2.GetElementCount()); // 获取Tensor的元素数量：
        // cout << input_tensor_info_2.GetElementCount() << endl;

        int count = 0;
        while (getline(infile, line))
        {
            lines.push_back(line);
        }

        infile.close();
        for (std::string l : lines)
        {
            std::string space_delimiter = " ";
            size_t pos = 0;
            int count = 0;
            while ((pos = l.find(space_delimiter)) != std::string::npos)
            {
                if (count == 0)
                {
                    int u = std::stoi(l.substr(0, pos));
                    for (int i = 0; i < 100; ++i)
                    {
                        user.emplace_back(u);
                    }
                    l.erase(0, pos + space_delimiter.length());
                    // cout << u << endl;
                    ++count;
                    continue;
                }
                else
                {
                    item.emplace_back(std::stoi(l.substr(0, pos)));
                    l.erase(0, pos + space_delimiter.length());
                }
            }
            item.emplace_back(std::stoi(l.substr(0, l.length())));
            count = 0;
        }
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value(Ort::Value::CreateTensor<int>(memory_info,
                                                                         user.data(),
                                                                         user.size(),
                                                                         input_shape_1.data(),
                                                                         input_shape_1.size())));
        input_tensors.push_back(Ort::Value(Ort::Value::CreateTensor<int>(memory_info,
                                                                         item.data(),
                                                                         item.size(),
                                                                         input_shape_2.data(),
                                                                         input_shape_2.size())));

        // 获取输出张量的元数据
        Ort::TypeInfo output_info = session.GetOutputTypeInfo(0);          // 用于获取 ONNX 模型的第一个输入的类型信息。
        auto output_tensor_info = output_info.GetTensorTypeAndShapeInfo(); // 用于获取ONNX模型中张量的类型和形状信息
        std::vector<int64_t> output_shape = output_tensor_info.GetShape(); // 获取数据形状
        std::vector<float> output_values(output_tensor_info.GetElementCount());
        // cout << output_tensor_info.GetElementCount() << endl; // 获取Tensor的元素数量：

        std::vector<Ort::Value> output_tensors;
        output_tensors.push_back(Ort::Value::CreateTensor<float>(memory_info,
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
        count = 0;
        std::vector<std::pair<int, float>> res;
        for (int i : item)
        {
            res.emplace_back(std::make_pair(i, output_data[count++]));
        }
        std::sort(res.begin(), res.end(), [&](std::pair<int, float> a, std::pair<int, float> b)
                  { return a.second > b.second; });
        return  res[0].first;
    }
}

// g++ -std=c++11 -I/path/to/onnxruntime/include -L/path/to/onnxruntime/lib -lonnxruntime -lcudart -lcudnn -o test test.cpp