#include "feature.hpp"
#include <numeric>
#include <algorithm>

I3D::I3D() : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

I3D::~I3D() {
}

int I3D::Init(const std::string& model_path, int device_id) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 如果 device_id >= 0，尝试使用 CUDA
        if (device_id >= 0) {
            // 注意：在实际部署时，需确保链接了 ONNX Runtime 的 CUDA 版本
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = device_id;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }

        // 加载模型
        // Windows 下 Ort::Session 构造函数接收宽字符串
#ifdef _WIN32
        std::wstring w_model_path(model_path.begin(), model_path.end());
        session_ = Ort::Session(env_, w_model_path.c_str(), session_options);
#else
        session_ = Ort::Session(env_, model_path.c_str(), session_options);
#endif

        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "I3D Init Error: " << e.what() << std::endl;
        return -1;
    }
}

I3DOutput I3D::Run(const std::vector<cv::Mat>& frames) {
    if (frames.size() != static_cast<size_t>(frames_count_)) {
        std::cerr << "I3D Run Error: frames count must be " << frames_count_ << std::endl;
        return {};
    }

    try {
        // 1. 预处理数据
        std::vector<float> input_tensor_values(batch_size_ * channels_ * frames_count_ * height_ * width_);
        Preprocess(frames, input_tensor_values);

        // 2. 创建输入 Tensor
        std::vector<int64_t> input_shape = {batch_size_, channels_, frames_count_, height_, width_};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size()
        );

        // 3. 运行推理 (请求两个输出)
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr}, 
            input_node_names_.data(), &input_tensor, 1, 
            output_node_names_.data(), output_node_names_.size()
        );

        // 4. 解析输出
        I3DOutput result;
        
        // 解析第一个输出: features (1024维)
        float* feat_ptr = output_tensors[0].GetTensorMutableData<float>();
        auto feat_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        result.features.assign(feat_ptr, feat_ptr + feat_info.GetElementCount());

        // 解析第二个输出: logits (17维)
        float* logit_ptr = output_tensors[1].GetTensorMutableData<float>();
        auto logit_info = output_tensors[1].GetTensorTypeAndShapeInfo();
        result.logits.assign(logit_ptr, logit_ptr + logit_info.GetElementCount());

        return result;

    } catch (const Ort::Exception& e) {
        std::cerr << "I3D Run Error: " << e.what() << std::endl;
        return {};
    }
}

void I3D::Preprocess(const std::vector<cv::Mat>& frames, std::vector<float>& output_data) {
    // 数据布局要求: NCHWD (N=1, C=3, T=16, H=224, W=224)
    // 索引公式: c*T*H*W + t*H*W + h*W + w
    
    int64_t T = frames_count_;
    int64_t H = height_;
    int64_t W = width_;

    for (int t = 0; t < T; ++t) {
        cv::Mat resized;
        // 1. Resize 到 224x224
        cv::resize(frames[t], resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
        
        // 2. BGR 转 RGB
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        // 3. 填充到 output_data
        // 这里手动映射到 NCHWD 布局
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                cv::Vec3b pixel = resized.at<cv::Vec3b>(h, w);
                
                // 归一化: ImageNet (value / 255.0 - mean) / std 
                float r = (static_cast<float>(pixel[0]) / 255.0f - 0.485f) / 0.229f;
                float g = (static_cast<float>(pixel[1]) / 255.0f - 0.456f) / 0.224f;
                float b = (static_cast<float>(pixel[2]) / 255.0f - 0.406f) / 0.225f;

                // C=0: R, C=1: G, C=2: B
                output_data[0 * T * H * W + t * H * W + h * W + w] = r;
                output_data[1 * T * H * W + t * H * W + h * W + w] = g;
                output_data[2 * T * H * W + t * H * W + h * W + w] = b;
            }
        }
    }
}
