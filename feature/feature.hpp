#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <./service/config.hpp>

class I3D {
public:
    I3D();
    ~I3D();

    /**
     * @brief 初始化 I3D 模型
     * @param model_path 模型路径 (.onnx)
     * @param device_id GPU 设备 ID (若为 -1 则使用 CPU)
     * @return 0 成功, 其他 失败
     */
    int Init(const std::string& model_path, int device_id = 0);

    /**
     * @brief 提取视频序列特征
     * @param frames 输入帧序列 (通常为 16 帧)
     * @return 提取出的 1024 维特征向量
     */
    std::vector<float> Run(const std::vector<cv::Mat>& frames);

private:
    // ONNX Runtime 相关组件
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "I3D_Feature_Extractor"};
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};

    // 模型输入输出节点名称
    std::vector<const char*> input_node_names_ = {"input"};
    std::vector<const char*> output_node_names_ = {"output"};

    // 输入形状参数
    int64_t batch_size_ = BATCH_SIZE;
    int64_t channels_ = CHANNEL;
    int64_t frames_count_ = CHUNK_SIZE;
    int64_t height_ = INPUT_H;
    int64_t width_ = INPUT_W;

    /**
     * @brief 预处理：将 cv::Mat 序列转换为 Tensor 数据
     * @param frames 输入帧序列
     * @param output_data 输出的平铺数组
     */
    void Preprocess(const std::vector<cv::Mat>& frames, std::vector<float>& output_data);
};