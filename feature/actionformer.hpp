#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

// To represent predicted action segments
struct ActionSegment {
    float start_time;
    float end_time;
    int label;
    float score;
};

class ActionFormer {
public:
    ActionFormer();
    ~ActionFormer();

    /**
     * @brief Initialize ActionFormer model
     * @param model_path Path to model (.onnx)
     * @param device_id GPU device ID (-1 for CPU)
     * @param num_classes Number of action categories (e.g. 11)
     * @return 0 for success, others for failure
     */
    int Init(const std::string& model_path, int device_id = 0, int num_classes = 11);

    /**
     * @brief Run ActionFormer model continually with the newest extracted feature
     * @param new_feature The 1024-dim feature vector from I3D
     * @param fps Video FPS to calculate start/end time
     * @param chunk_size Usually 16 frames per feature
     * @return A list of the latest detected action segments
     */
    std::vector<ActionSegment> Run(const std::vector<float>& new_feature, float fps, int chunk_size);

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "ActionFormer"};
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};

    std::vector<const char*> input_node_names_ = {"inputs", "masks"};
    // Since there are 24 output tensors from 6 levels * 4 (logits, offsets, masks, points)
    std::vector<const char*> output_node_names_;
    std::vector<std::string> output_node_names_str_;

    int num_classes_;
    int64_t max_seq_len_ = 2304;
    int64_t input_dim_ = 1024;
    
    // Feature buffer
    std::vector<std::vector<float>> feature_buffer_;
    int total_frames_processed_ = 0;

    // Helper functions for postprocessing
    float sigmoid(float x);
};
