#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

// To represent predicted action segments
#ifndef ACTION_SEGMENT_DEF
#define ACTION_SEGMENT_DEF
struct ActionSegment {
    float start_time;
    float end_time;
    int label;
    float score;
};
#endif

class Tridet {
public:
    Tridet();
    ~Tridet();

    /**
     * @brief Initialize Tridet model
     * @param model_path Path to model (.onnx)
     * @param device_id GPU device ID (-1 for CPU)
     * @param num_classes Number of action categories (e.g. 11)
     * @param use_trident Whether the model was trained with Trident head
     * @param num_bins Number of bins for Trident head offset encoding
     * @return 0 for success, others for failure
     */
    int Init(const std::string& model_path, int device_id = 0, int num_classes = 11, bool use_trident = true, int num_bins = 16);

    /**
     * @brief Run Tridet model continually with the newest extracted feature
     * @param new_feature The 1024-dim feature vector from I3D
     * @param fps Video FPS to calculate start/end time
     * @param chunk_size Usually 16 frames per feature
     * @return A list of the latest detected action segments
     */
    std::vector<ActionSegment> Run(const std::vector<float>& new_feature, float fps, int chunk_size);

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "Tridet"};
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};

    std::vector<const char*> input_node_names_ = {"features"};
    std::vector<const char*> output_node_names_ = {"cls_logits", "offsets", "lb_logits", "rb_logits", "points"};

    int num_classes_;
    int64_t max_seq_len_ = 2304;
    int64_t input_dim_ = 1024;
    bool use_trident_head_;
    int num_bins_;
    
    // Feature buffer
    std::vector<std::vector<float>> feature_buffer_;

    // Helper functions for postprocessing
    float sigmoid(float x);
    std::vector<float> softmax(const std::vector<float>& logits);
};
