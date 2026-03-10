#include "actionformer.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

ActionFormer::ActionFormer() : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

ActionFormer::~ActionFormer() {
}

int ActionFormer::Init(const std::string& model_path, int device_id, int num_classes) {
    num_classes_ = num_classes;
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (device_id >= 0) {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = device_id;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }

#ifdef _WIN32
        std::wstring w_model_path(model_path.begin(), model_path.end());
        session_ = Ort::Session(env_, w_model_path.c_str(), session_options);
#else
        session_ = Ort::Session(env_, model_path.c_str(), session_options);
#endif

        // Get output names: ActionFormer outputs 6 * 4 = 24 tensors dynamically named.
        size_t num_outputs = session_.GetOutputCount();
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < num_outputs; ++i) {
            auto output_name_ptr = session_.GetOutputNameAllocated(i, allocator);
            output_node_names_str_.push_back(output_name_ptr.get());
        }
        for (const auto& str : output_node_names_str_) {
            output_node_names_.push_back(str.c_str());
        }

        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ActionFormer Init Error: " << e.what() << std::endl;
        return -1;
    }
}

float ActionFormer::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<ActionSegment> ActionFormer::Run(const std::vector<float>& new_feature, float fps, int chunk_size) {
    std::vector<ActionSegment> detected_actions;
    
    if (new_feature.size() != input_dim_) {
        std::cerr << "ActionFormer Error: Feature dimension mismatch (expected 1024)." << std::endl;
        return detected_actions;
    }
    
    feature_buffer_.push_back(new_feature);
    if (feature_buffer_.size() > max_seq_len_) {
        feature_buffer_.erase(feature_buffer_.begin());
    }
    
    total_frames_processed_ += chunk_size;

    // Prepare batched_inputs [1, 1024, 2304] and batched_masks [1, 1, 2304]
    std::vector<float> input_tensor_values(1 * input_dim_ * max_seq_len_, 0.0f);
    std::vector<uint8_t> mask_tensor_values(1 * 1 * max_seq_len_, 0);

    // Copy buffer into input_tensor_values
    // Shape is [batch, channel, seq_len] -> [1, 1024, 2304]
    size_t seq_len = feature_buffer_.size();
    for (size_t t = 0; t < seq_len; ++t) {
        mask_tensor_values[t] = 1;
        for (size_t c = 0; c < input_dim_; ++c) {
            input_tensor_values[c * max_seq_len_ + t] = feature_buffer_[t][c];
        }
    }

    std::vector<int64_t> input_shape = {1, input_dim_, max_seq_len_};
    std::vector<int64_t> mask_shape = {1, 1, max_seq_len_};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());
        
    Ort::Value mask_tensor = Ort::Value::CreateTensor<bool>(
        memory_info_, reinterpret_cast<bool*>(mask_tensor_values.data()), mask_tensor_values.size(),
        mask_shape.data(), mask_shape.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    input_tensors.push_back(std::move(mask_tensor));

    try {
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr}, 
            input_node_names_.data(), input_tensors.data(), input_tensors.size(), 
            output_node_names_.data(), output_node_names_.size()
        );

        // --------------------------------------------------------------------------
        // Post-processing
        // --------------------------------------------------------------------------
        int levels = 6;
        float pre_nms_thresh = 0.001f;
        
        std::vector<ActionSegment> proposals;

        for (int l = 0; l < levels; ++l) {
            auto& logits_tensor = output_tensors[l];
            auto& offsets_tensor = output_tensors[levels + l];
            auto& points_tensor  = output_tensors[3*levels + l];
            
            float* logits = logits_tensor.GetTensorMutableData<float>();
            float* offsets = offsets_tensor.GetTensorMutableData<float>();
            float* points = points_tensor.GetTensorMutableData<float>();
            
            auto logits_shape = logits_tensor.GetTensorTypeAndShapeInfo().GetShape();
            int64_t T_i = logits_shape[1]; // [1, T_i, num_classes]
            
            for (int t = 0; t < T_i; ++t) {
                // ActionFormer Python 是每个类别独立计算 sigmoid
                for (int c = 0; c < num_classes_; ++c) {
                    float prob = sigmoid(logits[t * num_classes_ + c]);
                    
                    // 1. pre_nms_thresh 过滤
                    if (prob > pre_nms_thresh) {
                        float pt_t = points[t * 4 + 0]; // time point
                        float pt_stride = points[t * 4 + 3];
                        float offset_l = offsets[t * 2 + 0];
                        float offset_r = offsets[t * 2 + 1];
                        
                        float start_feat = pt_t - offset_l * pt_stride;
                        float end_feat = pt_t + offset_r * pt_stride;
                        
                        // ActionFormer 训练时有一个 duration_thresh，过短的不要
                        float seg_area = end_feat - start_feat;
                        if (seg_area > 0.05f) { // duration thresh (相对于 feature grid)
                            // 换算成最终视频秒数
                            float feat_stride = 4.0f; // Config: feat_stride: 4
                            int feat_num_frames = 16;
                            float start_time = (start_feat * feat_stride + 0.5f * feat_num_frames) / fps;
                            float end_time = (end_feat * feat_stride + 0.5f * feat_num_frames) / fps;

                            ActionSegment seg;
                            seg.start_time = start_time;
                            seg.end_time = end_time;
                            seg.score = prob;
                            seg.label = c;
                            proposals.push_back(seg);
                        }
                    }
                }
            }
        }
        
        // ------------- 2. NMS (非极大值抑制) / Soft-NMS -------------
        // 按照得分进行全局降序排序
        std::sort(proposals.begin(), proposals.end(), [](const ActionSegment& a, const ActionSegment& b){
            return a.score > b.score;
        });

        // 取 topK proposals (例如 2000)
        int pre_nms_topk = 2000;
        if (proposals.size() > pre_nms_topk) {
            proposals.erase(proposals.begin() + pre_nms_topk, proposals.end());
        }

        // C++ 实现的 Gaussian Soft-NMS
        float nms_sigma = 0.5f; 
        float min_score = 0.001f;
        int max_seg_num = 200;

        std::vector<ActionSegment> keep_segments;

        // 这里我们实现简化的每类别独立的 Soft NMS
        for (int c = 0; c < num_classes_; ++c) {
            std::vector<ActionSegment> cls_proposals;
            for (const auto& p : proposals) if (p.label == c) cls_proposals.push_back(p);

            while (!cls_proposals.empty()) {
                // 取出当前类别得分最高的作为基准
                auto max_it = std::max_element(cls_proposals.begin(), cls_proposals.end(), [](const ActionSegment& a, const ActionSegment& b) {
                    return a.score < b.score;
                });
                ActionSegment best_seg = *max_it;
                cls_proposals.erase(max_it);

                keep_segments.push_back(best_seg);

                // 根据 best_seg 计算 Soft-NMS 分数衰减
                std::vector<ActionSegment> remaining;
                for (auto& p : cls_proposals) {
                    // 计算 IoU
                    float inter_start = std::max(best_seg.start_time, p.start_time);
                    float inter_end = std::min(best_seg.end_time, p.end_time);
                    float inter_area = std::max(0.0f, inter_end - inter_start);
                    float union_area = (best_seg.end_time - best_seg.start_time) + (p.end_time - p.start_time) - inter_area;
                    float iou = (union_area > 0) ? (inter_area / union_area) : 0.0f;

                    // Gaussian penalty
                    p.score = p.score * std::exp(-(iou * iou) / nms_sigma);

                    if (p.score > min_score) {
                        remaining.push_back(p);
                    }
                }
                cls_proposals = remaining;
            }
        }

        // 最后再次统一按分数排序并限制 max_seg_num
        std::sort(keep_segments.begin(), keep_segments.end(), [](const ActionSegment& a, const ActionSegment& b){
            return a.score > b.score;
        });
        if (keep_segments.size() > max_seg_num) {
            keep_segments.erase(keep_segments.begin() + max_seg_num, keep_segments.end());
        }

        detected_actions = std::move(keep_segments);

    } catch (const Ort::Exception& e) {
        std::cerr << "ActionFormer Run Error: " << e.what() << std::endl;
    }

    return detected_actions;
}
