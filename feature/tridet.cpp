#include "tridet.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

Tridet::Tridet() : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

Tridet::~Tridet() {
}

int Tridet::Init(const std::string& model_path, int device_id, int num_classes, bool use_trident, int num_bins) {
    num_classes_ = num_classes;
    use_trident_head_ = use_trident;
    num_bins_ = num_bins;

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

        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "Tridet Init Error: " << e.what() << std::endl;
        return -1;
    }
}

float Tridet::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<float> Tridet::softmax(const std::vector<float>& logits) {
    if (logits.empty()) return {};
    float max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exp_vals(logits.size());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        exp_vals[i] = std::exp(logits[i] - max_val);
        sum_exp += exp_vals[i];
    }
    for (size_t i = 0; i < exp_vals.size(); ++i) {
        exp_vals[i] /= sum_exp;
    }
    return exp_vals;
}

std::vector<ActionSegment> Tridet::Run(const std::vector<float>& new_feature, float fps, int chunk_size) {
    std::vector<ActionSegment> detected_actions;
    
    if (new_feature.size() != input_dim_) {
        std::cerr << "Tridet Error: Feature dimension mismatch." << std::endl;
        return detected_actions;
    }
    
    feature_buffer_.push_back(new_feature);
    if (feature_buffer_.size() > max_seq_len_) {
        feature_buffer_.erase(feature_buffer_.begin());
    }

    std::vector<float> input_tensor_values(1 * input_dim_ * max_seq_len_, 0.0f);
    
    size_t seq_len = feature_buffer_.size();
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t c = 0; c < input_dim_; ++c) {
            input_tensor_values[c * max_seq_len_ + t] = feature_buffer_[t][c];
        }
    }

    std::vector<int64_t> input_shape = {1, input_dim_, max_seq_len_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());
        
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));

    try {
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr}, 
            input_node_names_.data(), input_tensors.data(), input_tensors.size(), 
            output_node_names_.data(), output_node_names_.size()
        );

        // Parse outputs: cls_logits, offsets, lb_logits, rb_logits, points
        float* cls_logits = output_tensors[0].GetTensorMutableData<float>();
        float* offsets = output_tensors[1].GetTensorMutableData<float>();
        float* lb_logits = output_tensors[2].GetTensorMutableData<float>();
        float* rb_logits = output_tensors[3].GetTensorMutableData<float>();
        float* points = output_tensors[4].GetTensorMutableData<float>();

        auto cls_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t total_pts = cls_shape[1]; // Total sum of T_i
        
        float pre_nms_thresh = 0.001f;
        std::vector<ActionSegment> proposals;

        for (int i = 0; i < total_pts; ++i) {
            float pt_t = points[i * 4 + 0];
            float pt_stride = points[i * 4 + 3];

            for (int c = 0; c < num_classes_; ++c) {
                float prob = sigmoid(cls_logits[i * num_classes_ + c]);

                if (prob > pre_nms_thresh) {
                    float offset_l = 0.0f, offset_r = 0.0f;

                    if (use_trident_head_) {
                        std::vector<float> s_logits(num_bins_ + 1, 0.0f);
                        std::vector<float> e_logits(num_bins_ + 1, 0.0f);

                        for (int k = 0; k <= num_bins_; ++k) {
                            int idx_l = i - num_bins_ + k;
                            if (idx_l >= 0 && points[idx_l * 4 + 3] == pt_stride) {
                                s_logits[k] = lb_logits[idx_l * num_classes_ + c];
                            }

                            int idx_r = i + k;
                            if (idx_r < total_pts && points[idx_r * 4 + 3] == pt_stride) {
                                e_logits[k] = rb_logits[idx_r * num_classes_ + c];
                            }

                            // Add center offset value
                            int offset_idx_l = i * 2 * (num_bins_ + 1) + k;
                            int offset_idx_r = i * 2 * (num_bins_ + 1) + (num_bins_ + 1) + k;
                            s_logits[k] += offsets[offset_idx_l];
                            e_logits[k] += offsets[offset_idx_r];
                        }

                        std::vector<float> prob_l = softmax(s_logits);
                        std::vector<float> prob_r = softmax(e_logits);

                        for (int k = 0; k <= num_bins_; ++k) {
                            offset_l += prob_l[k] * (num_bins_ - k);
                            offset_r += prob_r[k] * k;
                        }

                    } else {
                        offset_l = offsets[i * 2 + 0];
                        offset_r = offsets[i * 2 + 1];
                    }

                    float start_feat = pt_t - offset_l * pt_stride;
                    float end_feat = pt_t + offset_r * pt_stride;
                    float seg_area = end_feat - start_feat;

                    if (seg_area > 0.05f) {
                        float feat_stride = 4.0f; 
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

        // Soft-NMS processing
        std::sort(proposals.begin(), proposals.end(), [](const ActionSegment& a, const ActionSegment& b){
            return a.score > b.score;
        });

        int pre_nms_topk = 2000;
        if (proposals.size() > pre_nms_topk) proposals.erase(proposals.begin() + pre_nms_topk, proposals.end());

        float nms_sigma = 0.5f; 
        float min_score = 0.001f;
        int max_seg_num = 200;
        std::vector<ActionSegment> keep_segments;

        for (int c = 0; c < num_classes_; ++c) {
            std::vector<ActionSegment> cls_proposals;
            for (const auto& p : proposals) if (p.label == c) cls_proposals.push_back(p);

            while (!cls_proposals.empty()) {
                auto max_it = std::max_element(cls_proposals.begin(), cls_proposals.end(), [](const ActionSegment& a, const ActionSegment& b) {
                    return a.score < b.score;
                });
                ActionSegment best_seg = *max_it;
                cls_proposals.erase(max_it);
                keep_segments.push_back(best_seg);

                std::vector<ActionSegment> remaining;
                for (auto& p : cls_proposals) {
                    float inter_start = std::max(best_seg.start_time, p.start_time);
                    float inter_end = std::min(best_seg.end_time, p.end_time);
                    float inter_area = std::max(0.0f, inter_end - inter_start);
                    float union_area = (best_seg.end_time - best_seg.start_time) + (p.end_time - p.start_time) - inter_area;
                    float iou = (union_area > 0) ? (inter_area / union_area) : 0.0f;

                    p.score = p.score * std::exp(-(iou * iou) / nms_sigma);
                    if (p.score > min_score) remaining.push_back(p);
                }
                cls_proposals = remaining;
            }
        }

        std::sort(keep_segments.begin(), keep_segments.end(), [](const ActionSegment& a, const ActionSegment& b){
            return a.score > b.score;
        });
        if (keep_segments.size() > max_seg_num) keep_segments.erase(keep_segments.begin() + max_seg_num, keep_segments.end());

        detected_actions = std::move(keep_segments);

    } catch (const Ort::Exception& e) {
        std::cerr << "Tridet Run Error: " << e.what() << std::endl;
    }

    return detected_actions;
}

std::vector<ActionSegment> Tridet::RunOffline(const std::vector<std::vector<float>>& all_features, float fps, int chunk_size, const std::vector<float>& global_logits) {
    std::vector<ActionSegment> all_proposals;
    if (all_features.empty()) return all_proposals;
    
    // N个特征组（每组代表一个 chunk_size 长的源视频）
    // 为了平滑过渡避免动作腰斩，每次推进 1/2 个最大窗口长以利用 NMS 重叠消除进行补偿关联
    int step = max_seq_len_ / 2; 

    for (size_t start_idx = 0; start_idx < all_features.size(); start_idx += step) {
        std::vector<float> input_tensor_values(1 * input_dim_ * max_seq_len_, 0.0f);
        size_t end_idx = std::min(start_idx + max_seq_len_, all_features.size());
        size_t actual_len = end_idx - start_idx;
        
        for (size_t t = 0; t < actual_len; ++t) {
            for (size_t c = 0; c < input_dim_; ++c) {
                input_tensor_values[c * max_seq_len_ + t] = all_features[start_idx + t][c];
            }
        }
        
        std::vector<int64_t> input_shape = {1, input_dim_, max_seq_len_};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());
            
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor));

        try {
            auto output_tensors = session_.Run(
                Ort::RunOptions{nullptr}, 
                input_node_names_.data(), input_tensors.data(), input_tensors.size(), 
                output_node_names_.data(), output_node_names_.size()
            );

            float* cls_logits = output_tensors[0].GetTensorMutableData<float>();
            float* offsets = output_tensors[1].GetTensorMutableData<float>();
            float* lb_logits = output_tensors[2].GetTensorMutableData<float>();
            float* rb_logits = output_tensors[3].GetTensorMutableData<float>();
            float* points = output_tensors[4].GetTensorMutableData<float>();

            auto cls_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            int64_t total_pts = cls_shape[1]; 
            
            float pre_nms_thresh = 0.001f;
            for (int i = 0; i < total_pts; ++i) {
                float pt_t = points[i * 4 + 0];
                float pt_stride = points[i * 4 + 3];

                for (int c = 0; c < num_classes_; ++c) {
                    float prob = sigmoid(cls_logits[i * num_classes_ + c]);

                    if (prob > pre_nms_thresh) {
                        float offset_l = 0.0f, offset_r = 0.0f;
                        if (use_trident_head_) {
                            std::vector<float> s_logits(num_bins_ + 1, 0.0f);
                            std::vector<float> e_logits(num_bins_ + 1, 0.0f);
                            for (int k = 0; k <= num_bins_; ++k) {
                                int idx_l = i - num_bins_ + k;
                                if (idx_l >= 0 && points[idx_l * 4 + 3] == pt_stride) s_logits[k] = lb_logits[idx_l * num_classes_ + c];
                                int idx_r = i + k;
                                if (idx_r < total_pts && points[idx_r * 4 + 3] == pt_stride) e_logits[k] = rb_logits[idx_r * num_classes_ + c];
                                int offset_idx_l = i * 2 * (num_bins_ + 1) + k;
                                int offset_idx_r = i * 2 * (num_bins_ + 1) + (num_bins_ + 1) + k;
                                s_logits[k] += offsets[offset_idx_l];
                                e_logits[k] += offsets[offset_idx_r];
                            }
                            std::vector<float> prob_l = softmax(s_logits);
                            std::vector<float> prob_r = softmax(e_logits);
                            for (int k = 0; k <= num_bins_; ++k) {
                                offset_l += prob_l[k] * (num_bins_ - k);
                                offset_r += prob_r[k] * k;
                            }
                        } else {
                            offset_l = offsets[i * 2 + 0];
                            offset_r = offsets[i * 2 + 1];
                        }

                        float start_feat = pt_t - offset_l * pt_stride;
                        float end_feat = pt_t + offset_r * pt_stride;
                        
                        // 只在此处将滑动窗口内的网络预测偏移矫正回到基于原视频流全长真实时间线上：
                        start_feat += start_idx;
                        end_feat += start_idx;
                        
                        float seg_area = end_feat - start_feat;
                        if (seg_area > 0.05f) {
                            float feat_stride = 4.0f; 
                            float start_time = (start_feat * feat_stride + 0.5f * chunk_size) / fps;
                            float end_time = (end_feat * feat_stride + 0.5f * chunk_size) / fps;
                            ActionSegment seg;
                            seg.start_time = start_time;
                            seg.end_time = end_time;
                            seg.label = c;
                            
                            // 执行分值融合 (Score Fusion)
                            if (!global_logits.empty() && c < global_logits.size()) {
                                // 公式: sqrt(Prob_TAD * Prob_Global)
                                seg.score = std::sqrt(prob * global_logits[c]);
                            } else {
                                seg.score = prob;
                            }
                            
                            all_proposals.push_back(seg);
                        }
                    }
                }
            }
        } catch (const Ort::Exception& e) {
            std::cerr << "Tridet RunOffline Error: " << e.what() << std::endl;
        }
        
        // 由于包含滑动关联策略并向后推 1/2 窗口
        if (end_idx >= all_features.size()) break;
    }
    
    // 对所有的滑动窗口重合的交叉区域或高分建议区进行同一目标的提纯 NMS
    std::sort(all_proposals.begin(), all_proposals.end(), [](const ActionSegment& a, const ActionSegment& b){
        return a.score > b.score;
    });

    float nms_sigma = 0.5f; 
    float min_score = 0.001f;
    std::vector<ActionSegment> keep_segments;

    for (int c = 0; c < num_classes_; ++c) {
        std::vector<ActionSegment> cls_proposals;
        for (const auto& p : all_proposals) if (p.label == c) cls_proposals.push_back(p);

        while (!cls_proposals.empty()) {
            auto max_it = std::max_element(cls_proposals.begin(), cls_proposals.end(), [](const ActionSegment& a, const ActionSegment& b) {
                return a.score < b.score;
            });
            ActionSegment best_seg = *max_it;
            cls_proposals.erase(max_it);
            keep_segments.push_back(best_seg);

            std::vector<ActionSegment> remaining;
            for (auto& p : cls_proposals) {
                float inter_start = std::max(best_seg.start_time, p.start_time);
                float inter_end = std::min(best_seg.end_time, p.end_time);
                float inter_area = std::max(0.0f, inter_end - inter_start);
                float union_area = (best_seg.end_time - best_seg.start_time) + (p.end_time - p.start_time) - inter_area;
                float iou = (union_area > 0) ? (inter_area / union_area) : 0.0f;
                // Soft NMS
                p.score = p.score * std::exp(-(iou * iou) / nms_sigma);
                // 或使用极为保守的硬保留 (这在高并发段长有效)
                // if(iou >= 0.2f) p.score = 0; 
                
                if (p.score > min_score) remaining.push_back(p);
            }
            cls_proposals = remaining;
        }
    }
    return keep_segments;
}
