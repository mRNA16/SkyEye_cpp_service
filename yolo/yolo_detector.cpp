#include "yolo_detector.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {
#ifdef _WIN32
std::wstring ToWidePath(const std::string& path) {
    if (path.empty()) {
        return std::wstring();
    }

    int len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.data(), (int)path.size(), nullptr, 0);
    UINT code_page = CP_UTF8;
    DWORD flags = MB_ERR_INVALID_CHARS;
    if (len <= 0) {
        code_page = CP_ACP;
        flags = 0;
        len = MultiByteToWideChar(code_page, flags, path.data(), (int)path.size(), nullptr, 0);
    }

    if (len <= 0) {
        return std::wstring(path.begin(), path.end());
    }

    std::wstring wide_path((size_t)len, L'\0');
    MultiByteToWideChar(code_page, flags, path.data(), (int)path.size(), wide_path.data(), len);
    return wide_path;
}
#endif

cv::Rect ClipBox(float x, float y, float width, float height, const cv::Size& image_size) {
    int left = std::clamp((int)std::floor(x), 0, image_size.width);
    int top = std::clamp((int)std::floor(y), 0, image_size.height);
    int right = std::clamp((int)std::ceil(x + width), 0, image_size.width);
    int bottom = std::clamp((int)std::ceil(y + height), 0, image_size.height);

    if (right <= left || bottom <= top) {
        return cv::Rect();
    }
    return cv::Rect(left, top, right - left, bottom - top);
}
}

YoloPoseDetector::YoloPoseDetector() : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

YoloPoseDetector::~YoloPoseDetector() {
}

int YoloPoseDetector::Init(const std::string& model_path, int device_id, float conf_threshold, float nms_threshold) {
    conf_threshold_ = conf_threshold;
    nms_threshold_ = nms_threshold;
    initialized_ = false;

    auto create_session = [&](bool use_cuda) {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (use_cuda) {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = device_id;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }

#ifdef _WIN32
        std::wstring w_model_path = ToWidePath(model_path);
        session_ = Ort::Session(env_, w_model_path.c_str(), session_options);
#else
        session_ = Ort::Session(env_, model_path.c_str(), session_options);
#endif
    };

    try {
        if (device_id >= 0) {
            try {
                create_session(true);
            } catch (const Ort::Exception& e) {
                std::cerr << "YoloPoseDetector CUDA Init Warning: " << e.what()
                          << ". Falling back to CPU." << std::endl;
                create_session(false);
            }
        } else {
            create_session(false);
        }
        initialized_ = true;
        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "YoloPoseDetector Init Error: " << e.what() << std::endl;
        return -1;
    }
}

void YoloPoseDetector::Preprocess(const cv::Mat& frame, cv::Mat& output, float& scale, cv::Point& pad) {
    int w = frame.cols;
    int h = frame.rows;
    scale = std::min((float)input_size_.width / w, (float)input_size_.height / h);
    int nw = (int)(w * scale);
    int nh = (int)(h * scale);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh));

    output = cv::Mat::ones(input_size_, CV_8UC3) * 114;
    pad.x = (input_size_.width - nw) / 2;
    pad.y = (input_size_.height - nh) / 2;
    resized.copyTo(output(cv::Rect(pad.x, pad.y, nw, nh)));

    // Standardize: BGR to RGB, then normalize to [0, 1]
    cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
    output.convertTo(output, CV_32FC3, 1.0 / 255.0);
}

std::vector<DetectionPose> YoloPoseDetector::Detect(const cv::Mat& frame) {
    std::vector<DetectionPose> results;
    if (frame.empty()) return results;
    if (!initialized_) {
        std::cerr << "YoloPoseDetector Detect Error: detector is not initialized." << std::endl;
        return results;
    }

    cv::Mat preprocessed;
    float scale;
    cv::Point pad;
    Preprocess(frame, preprocessed, scale, pad);

    // Prepare input tensor (NCHW)
    std::vector<float> input_tensor_values;
    input_tensor_values.resize(1 * 3 * input_size_.width * input_size_.height);
    
    std::vector<cv::Mat> input_channels(3);
    for (int i = 0; i < 3; ++i) {
        input_channels[i] = cv::Mat(input_size_, CV_32FC1, input_tensor_values.data() + i * input_size_.width * input_size_.height);
    }
    cv::split(preprocessed, input_channels);

    std::vector<int64_t> input_shape = {1, 3, input_size_.height, input_size_.width};
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

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        const int expected_channels = 4 + num_classes_ + num_keypoints_ * 3;
        if (output_shape.size() != 3 || output_shape[0] != 1) {
            std::cerr << "YoloPoseDetector Detect Error: unexpected output rank/shape. Rank="
                      << output_shape.size() << std::endl;
            return results;
        }

        bool channel_first = false;
        int output_channels = 0;
        int anchors = 0;
        if (output_shape[1] == expected_channels) {
            channel_first = true;
            output_channels = (int)output_shape[1];
            anchors = (int)output_shape[2];
        } else if (output_shape[2] == expected_channels) {
            channel_first = false;
            output_channels = (int)output_shape[2];
            anchors = (int)output_shape[1];
        } else {
            std::cerr << "YoloPoseDetector Detect Error: expected " << expected_channels
                      << " output channels, got shape ["
                      << output_shape[0] << ", " << output_shape[1] << ", " << output_shape[2]
                      << "]." << std::endl;
            return results;
        }

        auto output_at = [&](int c, int anchor) -> float {
            if (channel_first) {
                return output_data[c * anchors + anchor];
            }
            return output_data[anchor * output_channels + c];
        };

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<KeyPoint>> all_keypoints;

        for (int i = 0; i < anchors; ++i) {
            // Find max class score
            float max_score = 0;
            int class_id = -1;
            for (int j = 0; j < num_classes_; ++j) {
                float score = output_at(4 + j, i);
                if (score > max_score) {
                    max_score = score;
                    class_id = j;
                }
            }

            if (max_score > conf_threshold_) {
                float cx = output_at(0, i);
                float cy = output_at(1, i);
                float w = output_at(2, i);
                float h = output_at(3, i);

                // Recover to original image size
                float x = (cx - w / 2 - pad.x) / scale;
                float y = (cy - h / 2 - pad.y) / scale;
                float width = w / scale;
                float height = h / scale;
                cv::Rect clipped_box = ClipBox(x, y, width, height, frame.size());
                if (clipped_box.empty()) {
                    continue;
                }

                boxes.push_back(clipped_box);
                confidences.push_back(max_score);
                class_ids.push_back(class_id);

                // Extract keypoints
                std::vector<KeyPoint> kpts;
                for (int k = 0; k < num_keypoints_; ++k) {
                    float kx = (output_at(4 + num_classes_ + k * 3 + 0, i) - pad.x) / scale;
                    float ky = (output_at(4 + num_classes_ + k * 3 + 1, i) - pad.y) / scale;
                    float kc = output_at(4 + num_classes_ + k * 3 + 2, i);
                    kx = std::clamp(kx, 0.0f, (float)(frame.cols - 1));
                    ky = std::clamp(ky, 0.0f, (float)(frame.rows - 1));
                    kpts.push_back({kx, ky, kc});
                }
                all_keypoints.push_back(kpts);
            }
        }

        std::vector<int> indices;
        for (int cls = 0; cls < num_classes_; ++cls) {
            std::vector<cv::Rect> class_boxes;
            std::vector<float> class_scores;
            std::vector<int> original_indices;

            for (int i = 0; i < (int)class_ids.size(); ++i) {
                if (class_ids[i] == cls) {
                    class_boxes.push_back(boxes[i]);
                    class_scores.push_back(confidences[i]);
                    original_indices.push_back(i);
                }
            }

            if (class_boxes.empty()) {
                continue;
            }

            std::vector<int> class_keep;
            cv::dnn::NMSBoxes(class_boxes, class_scores, conf_threshold_, nms_threshold_, class_keep);
            for (int keep_idx : class_keep) {
                indices.push_back(original_indices[keep_idx]);
            }
        }

        std::sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
            return confidences[lhs] > confidences[rhs];
        });

        for (int idx : indices) {
            DetectionPose det;
            det.box = boxes[idx];
            det.score = confidences[idx];
            det.label = class_ids[idx];
            det.keypoints = all_keypoints[idx];
            results.push_back(det);
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "YoloPoseDetector Detect Error: " << e.what() << std::endl;
    }

    return results;
}
