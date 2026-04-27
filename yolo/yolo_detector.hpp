#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct KeyPoint {
    float x;
    float y;
    float confidence;
};

struct DetectionPose {
    cv::Rect box;
    float score;
    int label;
    std::vector<KeyPoint> keypoints;
};

class YoloPoseDetector {
public:
    YoloPoseDetector();
    ~YoloPoseDetector();

    /**
     * @brief Initialize YOLOv8-Pose model
     * @param model_path Path to .onnx model
     * @param device_id GPU device ID (-1 for CPU)
     * @param conf_threshold Confidence threshold
     * @param nms_threshold NMS IOU threshold
     * @return 0 for success, others for failure
     */
    int Init(const std::string& model_path, int device_id = 0, float conf_threshold = 0.25f, float nms_threshold = 0.45f);

    /**
     * @brief Run inference on an image
     * @param frame Input image (BGR)
     * @return List of detections with pose
     */
    std::vector<DetectionPose> Detect(const cv::Mat& frame);

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "YoloPoseDetector"};
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};

    std::vector<const char*> input_node_names_ = {"images"};
    std::vector<const char*> output_node_names_ = {"output0"};

    float conf_threshold_;
    float nms_threshold_;
    int num_classes_ = 6;
    int num_keypoints_ = 1;
    cv::Size input_size_ = cv::Size(640, 640);
    bool initialized_ = false;

    // Helper for preprocessing
    void Preprocess(const cv::Mat& frame, cv::Mat& output, float& scale, cv::Point& pad);
};
