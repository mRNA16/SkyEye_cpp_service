#include "yolo/yolo_detector.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    std::string model_path = R"(E:\pilot\yolo\config\best.onnx)";
    std::string image_path = R"(E:\pilot\test\test_yolo.png)";

    YoloPoseDetector detector;
    if (detector.Init(model_path, 0) != 0) {
        std::cerr << "Failed to initialize detector" << std::endl;
        return -1;
    }

    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    auto detections = detector.Detect(frame);

    std::cout << "Detected " << detections.size() << " objects." << std::endl;

    for (const auto& det : detections) {
        std::cout << "Class: " << det.label 
                  << " Score: " << det.score 
                  << " Box: [" << det.box.x << ", " << det.box.y << ", " << det.box.width << ", " << det.box.height << "]" << std::endl;
        
        for (size_t i = 0; i < det.keypoints.size(); ++i) {
            std::cout << "  Keypoint " << i << ": (" << det.keypoints[i].x << ", " << det.keypoints[i].y << ") Conf: " << det.keypoints[i].confidence << std::endl;
        }

        // Draw results
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label_str = "Class " + std::to_string(det.label) + " " + std::to_string(det.score).substr(0, 4);
        cv::putText(frame, label_str, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        for (const auto& kpt : det.keypoints) {
            if (kpt.confidence > 0.5) {
                cv::circle(frame, cv::Point((int)kpt.x, (int)kpt.y), 5, cv::Scalar(0, 0, 255), -1);
            }
        }
    }

    // Save result
    std::string output_path = R"(E:\pilot\test\detect_yolo.png)";
    cv::imwrite(output_path, frame);
    std::cout << "Result saved to " << output_path << std::endl;

    return 0;
}
