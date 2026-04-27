#pragma once

#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <sstream>
#include <opencv2/opencv.hpp>

#include "httplib.h"
#include "json.hpp"
#include "thread_safe_dict.hpp"
#include "thread_safe_queue.hpp"
#include "hybrid_video_queue.hpp"
#include "../feature/feature.hpp"
#include "../feature/tridet.hpp"
#include "../yolo/yolo_detector.hpp"

#include <rtc/rtc.hpp>
#include <memory>
#include <map>
#include <mutex>
#include <future>
#include <optional>

using json = nlohmann::json;

class PilotWebServer {
public:
	int boot();
private:
	int set_server_logger();
	int set_camera_interface();

protected:
	int loadModels();
	int distribute_GPU(int occupy, int design);
	int cancel_GPU(int gpu_id, int occupy);
	int launch_camera(const std::string& camera_id,const std::string& input_url);
	int launch_local_video(const std::string& session_id, const std::string& file_path);
	int live(ThreadSafeQueue<cv::Mat>&, const std::string&);
	cv::Mat draw_yolo_detections(const cv::Mat& frame, const std::vector<DetectionPose>& detections);
	int extract_features(HybridVideoQueue&, ThreadSafeQueue<std::vector<float>>&,
	                     std::vector<float>& logits_accum, int& logits_count, std::mutex& logits_mtx);
	int tridet_predict(ThreadSafeQueue<std::vector<float>>&, float, const std::string&,
	                   std::vector<float>& logits_accum, int& logits_count, std::mutex& logits_mtx,
	                   std::shared_ptr<Tridet> tridet_instance);
	void set_task_status(const std::string& camera_id, const std::string& status, const std::string& msg = "");
	json get_task_status_json(const std::string& camera_id);
	bool report_exists(const std::string& camera_id) const;

	// WebRTC 相关
	struct WebRTCSession {
		std::shared_ptr<rtc::PeerConnection> pc;
		std::shared_ptr<rtc::Track> track;
		std::shared_ptr<rtc::H264RtpPacketizer> packetizer; // 持有打包器链的生命周期
		std::function<void(const rtc::byte*, size_t, int64_t)> send_video;
		std::atomic<bool> track_ready{false}; // onOpen 触发后置 true，onClosed 置 false
	};
	// camera_id -> sessions
	std::mutex sessions_mtx;
	std::map<std::string, std::vector<std::shared_ptr<WebRTCSession>>> webrtc_sessions;

	httplib::Server server_;
	std::atomic<bool> enable_display_{ false }; // 是否启用本地 OpenCV 窗口预览
	ThreadSafeDict<std::string, bool> camera_thread_manager;
	ThreadSafeDict<std::string, bool> keyframe_requests; // 记录各相机是否需要立即产出关键帧
	ThreadSafeDict<int, int> GPU_ID_manager;

	// I3D 特征提取模型（全局共享，只读推理，线程安全）
	std::shared_ptr<I3D> i3d_model_;
	std::shared_ptr<YoloPoseDetector> yolo_model_;
	std::mutex yolo_mtx_;

	// Tridet 时序动作检测模型：每个 session 自建实例，不再共享
	// （移除 tridet_model_ 类成员，见 launch_camera / launch_local_video）

	// 伪在线预测结果：camera_id -> 最新 Run() 输出（仅供 Log 展示，非最终报告）
	std::mutex live_pred_mtx_;
	std::map<std::string, std::vector<ActionSegment>> live_predictions_;

	struct TaskStatus {
		std::string status = "unknown";
		std::string msg;
	};
	std::mutex task_status_mtx_;
	std::map<std::string, TaskStatus> task_status_;
};

