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

// libdatachannel
#include <rtc/rtc.hpp>
#include <memory>
#include <map>
#include <mutex>
#include <future>

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
	int live(ThreadSafeQueue<cv::Mat>&, const std::string&);
	int extract_features(HybridVideoQueue&,ThreadSafeQueue<std::vector<float>>&);
	int tridet_predict(ThreadSafeQueue<std::vector<float>>&,float, const std::string&);

	// WebRTC 相关
	struct WebRTCSession {
		std::shared_ptr<rtc::PeerConnection> pc;
		std::shared_ptr<rtc::Track> track;
		std::shared_ptr<rtc::H264RtpPacketizer> packetizer; // 持有打包器链的生命周期
		std::function<void(const rtc::byte*, size_t)> send_video;
		std::atomic<bool> track_ready{false}; // onOpen 触发后置 true，onClosed 置 false
	};
	// camera_id -> sessions
	std::mutex sessions_mtx;
	std::map<std::string, std::vector<std::shared_ptr<WebRTCSession>>> webrtc_sessions;

	httplib::Server server_;
	ThreadSafeDict<std::string, bool> camera_thread_manager;
	ThreadSafeDict<int, int> GPU_ID_manager;

	// I3D 特征提取模型
	std::shared_ptr<I3D> i3d_model_;

	// Tridet 时序动作检测模型
	std::shared_ptr<Tridet> tridet_model_;
};

