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
#include "../feature/actionformer.hpp"

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
	int live(ThreadSafeQueue<cv::Mat>&);
	int extract_features(HybridVideoQueue&,ThreadSafeQueue<std::vector<float>>&);
	int actionformer_predict(ThreadSafeQueue<std::vector<float>>&,float);

	httplib::Server server_;
	ThreadSafeDict<std::string, bool> camera_thread_manager;
	ThreadSafeDict<int, int> GPU_ID_manager;

	// I3D 特征提取模型
	std::shared_ptr<I3D> i3d_model_;

	// ActionFormer 时序动作检测模型
	std::shared_ptr<ActionFormer> actionformer_model_;
};

