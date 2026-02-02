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

using json = nlohmann::json;

class PilotWebServer {
public:
	int boot();
private:
	int init();
	int set_server_logger();
	int set_camera_interface();
	int distribute_GPU(int occupy, int design);
	int cancel_GPU(int gpu_id, int occupy);

protected:
	int launch_camera(const std::string& camera_id,const std::string& input_url);

	httplib::Server server_;
	ThreadSafeDict<std::string, bool> camera_thread_manager;
	ThreadSafeDict<int, int> GPU_ID_manager;
};

