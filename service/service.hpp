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

using json = nlohmann::json;

class PilotWebServer {
public:
	int boot();
private:
	int init();
	int set_server_logger();
	int set_camera_interface();

	httplib::Server server_;
};

