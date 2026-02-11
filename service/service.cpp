#include "service.hpp"
#include "config.hpp"
#include "utils/WebServerUtils.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>

using json = nlohmann::json;

int PilotWebServer::boot() {
	std::cout << "PilotWebServer Init..." << std::endl;
	init();
	set_server_logger();
	set_camera_interface();
	server_.listen("0.0.0.0", 8080);

	return 0;
}

int PilotWebServer::init() {
	// TODO:
	return 0;
}

int PilotWebServer::set_server_logger() {
	server_.set_logger([](const httplib::Request& req, const httplib::Response& res) {
		std::cout << WebServerUtils::get_timestamp() << "[" << req.remote_addr << "]"
			<< req.method << " " << req.path
			<< "->" << res.status << std::endl;
	});
	return 0;
}

int PilotWebServer::set_camera_interface() {
	server_.Post("/launch_camera", [this](const httplib::Request& req, httplib::Response& res) {
		if (WebServerUtils::check_head(req, res)) return;
		std::vector<std::string> meta_fields = { "camera_id", "video_url" };
		if (WebServerUtils::check_field(req, res, meta_fields)) return;
		json request = json::parse(req.body);
		std::string camera_id = request["camera_id"];
		std::string rtsp_url = request["video_url"];


		json response;
		if (!camera_thread_manager.has(camera_id) || camera_thread_manager.get(camera_id) == false) {
			camera_thread_manager.set(camera_id, true);
			std::thread([=]() {
				launch_camera(camera_id, rtsp_url);
			}).detach();
			response["code"] = 200;
			response["msg"] = "Success to launch camera:" + camera_id;
		}
		else {
			response["code"] = 200;
			response["msg"] = "Camera " + camera_id + "is already launched";
		}
		res.status = 200;
		res.set_content(response.dump(), "application/json");
	});

	server_.Post("/offline_camera", [this](const httplib::Request& req, httplib::Response& res) {
		if (WebServerUtils::check_head(req, res)) return;
		std::vector<std::string> meta_fields = { "camera_id" };
		if (WebServerUtils::check_field(req, res, meta_fields)) return;
		json request = json::parse(req.body);
		std::string camera_id = request["camera_id"];

		json response;
		if (camera_thread_manager.has(camera_id) &&
			camera_thread_manager.get(camera_id) == true) {

			camera_thread_manager.set(camera_id, false);
			response["code"] = 200;
			response["msg"] = "Success to offline camera " + camera_id;
		}
		else {
			response["code"] = 200;
			response["msg"] = "There is no online camera " + camera_id;
		}
		res.status = 200;
		res.set_content(response.dump(), "application/json");
	});

	return 0;
}

int PilotWebServer::launch_camera(const std::string& camera_id, const std::string& rtsp_url) {
	cv::VideoCapture cap(rtsp_url,cv::CAP_FFMPEG);
	if (!cap.isOpened()) {
		std::cerr << "Failed to open rtsp stream:" << rtsp_url << std::endl;
		return -1;
	}
	int width_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int height_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	int fps_ = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
	if (fps_ <= 0 || fps_ > 60) fps_ = 15;
	cap.release();
	
	std::ostringstream cmd;
	cmd << "ffmpeg -loglevel error "
		<< "-rtsp_transport tcp "
		<< "-i " << rtsp_url
		<< " -f rawvideo -pix_fmt bgr24 "
		<< " -s " << width_ << "x" << height_
		<< " -r " << fps_
		<< " pipe:1";

#ifdef _WIN32
	FILE* pipe_in = _popen(cmd.str().c_str(), "rb");
#else
	FILE* pipe_in = popen(cmd.str().c_str(), "r");
#endif
	
	if (!pipe_in) {
		std::cerr << "Failed to open FFmpeg pipe" << std::endl;
		return -1;
	}

	const size_t frame_size = static_cast<size_t>(width_) * height_ * 3;
	uchar* buffer = new uchar[frame_size];
	cv::Mat input_frame(height_, width_, CV_8UC3);
	std::cout << "Start processing the stream: " << width_ << "x" << height_ << " @ " << fps_ << "fps" << std::endl;


	while (camera_thread_manager.get(camera_id)) {
		size_t bytes_read = fread(buffer,1,frame_size,pipe_in);
		if (bytes_read != frame_size) {
			std::cerr << "Frames read have error.Video Processing has break." << std::endl;
			break;
		}

		input_frame.data = buffer;

		if (input_frame.empty()) {
			std::cout << "[Warning] Frame is empty! Skipping..." << std::endl;
			continue;
		}
		// 调试使用，上线去除
		cv::imshow("Pilot Training Real-time", input_frame);
		if (cv::waitKey(1) == 27) break;
	}

	delete[] buffer;
#ifdef _WIN32
	if (pipe_in) _pclose(pipe_in);
#else
	if (pipe_in) pclose(pipe_in);
#endif
	std::cout << "Stream process end" << std::endl;

	return 0;
}

int PilotWebServer::extract_features(const std::string& camera_id, const std::string& rtsp_url) {
	// 视频流处理
	cv::VideoCapture cap(rtsp_url, cv::CAP_FFMPEG);
	if (!cap.isOpened()) {
		std::cerr << "Failed to open rtsp stream:" << rtsp_url << std::endl;
		return -1;
	}
	int width_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int height_ = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	int fps_ = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
	if (fps_ <= 0 || fps_ > 60) fps_ = 15;
	cap.release();

	std::ostringstream cmd;
	cmd << "ffmpeg -loglevel error "
		<< "-rtsp_transport tcp "
		<< "-i " << rtsp_url
		<< " -f rawvideo -pix_fmt bgr24 "
		<< " -s " << width_ << "x" << height_
		<< " -r " << fps_
		<< " pipe:1";

#ifdef _WIN32
	FILE* pipe_in = _popen(cmd.str().c_str(), "rb");
#else
	FILE* pipe_in = popen(cmd.str().c_str(), "r");
#endif
	if (!pipe_in) {
		std::cerr << "Failed to open FFmpeg pipe" << std::endl;
		return -1;
	}

	// 创建缓冲区，为i3d提供输入准备
	const size_t frame_size = static_cast<size_t>(width_) * height_ * 3;
	uchar* buffer = new uchar[frame_size];
	cv::Mat input_frame(height_, width_, CV_8UC3);
	std::vector<cv::Mat> window_frames;
	int frame_count = 0;

	while(camera_thread_manager.get(camera_id)){
		size_t bytes_read = fread(buffer, 1, frame_size, pipe_in);
		if (bytes_read != frame_size){
			std::cerr << "Frames read have error.Video Processing has break." << std::endl;
			break;
		}

		input_frame.data = buffer;
	}
}

int PilotWebServer::distribute_GPU(int occupy, int design) {
	if (design >= 0) {
		if (GPU_ID_manager.has(design)) {
			int ori_occupy = GPU_ID_manager.get(design);
			GPU_ID_manager.set(design, ori_occupy + occupy);
		}
		return design;
	}
	int res = 0;
	int min_occupy = 999;
	for (int i = MIN_GPU_ID; i <= MAX_GPU_ID; ++i) {
		int oc = GPU_ID_manager.get(i);
		if (oc < min_occupy) {
			min_occupy = oc;
			res = i;
		}
	}
	GPU_ID_manager.set(res, min_occupy + occupy);
	return res;
}

int PilotWebServer::cancel_GPU(int gpu_id, int occupy) {
	if (!GPU_ID_manager.has(gpu_id)) return -1;
	int ori_occupy = GPU_ID_manager.get(gpu_id);
	if (ori_occupy < occupy) return -1;
	GPU_ID_manager.set(gpu_id, ori_occupy - occupy);
	return 0;
}