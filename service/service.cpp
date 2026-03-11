#include "service.hpp"
#include "config.hpp"
#include "utils/WebServerUtils.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>

using json = nlohmann::json;

int PilotWebServer::boot() {
	std::cout << "PilotWebServer Init..." << std::endl;
	loadModels();
	set_server_logger();
	set_camera_interface();
	server_.listen("0.0.0.0", 8080);

	return 0;
}

int PilotWebServer::loadModels() {
	// Load i3d onnx
	std::cout << "Loading I3D Model from: " << I3D_MODEL_PATH << std::endl;
	i3d_model_ = std::make_shared<I3D>();
	if (i3d_model_->Init(I3D_MODEL_PATH, 0) != 0) {
		std::cerr << "Failed to initialize I3D model!" << std::endl;
		return -1;
	}
	std::cout << "I3D Model initialized successfully." << std::endl;

	// Load actionformer onnx
	std::cout << "Loading ActionFormer Model from: " << ACTIONFORMER_MODEL_PATH << std::endl;
	actionformer_model_ = std::make_shared<ActionFormer>();
	if (actionformer_model_->Init(ACTIONFORMER_MODEL_PATH, 0, NUM_CLASSES) != 0) {
		std::cerr << "Failed to initialize ActionFormer model!" << std::endl;
		return -1;
	}
	std::cout << "ActionFormer Model initialized successfully." << std::endl;

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

	const size_t frame_size = static_cast<size_t>(width_) * height_ * 3;
	uchar* buffer = new uchar[frame_size];
	std::cout << "Start processing the stream: " << width_ << "x" << height_ << " @ " << fps_ << "fps" << std::endl;

	ThreadSafeQueue<cv::Mat> display_queue;
	// 混合模式队列：内存中只允许堆积 200 帧，超过后自动写向系统硬盘缓存文件
	std::string temp_algo_buffer = "algo_buffer_camera_" + camera_id + ".bin";
	HybridVideoQueue frame_queue(200, temp_algo_buffer, width_, height_, CV_8UC3);
	ThreadSafeQueue<std::vector<float>> feature_queue;

	// 启动消费者线程
	std::thread thread_live(&PilotWebServer::live, this, std::ref(display_queue));
	std::thread thread_extract(&PilotWebServer::extract_features, this, std::ref(frame_queue), std::ref(feature_queue));
	std::thread thread_predict(&PilotWebServer::actionformer_predict, this, std::ref(feature_queue), static_cast<float>(fps_));

	// 生产者主循环
	while (camera_thread_manager.get(camera_id)) {
		size_t total_bytes_read = 0;
		while (total_bytes_read < frame_size) {
			size_t bytes_read = fread(buffer + total_bytes_read, 1, frame_size - total_bytes_read, pipe_in);
			if (bytes_read == 0) {
				std::cerr << "Can't read new byte to build a new frame!" << std::endl;
				break; 
			}
			total_bytes_read += bytes_read;
		}

		if (total_bytes_read != frame_size) {
			if (feof(pipe_in)) {
				std::cerr << "End of ffmpeg stream reached." << std::endl;
			} else {
				std::cerr << "Error reading from ffmpeg pipe." << std::endl;
			}
			break;
		}

		cv::Mat input_frame(height_, width_, CV_8UC3, buffer);
		if (input_frame.empty()) {
			std::cout << "[Warning] Frame is empty! Skipping..." << std::endl;
			continue;
		}
		
		display_queue.push(input_frame.clone());
		frame_queue.push(input_frame.clone());
	}

	std::cout << "Waiting for consumers to finish..." << std::endl;
	
	// 1.停止直播和i3d入口视频帧队列
	display_queue.stop();
	frame_queue.stop();
	
	// 2. 等待直播结束，这个过程会很快
	if (thread_live.joinable()) thread_live.join();
	// 3. 等待特征提取线程结束，这个过程慢，磁盘中挤压视频帧很多
	if (thread_extract.joinable()) thread_extract.join();
	
	// 4. 特征提取宣告结束，ActionFormer消耗完所有的特征后结束
	feature_queue.stop();
	if (thread_predict.joinable()) thread_predict.join();

	delete[] buffer;
#ifdef _WIN32
	if (pipe_in) _pclose(pipe_in);
#else
	if (pipe_in) pclose(pipe_in);
#endif
	std::cout << "Stream process end" << std::endl;

	return 0;
}

int PilotWebServer::live(ThreadSafeQueue<cv::Mat>& display_queue) {
	// 直播消费者处理
	cv::Mat frame;
	while (display_queue.wait_and_pop(frame)) {
		if (frame.empty()) continue;
		cv::imshow("Pilot Training Real-time", frame);
		if (cv::waitKey(1) == 27) { // ESC
			break;
		}
	}
	cv::destroyWindow("Pilot Training Real-time");
	std::cout << "[Display Thread] Exit." << std::endl;
	return 0;
}

int PilotWebServer::extract_features(HybridVideoQueue& frame_queue,ThreadSafeQueue<std::vector<float>>& feature_queue) {
	// i3d消费者处理
	cv::Mat frame;
	std::deque<cv::Mat> local_window_frames;

	while (frame_queue.wait_and_pop(frame)) {
		if (frame.empty()) continue;
		local_window_frames.push_back(frame);

		if (local_window_frames.size() >= CHUNK_SIZE) {
			std::vector<cv::Mat> infer_frames(local_window_frames.begin(), local_window_frames.begin() + CHUNK_SIZE);
			std::vector<float> features = this->i3d_model_->Run(infer_frames);
			feature_queue.push(features);
			for (int i = 0; i < 4; ++i) local_window_frames.pop_front();
		}
	}
	std::cout << "[FeatureExtract] Frame queue computed completely. End." << std::endl;
	return 0;
}

int PilotWebServer::actionformer_predict(ThreadSafeQueue<std::vector<float>>& feature_queue, float fps) {
	// ActionFormer消费者处理
	std::vector<float> features;

	while (feature_queue.wait_and_pop(features)) {
		if (!features.empty() && this->actionformer_model_) {
			std::vector<ActionSegment> actions = actionformer_model_->Run(features, static_cast<float>(fps), CHUNK_SIZE);
			std::string predicted_action = "Background";
			float max_score = 0.0f;
			for (const auto& action : actions) {
				if (action.score > max_score) {
					max_score = action.score;
					predicted_action = "[" + std::to_string(action.end_time - action.start_time).substr(0,2) + "]"
						 + "Action " + std::to_string(action.label) + " (" + std::to_string(action.score).substr(0, 4) + ")";
				}
			}
			std::cout << "[ActionFormer Thread] Predicted: " << predicted_action
				<< " | Remaining feature queue size: " << feature_queue.size() << std::endl;
		}
	}
	std::cout << "[ActionFormer Thread] Exit." << std::endl;
	return 0;
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