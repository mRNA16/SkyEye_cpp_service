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
	std::cout << "Loading I3D Model from: " << I3D_MODEL_PATH << std::endl;
	i3d_model_ = std::make_shared<I3D>();
	if (i3d_model_->Init(I3D_MODEL_PATH, 0) != 0) {
		std::cerr << "Failed to initialize I3D model!" << std::endl;
		return -1;
	}
	std::cout << "I3D Model initialized successfully." << std::endl;

	std::cout << "Loading ActionFormer Model from: " << ACTIONFORMER_MODEL_PATH << std::endl;
	actionformer_model_ = std::make_shared<ActionFormer>();
	if (actionformer_model_->Init(ACTIONFORMER_MODEL_PATH, 0, 11) != 0) {
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
		size_t total_bytes_read = 0;
		while (total_bytes_read < frame_size) {
			size_t bytes_read = fread(buffer + total_bytes_read, 1, frame_size - total_bytes_read, pipe_in);
			if (bytes_read == 0) {
				break; // End of file or error
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
	
	std::cout << "Start feature extraction for camera: " << camera_id << std::endl;

	std::mutex mtx;
	std::vector<cv::Mat> shared_window_frames;
	std::string shared_current_action = "Background";
	bool stream_running = true;

	std::thread inference_thread([&]() {
		std::vector<cv::Mat> local_window_frames;
		int cnt = 0;
		int local_total_frames = 0;
		
		while (stream_running || !shared_window_frames.empty()) {
			bool has_enough_frames = false;
			{
				std::lock_guard<std::mutex> lock(mtx);
				if (shared_window_frames.size() >= CHUNK_SIZE) {
					local_window_frames.assign(shared_window_frames.begin(), shared_window_frames.begin() + CHUNK_SIZE);
					has_enough_frames = true;
					// Note: total_frames here roughly tracks what the oldest frame in the chunk corresponds to.
					local_total_frames += 4; 
				}
			}

			if (has_enough_frames) {
				std::vector<float> features = i3d_model_->Run(local_window_frames);
				if (!features.empty() && actionformer_model_) {
                    cnt++;
					std::vector<ActionSegment> actions = actionformer_model_->Run(features, static_cast<float>(fps_), CHUNK_SIZE);
					float current_time = static_cast<float>(local_total_frames + CHUNK_SIZE) / fps_; // time of boundary
					std::string predicted_action = "Background";
					float max_score = 0.0f;
					
					for (const auto& action : actions) {
						if (current_time >= action.start_time - 1.0f && current_time <= action.end_time + 1.0f) {
							if (action.score > max_score) {
								max_score = action.score;
								predicted_action = "Action " + std::to_string(action.label) + " (" + std::to_string(action.score).substr(0, 4) + ")";
							}
						}
					}
					
					std::cout << "[" << cnt << "] Extracted features: " << features[0] << " | predicted: " << predicted_action << std::endl;
					
					{
						std::lock_guard<std::mutex> lock(mtx);
						shared_current_action = predicted_action;
						
						// If the GPU is too slow, we drop frames to maintain real-time property.
						// Otherwise we slide by 4 frames (stride=4).
						int frames_to_erase = 4;
						if (shared_window_frames.size() > CHUNK_SIZE * 2) {
							std::cout << "drop happens!" << "\n";
							frames_to_erase = shared_window_frames.size() - CHUNK_SIZE; // Drop excessive frames
						}
						shared_window_frames.erase(shared_window_frames.begin(), shared_window_frames.begin() + frames_to_erase);
					}
				} else {
					// Fallback when inference yields empty
					std::lock_guard<std::mutex> lock(mtx);
					shared_window_frames.erase(shared_window_frames.begin(), shared_window_frames.begin() + 4);
				}
			} else {
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
		}
	});

	int total_frames = 0;
	while(camera_thread_manager.get(camera_id)){
		size_t total_bytes_read = 0;
		while (total_bytes_read < frame_size) {
			size_t bytes_read = fread(buffer + total_bytes_read, 1, frame_size - total_bytes_read, pipe_in);
			if (bytes_read == 0) {
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

		cv::Mat current_frame(height_, width_, CV_8UC3, buffer);
		
		std::string display_action;
		{
			std::lock_guard<std::mutex> lock(mtx);
			shared_window_frames.push_back(current_frame.clone());
			display_action = shared_current_action;
		}

		total_frames++;

		// Draw and display immediately (real-time 15FPS guaranteed)
		cv::putText(current_frame, display_action, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
		cv::imshow("Extracted Features Stream", current_frame);
		if (cv::waitKey(1) == 27) break;
	}

	stream_running = false;
	inference_thread.join();

	delete[] buffer;
#ifdef _WIN32
	if (pipe_in) _pclose(pipe_in);
#else
	if (pipe_in) pclose(pipe_in);
#endif
	std::cout << "Feature extraction end for camera: " << camera_id << std::endl;
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