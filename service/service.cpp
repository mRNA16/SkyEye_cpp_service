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

	// Load Tridet onnx
	std::cout << "Loading Tridet Model from: " << TRIDET_MODEL_PATH << std::endl;
	tridet_model_ = std::make_shared<Tridet>();
	if (tridet_model_->Init(TRIDET_MODEL_PATH, 0, NUM_CLASSES) != 0) {
		std::cerr << "Failed to initialize Tridet model!" << std::endl;
		return -1;
	}
	std::cout << "Tridet Model initialized successfully." << std::endl;

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

	server_.Post("/get_report", [this](const httplib::Request& req, httplib::Response& res) {
		if (WebServerUtils::check_head(req, res)) return;
		std::vector<std::string> meta_fields = { "camera_id" };
		if (WebServerUtils::check_field(req, res, meta_fields)) return;
		json request = json::parse(req.body);
		std::string camera_id = request["camera_id"];

		json response;
		std::string report_file = "report_" + camera_id + ".json";
		std::ifstream ifs(report_file);
		if (ifs.is_open()) {
			json report_data;
			ifs >> report_data;
			response["code"] = 200;
			response["data"] = report_data;
		} else {
			response["code"] = 404;
			response["msg"] = "Report not found or the live stream is still ongoing.";
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
	// 堆开辟200帧空间，溢出以二进制文件存储至磁盘
	std::string temp_algo_buffer = "algo_buffer_camera_" + camera_id + ".bin";
	HybridVideoQueue frame_queue(1000, temp_algo_buffer, 448, 448, CV_8UC3);
	ThreadSafeQueue<std::vector<float>> feature_queue;

	// 启动消费者线程
	std::thread thread_live(&PilotWebServer::live, this, std::ref(display_queue));
	std::thread thread_extract(&PilotWebServer::extract_features, this, std::ref(frame_queue), std::ref(feature_queue));
	std::thread thread_predict(&PilotWebServer::tridet_predict, this, std::ref(feature_queue), static_cast<float>(fps_), camera_id);

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
		
		// 调整大小至 448x448 以匹配模型输入并节省内存/磁盘开销
		cv::Mat resized_frame;
		cv::resize(input_frame, resized_frame, cv::Size(448, 448));
		frame_queue.push(resized_frame);
	}

	std::cout << "Waiting for consumers to finish..." << std::endl;
	
	// 1.停止直播和i3d入口视频帧队列
	display_queue.stop();
	frame_queue.stop();
	
	// 2. 等待直播结束，这个过程会很快
	if (thread_live.joinable()) thread_live.join();
	// 3. 等待特征提取线程结束，这个过程慢，磁盘中挤压视频帧很多
	if (thread_extract.joinable()) thread_extract.join();
	
	// 4. 特征提取宣告结束，Tridet 消耗完所有的特征后结束
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

// 定义对应的真实语义：从 0 背景，再到 1-16 代表具体的控制台动作
const std::vector<std::string> ACTION_NAMES = {
	"Background", "Yoke", "ThrottleLever", "LandingGear", "SpeedBrakes", "Flap",
	"Computer", "TrimWheel", "EngineSwitch", "EngineModeSel", "RudTrim",
	"EFISControl", "SpeedSel", "HeadingSel", "AltitudeSel", "VerticalSpeedSel", "AutoPilot"
};

int PilotWebServer::tridet_predict(ThreadSafeQueue<std::vector<float>>& feature_queue, float fps, const std::string& camera_id) {
	// Tridet消费者处理
	std::vector<float> features;
	std::vector<ActionSegment> global_segments;

	while (feature_queue.wait_and_pop(features)) {
		if (!features.empty() && this->tridet_model_) {
			std::vector<ActionSegment> actions = tridet_model_->Run(features, static_cast<float>(fps), CHUNK_SIZE);
			for (const auto& action : actions) {
				// 收集每一段由滑窗内预测出的有效结果用于最终全时段去重
				if(action.score > 0.05f) {
					global_segments.push_back(action);
				}
			}
		}
	}
	
	std::cout << "[Tridet Thread] Stream finished. Generating final report..." << std::endl;
	
	// 在全部结束后使用全局 1D-IoU NMS 清理同一动作被随着不同窗口预测产生的多重叠碎片
	std::sort(global_segments.begin(), global_segments.end(), [](const ActionSegment& a, const ActionSegment& b){
		return a.score > b.score;
	});
	std::vector<ActionSegment> nms_results;
	for (auto& seg : global_segments) {
		bool drop = false;
		for (auto& res : nms_results) {
			if (seg.label == res.label) {
				float inter_start = std::max(seg.start_time, res.start_time);
				float inter_end = std::min(seg.end_time, res.end_time);
				float inter = std::max(0.0f, inter_end - inter_start);
				float uni = (seg.end_time - seg.start_time) + (res.end_time - res.start_time) - inter;
				if (uni > 0 && (inter / uni) > 0.2f) { // IoU 阈值控制融合
					drop = true; break;
				}
			}
		}
		if (!drop) nms_results.push_back(seg);
	}
	
	// 并行保存为结果并允许被提取
	std::sort(nms_results.begin(), nms_results.end(), [](const ActionSegment& a, const ActionSegment& b){ return a.start_time < b.start_time; });
	json report;
	report["camera_id"] = camera_id;
	report["summary"] = "Action Detection Report";
	report["actions"] = json::array();
	for (auto& seg : nms_results) {
		json item;
		item["start"] = seg.start_time;
		item["end"] = seg.end_time;
		item["score"] = seg.score;
		item["action"] = (seg.label >= 0 && seg.label < ACTION_NAMES.size()) ? ACTION_NAMES[seg.label] : "Action " + std::to_string(seg.label);
		report["actions"].push_back(item);
	}
	std::ofstream ofs("report_" + camera_id + ".json");
	ofs << report.dump(4);
	
	std::cout << "[Tridet Thread] Report saved to report_" << camera_id << ".json. Thread Exit." << std::endl;
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