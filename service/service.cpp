#include "service.hpp"
#include "config.hpp"
#include "utils/WebServerUtils.hpp"
#include "utils/H264Encoder.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include <future>

using json = nlohmann::json;

int PilotWebServer::boot() {
	std::cout << "PilotWebServer Init..." << std::endl;
	loadModels();
	set_server_logger();
	
	// 分别显式注册三个路径的 OPTIONS 预检请求（解决部分 httplib 版本正则通配失败的跨域问题）
	auto cors_options_handler = [](const httplib::Request& req, httplib::Response& res) {
		res.set_header("Access-Control-Allow-Origin", "*");
		res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
		res.set_header("Access-Control-Allow-Headers", "Content-Type");
		res.status = 200;
	};
	server_.Options("/launch_camera", cors_options_handler);
	server_.Options("/offline_camera", cors_options_handler);
	server_.Options("/get_report", cors_options_handler);
	server_.Options("/webrtc/offer", cors_options_handler);
	
	// 对所有成功进入的常规请求（POST）最后带上跨域许可凭证
	server_.set_post_routing_handler([](const auto& req, auto& res) {
		res.set_header("Access-Control-Allow-Origin", "*");
		res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
		res.set_header("Access-Control-Allow-Headers", "Content-Type");
	});
	
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

	// WebRTC 信令端点 (Offer -> Answer)
	server_.Post("/webrtc/offer", [this](const httplib::Request& req, httplib::Response& res) {
		json data;
		try {
			data = json::parse(req.body);
		} catch (...) {
			res.status = 400;
			res.set_content("Invalid JSON", "text/plain");
			return;
		}

		std::string camera_id = data.value("camera_id", "");
		std::string sdp = data.value("sdp", "");
		if (camera_id.empty() || sdp.empty()) {
			res.status = 400;
			res.set_content("Missing camera_id or sdp", "text/plain");
			return;
		}

		rtc::Configuration config;
		auto pc = std::make_shared<rtc::PeerConnection>(config);
		
		// 使用 Description::Video 创建支持 H.264 的视频轨道
		rtc::Description::Video video("video", rtc::Description::Direction::SendOnly);
		video.addH264Codec(96); // 96 为 H.264 的常用 Payload Type
		auto local_track = pc->addTrack(video);

		auto session = std::make_shared<WebRTCSession>();
		session->pc = pc;
		// forward encoded NALUs to the local track
		session->send_video = [local_track](const rtc::byte* data, size_t size) {
			try {
				if (local_track) local_track->send(data, size);
			} catch (...) {}
		};

		{
			std::lock_guard<std::mutex> lock(sessions_mtx);
			webrtc_sessions[camera_id].push_back(session);
		}

		pc->onStateChange([this, camera_id, session](rtc::PeerConnection::State state) {
			std::cout << "[WebRTC] Camera " << camera_id << " State: " << state << std::endl;
			if (state == rtc::PeerConnection::State::Closed || state == rtc::PeerConnection::State::Failed) {
				std::lock_guard<std::mutex> lock(sessions_mtx);
				auto& sessions = webrtc_sessions[camera_id];
				sessions.erase(std::remove(sessions.begin(), sessions.end(), session), sessions.end());
			}
		});

		// 必须使用 shared_ptr 封装 promise，因为回调是异步的，原 local 变量生命周期不足
		auto answer_promise = std::make_shared<std::promise<std::string>>();
		auto answer_future = answer_promise->get_future();
		auto done = std::make_shared<std::atomic<bool>>(false);
		
		pc->onLocalDescription([answer_promise, done](rtc::Description description) {
			if (!done->exchange(true)) {
				answer_promise->set_value(std::string(description));
			}
		});

		try {
			pc->setRemoteDescription(rtc::Description(sdp, rtc::Description::Type::Offer));
		} catch (const std::exception& e) {
			res.status = 500;
			res.set_content(std::string("SDP Error: ") + e.what(), "text/plain");
			return;
		}

		if (answer_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready) {
			json response;
			response["type"] = "answer";
			response["sdp"] = answer_future.get();
			res.set_content(response.dump(), "application/json");
		} else {
			res.status = 500;
			res.set_content("WebRTC Signaling Timeout", "text/plain");
		}
	});

	return 0;
}

int PilotWebServer::launch_camera(const std::string& camera_id, const std::string& rtsp_url) {
	cv::VideoCapture cap(rtsp_url, cv::CAP_FFMPEG);
	if (!cap.isOpened()) {
		std::cerr << "Failed to open rtsp stream:" << rtsp_url << std::endl;
		camera_thread_manager.set(camera_id, false);
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
	std::thread thread_live(&PilotWebServer::live, this, std::ref(display_queue), camera_id);
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

int PilotWebServer::live(ThreadSafeQueue<cv::Mat>& display_queue, const std::string& camera_id) {
	// 直播消费者处理
	cv::Mat frame;
	H264Encoder encoder;
	bool encoder_init = false;

	while (display_queue.wait_and_pop(frame)) {
		if (frame.empty()) continue;

		// 网页 WebRTC 视频输出
		{
			std::lock_guard<std::mutex> lock(sessions_mtx);
			if (webrtc_sessions.count(camera_id) && !webrtc_sessions[camera_id].empty()) {
				if (!encoder_init) {
					encoder_init = encoder.Init(frame.cols, frame.rows, 15);
				}
				if (encoder_init) {
					encoder.Encode(frame, [this, &camera_id](const uint8_t* data, size_t size) {
						// 注意：此处回调是同步触发的，已经在外层 sessions_mtx 的锁保护下
						// 因此绝对不能再次通过 std::lock_guard 锁定同一个 mutex，否则会导致死锁或 system_error
						auto& active_sessions = this->webrtc_sessions[camera_id];
						int sent_count = 0;
						for (auto& session : active_sessions) {
							if (session->pc->state() == rtc::PeerConnection::State::Connected) {
								try {
									if (session->send_video) {
										session->send_video(reinterpret_cast<const rtc::byte*>(data), size);
										sent_count++;
									}
								} catch (...) {}
							}
						}
						// 采样日志输出，避免过快刷屏
						static int frame_cnt = 0;
						if (sent_count > 0 && frame_cnt++ % 60 == 0) {
							std::cout << "[WebRTC] Streaming to " << sent_count << " peers for " << camera_id << std::endl;
						}
					});
				}
			}
		}

		cv::imshow("Pilot_" + camera_id, frame);
		if (cv::waitKey(1) == 27) { // ESC
			break;
		}
	}
	cv::destroyWindow("Pilot_" + camera_id);
	std::cout << "[Display Thread] " << camera_id << " Exit." << std::endl;
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
	std::vector<std::vector<float>> all_features;

	while (feature_queue.wait_and_pop(features)) {
		if (!features.empty() && this->tridet_model_) {
			all_features.push_back(features);
		}
	}
	
	std::cout << "[Tridet Thread] Stream finished. Acquired total feature chunks: " << all_features.size() << ". Generating offline high performance report..." << std::endl;
	
    // 离线使用批处理和1/2重叠度做最后仅仅一次全推理
	std::vector<ActionSegment> global_segments = tridet_model_->RunOffline(all_features, static_cast<float>(fps), CHUNK_SIZE);
	
	// 在全部结束后使用全局 1D-IoU NMS 清理同一动作被随着不同窗口预测产生的多重叠碎片
	std::sort(global_segments.begin(), global_segments.end(), [](const ActionSegment& a, const ActionSegment& b){
		return a.start_time < b.start_time; 
	});
	
	// 保存为结果并允许被提
	json report;
	report["camera_id"] = camera_id;
	report["summary"] = "Action Detection Report";
	report["actions"] = json::array();
	for (auto& seg : global_segments) {
        if(seg.score <= 0.05f) continue;
		json item;
		item["start"] = seg.start_time;
		item["end"] = seg.end_time;
		item["score"] = seg.score;
		item["action"] = (seg.label >= 0 && seg.label < ACTION_NAMES.size()) ? ACTION_NAMES[seg.label] : "Action " + std::to_string(seg.label);
		report["actions"].push_back(item);
	}
	std::ofstream ofs("report_" + camera_id + ".json");
	ofs << report.dump(4);
	
	std::cout << "[Tridet Thread] Report saved to report_" << camera_id << ".json. Process Exited" << std::endl;
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