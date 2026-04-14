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

	// 解决 file:// 协议下的 WebRTC 安全限制：直接通过 http://localhost:8080 访问前端
	server_.Get("/", [this](const httplib::Request& req, httplib::Response& res) {
		std::vector<std::string> search_paths = {
			"client/index.html",
			"../client/index.html",
			"../../client/index.html",
			"../../../client/index.html"
		};
		
		std::ifstream ifs;
		for (const auto& path : search_paths) {
			ifs.open(path);
			if (ifs.is_open()) break;
		}

		if (ifs.is_open()) {
			std::stringstream ss;
			ss << ifs.rdbuf();
			res.set_content(ss.str(), "text/html; charset=utf-8");
		} else {
			res.status = 404;
			res.set_content("<h3>SkyEye Error: index.html not found in any search paths!</h3>", "text/html");
		}
	});
	
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
		// 显式绑定到所有本地接口，解决虚拟网卡导致的路径不可达问题
		config.bindAddress = "0.0.0.0";
		auto pc = std::make_shared<rtc::PeerConnection>(config);

		// session 在 Step3 填充完成后再注册到 webrtc_sessions
		auto session = std::make_shared<WebRTCSession>();
		session->pc = pc;


		// 正确的应答流程（libdatachannel）：
		// 1. 注册所有回调（含 onTrack）
		// 2. setRemoteDescription(offer)   → 库解析 offer 的 m-line，
		//    对每个 m-line 触发 onTrack 回调，返回可用的 Track
		// 3. 在 onTrack 得到的 Track 上挂载 RTP 打包器
		//    ★ 不要 addTrack()！那会新增 m-line，导致 answer 与 offer 的 m-line 数量不匹配
		// 4. setLocalDescription()         生成与 offer m-line 顺序一致的 answer
		// 5. gatherLocalCandidates()       触发 ICE 收集

		// === Step 1: 注册所有回调 ===
		pc->onStateChange([this, camera_id, session](rtc::PeerConnection::State state) {
			std::cout << "[WebRTC] Camera " << camera_id << " Connection State -> " << state << std::endl;
			if (state == rtc::PeerConnection::State::Connecting) {
				std::cout << "[WebRTC] Camera " << camera_id << " PeerConnection CONNECTING" << std::endl;
			}
			if (state == rtc::PeerConnection::State::Connected) {
				std::cout << "🔥🔥🔥 [WebRTC Success] P2P tunnel established for " << camera_id << "!" << std::endl;
			}
			if (state == rtc::PeerConnection::State::Closed || state == rtc::PeerConnection::State::Failed) {
				std::lock_guard<std::mutex> lock(sessions_mtx);
				auto& sessions = webrtc_sessions[camera_id];
				sessions.erase(std::remove(sessions.begin(), sessions.end(), session), sessions.end());
			}
		});

		pc->onLocalCandidate([camera_id](rtc::Candidate candidate) {
			std::cout << "[WebRTC] Local Candidate (" << camera_id << "): " << std::string(candidate) << std::endl;
		});

		auto gather_promise = std::make_shared<std::promise<std::string>>();
		auto gather_future  = gather_promise->get_future();
		auto gather_done    = std::make_shared<std::atomic<bool>>(false);
		pc->onGatheringStateChange([pc, gather_promise, gather_done](rtc::PeerConnection::GatheringState state) {
			std::cout << "[WebRTC] Gathering State -> " << static_cast<int>(state) << std::endl;
			if (state == rtc::PeerConnection::GatheringState::Complete) {
				if (!gather_done->exchange(true)) {
					if (auto desc = pc->localDescription()) {
						std::cout << "[WebRTC] Local Description Ready, length = " << std::string(*desc).size() << std::endl;
						gather_promise->set_value(std::string(*desc));
					}
				}
			}
		});

		// === Step 2: 注册 onTrack 并解析 offer ===
		// 浏览器 offer 中 m=video 为 recvonly，setRemoteDescription 后
		// libdatachannel 会为该 m-line 创建一个 Track 并通过 onTrack 传出。
		// 服务端在此 Track 上发送视频，无需 addTrack()（那会多出一个 m-line）。
		auto track_promise  = std::make_shared<std::promise<std::shared_ptr<rtc::Track>>>();
		auto track_future   = track_promise->get_future();
		auto track_received = std::make_shared<std::atomic<bool>>(false);

		pc->onTrack([this, camera_id, track_promise, track_received](std::shared_ptr<rtc::Track> track) {
			std::cout << "[WebRTC] onTrack fired for camera " << camera_id
			          << ", mid=" << track->mid() << std::endl;

			if (!track_received->exchange(true)) {
				track_promise->set_value(track);
			}
		});

		try {
			pc->setRemoteDescription(rtc::Description(sdp, rtc::Description::Type::Offer));
		} catch (const std::exception& e) {
			res.status = 500;
			res.set_content(std::string("SDP setRemoteDescription Error: ") + e.what(), "text/plain");
			return;
		}

		// 等待 onTrack 传出 Track（通常在 setRemoteDescription 内同步触发）
		std::shared_ptr<rtc::Track> local_track;
		if (track_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready) {
			local_track = track_future.get();
			std::cout << "[WebRTC] Got track from onTrack, mid=" << local_track->mid() << std::endl;
		} else {
			std::cerr << "[WebRTC] ERROR: onTrack not fired after setRemoteDescription!" << std::endl;
			res.status = 500;
			res.set_content("onTrack not fired – cannot attach video sender", "text/plain");
			return;
		}

		// === Step 3: 在 onTrack 得到的 Track 上挂载 RTP 打包器 ===
		local_track->onOpen([this, camera_id]() {
			std::cout << "🟢 [WebRTC Track Open] Camera " << camera_id << " is now ready for streaming!" << std::endl;
			// 关键修复：Track 打开时立即请求一个关键帧，确保新接入的用户能立即看到画面
			this->keyframe_requests.set(camera_id, true);
		});
		local_track->onClosed([camera_id]() {
			std::cout << "[WebRTC] Track Closed for camera " << camera_id << std::endl;
		});

		constexpr uint32_t SSRC = 42;
		constexpr uint8_t TARGET_PT = 103; // 使用浏览器已声明支持的 H264 PT
		auto rtpConfig = std::make_shared<rtc::RtpPacketizationConfig>(
    		SSRC, "video", TARGET_PT, rtc::H264RtpPacketizer::ClockRate
		);

		// 打包器构造函数的第三个参数是 MTU 大小（1200），显式限制以防公网丢包
		auto packetizer    = std::make_shared<rtc::H264RtpPacketizer>(
			rtc::H264RtpPacketizer::Separator::StartSequence, rtpConfig, 1200
		);

		auto srReporter    = std::make_shared<rtc::RtcpSrReporter>(rtpConfig);
		auto nackResponder = std::make_shared<rtc::RtcpNackResponder>();
		
		// 使用 PliHandler 捕获浏览器的关键帧请求（PLI 信号）
		auto pliHandler = std::make_shared<rtc::PliHandler>([this, camera_id]() {
			std::cout << "📢 [WebRTC PLI] Receiver requested replacement keyframe for " << camera_id << std::endl;
			this->keyframe_requests.set(camera_id, true);
		});

		packetizer->addToChain(srReporter);
		srReporter->addToChain(nackResponder);
		nackResponder->addToChain(pliHandler); // 挂载到处理链条
		
		local_track->setMediaHandler(packetizer);

		// 填充 session
		session->track      = local_track;
		session->packetizer = packetizer;
		constexpr int ENCODE_FPS = 15;
		session->send_video = [local_track, rtpConfig, camera_id, ENCODE_FPS](const rtc::byte* data, size_t size, int64_t pts) {
			try {
				// 每一帧的所有 NALU 共享相同的 90kHz RTP 时间戳
				rtpConfig->timestamp = 160000 + static_cast<uint32_t>(pts * (90000 / ENCODE_FPS));

				// 关键修复：手动拆解 Annex-B 里的多个 NALU。
				// FFmpeg 经常将 SPS/PPS/IDR 粘在一起，导致打包器误将其识别为一个巨大的 SPS。
				const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data);
				const uint8_t* end = ptr + size;
				const uint8_t* nalu_start = nullptr;

				while (ptr < end) {
					// 寻找 NALU 起始码 (00 00 01 或 00 00 00 01)
					bool is_start_code = false;
					int skip = 0;
					if (ptr + 4 <= end && ptr[0] == 0 && ptr[1] == 0 && ptr[2] == 0 && ptr[3] == 1) {
						is_start_code = true; skip = 4;
					} else if (ptr + 3 <= end && ptr[0] == 0 && ptr[1] == 0 && ptr[2] == 1) {
						is_start_code = true; skip = 3;
					}

					if (is_start_code) {
						if (nalu_start) {
							// 发送上一个找到的完整 NALU，显式转换回 rtc::byte* 指针以匹配 rtc::binary 的迭代器构造
							local_track->send(rtc::binary(reinterpret_cast<const rtc::byte*>(nalu_start), reinterpret_cast<const rtc::byte*>(ptr)));
						}
						nalu_start = ptr;
						ptr += skip;
					} else {
						ptr++;
					}
				}
				
				// 发送 buffer 中的最后一个 NALU
				if (nalu_start) {
					// 调试日志：探测最后一个 NALU 类型
					uint8_t nalu_type = 0;
					if (nalu_start[2] == 1) nalu_type = nalu_start[3] & 0x1F;
					else if (nalu_start[3] == 1) nalu_type = nalu_start[4] & 0x1F;

					if (nalu_type == 7 || nalu_type == 8 || nalu_type == 5 || pts % 200 == 0) {
						std::cout << "🔍 [WebRTC Send] " << camera_id << " Type=" << (int)nalu_type 
						          << " Size=" << (end - nalu_start) << " TS=" << rtpConfig->timestamp << std::endl;
					}
					local_track->send(rtc::binary(reinterpret_cast<const rtc::byte*>(nalu_start), reinterpret_cast<const rtc::byte*>(end)));
				}
			} catch (const std::exception& e) {
				std::cerr << "🔴 [WebRTC Send Error] Camera " << camera_id << ": " << e.what() << std::endl;
			}
		};

		// === Step 4 & 5: 生成 answer 并收集 ICE 候选 ===
		try {
			std::cout << "--- [DEBUG] Browser Offer SDP ---\n" << sdp << "\n-------------------------------" << std::endl;
			pc->setLocalDescription();
			pc->gatherLocalCandidates();
		} catch (const std::exception& e) {
			res.status = 500;
			res.set_content(std::string("SDP setLocalDescription Error: ") + e.what(), "text/plain");
			return;
		}

		// 协商开启后注册会话
		{
			std::lock_guard<std::mutex> lock(sessions_mtx);
			webrtc_sessions[camera_id].push_back(session);
		}

		// 等待 ICE Gathering 完成...
		if (gather_future.wait_for(std::chrono::seconds(10)) == std::future_status::ready) {
			std::string answer_sdp = gather_future.get();

			// 增强型 SDP 处理 (PT 103 协商策略)
			{
				std::string& sdp = answer_sdp;
				// 1. 修正 setup 角色为 active (RFC 标准 Answer 必须返回明确角色)
				size_t setup_pos = sdp.find("a=setup:actpass");
				if (setup_pos != std::string::npos) sdp.replace(setup_pos, 15, "a=setup:active");

				// 2. 强制确保 H.264 的关键参数对齐（注意严格使用 \r\n）
				// packetization-mode=1 必须启用，否则浏览器无法重组 FU-A 分片包
				std::string h264_fmtp = "a=fmtp:103 packetization-mode=1;profile-level-id=42c01f\r\n";
				size_t fmtp_pos = sdp.find("a=fmtp:103");
				if (fmtp_pos != std::string::npos) {
					size_t line_end = sdp.find("\n", fmtp_pos);
					sdp.replace(fmtp_pos, line_end - fmtp_pos + 1, h264_fmtp);
				} else {
					size_t vpos = sdp.find("m=video");
					if (vpos != std::string::npos) {
						size_t nl = sdp.find("\n", vpos);
						sdp.insert(nl + 1, h264_fmtp);
					}
				}

				// 3. 注入 SSRC 声明 (确保 RTP 发包器 SSRC=42 与 Track 强关联)
				std::string ssrc_info = "a=ssrc:42 cname:skyeye-video\r\n"
				                        "a=ssrc:42 msid:stream1 track1\r\n"
				                        "a=ssrc:42 label:track1\r\n";
				size_t video_m = sdp.find("m=video");
				if (video_m != std::string::npos) {
					size_t next_m = sdp.find("m=", video_m + 8);
					if (next_m == std::string::npos) next_m = sdp.size();
					sdp.insert(next_m, ssrc_info);
				}
			}

			json response;
			response["type"] = "answer";
			response["sdp"]  = answer_sdp;
			std::cout << "--- [DEBUG] Server Answer SDP ---\n" << answer_sdp << "\n-------------------------------" << std::endl;
			std::cout << "[WebRTC] ICE Gathering complete. Sending Answer for " << camera_id << std::endl;
			res.set_content(response.dump(), "text/plain");
		} else {
			std::cerr << "[WebRTC] ICE Gathering TIMEOUT for " << camera_id << std::endl;
			res.status = 500;
			res.set_content("ICE Gathering Timeout", "text/plain");
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
	// ─────────────────────────────────────────────────────────────────────────
	// live() 架构说明：
	//   display_queue 同时被两条消费链路消费：
	//   1. 本线程（display 线程）：仅负责 cv::imshow，保证本地预览帧率流畅。
	//   2. encode 线程（内部新建）：专门做 H264 编码 + WebRTC RTP 发送。
	//
	// 关键设计：
	//   - encode 线程有独立的 encode_queue（最大缓存 2 帧），满时丢弃最旧帧，
	//     确保编码线程不会落后太多，也不会因编码慢而拖垮显示线程。
	//   - display 线程和 encode 线程完全解耦，互不阻塞。
	// ─────────────────────────────────────────────────────────────────────────

	// encode_queue 容量为 2：编码跟不上时丢弃旧帧，避免延迟积累
	ThreadSafeQueue<cv::Mat> encode_queue;

	// ── 编码线程 ──────────────────────────────────────────────────────────────
	std::thread encode_thread([this, &encode_queue, &camera_id]() {
		H264Encoder encoder;
		bool encoder_init = false;
		constexpr int ENCODE_W   = 1280;
		constexpr int ENCODE_H   = 720;
		constexpr int ENCODE_FPS = 15;

		cv::Mat enc_frame;
		while (encode_queue.wait_and_pop(enc_frame)) {
			if (enc_frame.empty()) continue;

			// 快照当前 sessions，避免持锁期间调用 send_video（防死锁）
			std::vector<std::shared_ptr<WebRTCSession>> sessions_snapshot;
			{
				std::lock_guard<std::mutex> lock(sessions_mtx);
				auto it = webrtc_sessions.find(camera_id);
				if (it != webrtc_sessions.end()) {
					sessions_snapshot = it->second;
				}
			}

			// 无 WebRTC 客户端时不编码，节省 CPU
			if (sessions_snapshot.empty()) continue;

			// 懒初始化编码器（首次有客户端接入才启动）
			if (!encoder_init) {
				encoder_init = encoder.Init(ENCODE_W, ENCODE_H, ENCODE_FPS, 2000000);
				if (!encoder_init) {
					std::cerr << "[EncodeThread] H264Encoder Init failed!" << std::endl;
					continue;
				}
				std::cout << "[EncodeThread] H264Encoder initialized for " << camera_id << std::endl;
			}

			// 缩放 + 编码
			cv::Mat small_frame;
			cv::resize(enc_frame, small_frame, cv::Size(ENCODE_W, ENCODE_H));

			// 检查是否有 PLI 请求，强制产生一个关键帧
			if (this->keyframe_requests.has(camera_id) && this->keyframe_requests.get(camera_id)) {
				encoder.ForceKeyframe();
				this->keyframe_requests.set(camera_id, false); // 重置标志
			}

			encoder.Encode(small_frame, [&sessions_snapshot](const uint8_t* data, size_t size, int64_t pts) {
				static std::atomic<int> packet_idx{ 0 };
				int idx = packet_idx.fetch_add(1);
				// 前 8 个 packet 打印头部字节，用于确认 Annex-B 格式
				if (idx < 8) {
					std::ostringstream oss;
					oss << "[H264] pkt#" << idx << " pts=" << pts << " size=" << size << " head=";
					for (size_t i = 0; i < std::min<size_t>(size, 8); ++i) {
						oss << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]) << ' ';
					}
					std::cout << oss.str() << std::endl;
				}
				for (auto& session : sessions_snapshot) {
					if (session->track && session->send_video) {
						try {
							session->send_video(reinterpret_cast<const rtc::byte*>(data), size, pts);
							if (idx < 4) {
								std::cout << "[WebRTC] send_video OK pkt#" << idx << " pts=" << pts << std::endl;
							}
						}
						catch (const std::exception& e) {
							if (idx < 8) std::cerr << "[WebRTC] send_video exception pkt#" << idx << ": " << e.what() << std::endl;
						}
						catch (...) {
							if (idx < 8) std::cerr << "[WebRTC] send_video unknown exception pkt#" << idx << std::endl;
						}
					}
				}
				});
// 移除过时的调试打印
		}
		std::cout << "[EncodeThread] " << camera_id << " Exit." << std::endl;
	});

	// ── 显示线程（当前线程）────────────────────────────────────────────────────
	cv::Mat frame;
	while (display_queue.wait_and_pop(frame)) {
		if (frame.empty()) continue;

		// 向编码队列投递帧副本；若队列已满（编码跟不上），丢弃最旧帧保持低延迟
		// try_push(val, 2) 表示队列超过 2 帧时不入队，防止延迟积累
		if (!encode_queue.try_push(frame.clone(), 2)) {
			cv::Mat dummy;
			encode_queue.try_pop(dummy);            // 弹出最旧帧
			encode_queue.try_push(frame.clone(), 2); // 压入最新帧
		}

		// 本地预览：不受编码耗时影响，帧率完全跟随 RTSP 源
		cv::imshow("Pilot_" + camera_id, frame);
		if (cv::waitKey(1) == 27) { // ESC 退出
			break;
		}
	}

	// display_queue 耗尽后停止 encode 线程
	encode_queue.stop();
	if (encode_thread.joinable()) encode_thread.join();

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