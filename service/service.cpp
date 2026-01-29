#include "service.hpp"
#include "utils/WebServerUtils.hpp"

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
	// 资源分配，初始化等
	// TODO: 后续不断补充
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
		// TODO: 后续根据需要补充元数据字段
		std::vector<std::string> meta_fields = { "camera_id", "video_url" };
		if (WebServerUtils::check_field(req, res, meta_fields)) return;
		json request = json::parse(req.body);
		std::string camera_id = request["camera_id"];
		std::string video_url = request["video_url"];
	});
}