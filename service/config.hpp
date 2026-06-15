#pragma once
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <limits.h>
#include <unistd.h>
#endif

constexpr int PRINT_DETAIL = 1;

// GPU分配
constexpr int MIN_GPU_ID = 0;
constexpr int MAX_GPU_ID = 0;

// I3D设置
constexpr int BATCH_SIZE = 1;
constexpr int CHANNEL = 3;
constexpr int CHUNK_SIZE = 16;
constexpr int INPUT_H = 224;
constexpr int INPUT_W = 224;

// Tridet / 新动作检测分类数
constexpr int NUM_CLASSES = 25;

// YOLO 姿态检测
constexpr float YOLO_CONF_THRESHOLD = 0.25f;
constexpr float YOLO_NMS_THRESHOLD = 0.45f;

namespace PilotConfig {

struct RuntimeConfig {
	std::filesystem::path base_dir;
	std::filesystem::path client_index_path;
	std::filesystem::path i3d_model_path;
	std::filesystem::path tridet_model_path;
	std::filesystem::path yolo_model_path;
	std::filesystem::path ffmpeg_path;
	std::filesystem::path output_dir;
	std::filesystem::path temp_dir;
	std::string host = "0.0.0.0";
	int port = 8080;
	int gpu_device_id = 0;
};

inline std::string Trim(std::string value) {
	auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
	value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](unsigned char ch) { return !is_space(ch); }));
	value.erase(std::find_if(value.rbegin(), value.rend(), [&](unsigned char ch) { return !is_space(ch); }).base(), value.end());
	return value;
}

inline std::filesystem::path ExecutableDir() {
#ifdef _WIN32
	std::wstring buffer(MAX_PATH, L'\0');
	DWORD len = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
	if (len > 0 && len < buffer.size()) {
		buffer.resize(len);
		return std::filesystem::path(buffer).parent_path();
	}
#elif defined(__linux__)
	char buffer[PATH_MAX] = {};
	ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
	if (len > 0) {
		buffer[len] = '\0';
		return std::filesystem::path(buffer).parent_path();
	}
#endif
	return std::filesystem::current_path();
}

inline std::filesystem::path ResolvePath(const std::filesystem::path& base_dir, const std::string& value) {
	std::filesystem::path path(value);
	if (path.is_absolute()) return path.lexically_normal();
	return (base_dir / path).lexically_normal();
}

inline std::unordered_map<std::string, std::string> ReadProperties(const std::filesystem::path& config_path) {
	std::unordered_map<std::string, std::string> values;
	std::ifstream ifs(config_path);
	if (!ifs.is_open()) return values;

	std::string line;
	while (std::getline(ifs, line)) {
		line = Trim(line);
		if (line.empty() || line[0] == '#') continue;
		size_t eq = line.find('=');
		if (eq == std::string::npos) continue;
		std::string key = Trim(line.substr(0, eq));
		std::string value = Trim(line.substr(eq + 1));
		if (!key.empty()) values[key] = value;
	}
	return values;
}

inline std::filesystem::path FindConfigPath() {
	const std::filesystem::path exe_dir = ExecutableDir();
	const std::filesystem::path cwd = std::filesystem::current_path();
#ifdef _WIN32
	const std::filesystem::path platform_relative = "config/pilot_deploy.properties";
#else
	const std::filesystem::path platform_relative = "config/pilot_deploy.linux.properties";
#endif
	const std::filesystem::path common_relative = "config/pilot_deploy.properties";
	const std::filesystem::path candidates[] = {
		exe_dir / platform_relative,
		exe_dir.parent_path() / platform_relative,
		cwd / platform_relative,
		cwd.parent_path() / platform_relative,
		exe_dir / common_relative,
		exe_dir.parent_path() / common_relative,
		cwd / common_relative,
		cwd.parent_path() / common_relative
	};
	for (const auto& candidate : candidates) {
		if (std::filesystem::exists(candidate)) return candidate;
	}
	return exe_dir / platform_relative;
}

inline int ReadInt(const std::unordered_map<std::string, std::string>& values, const std::string& key, int fallback) {
	auto it = values.find(key);
	if (it == values.end()) return fallback;
	try {
		return std::stoi(it->second);
	} catch (...) {
		std::cerr << "[Config] Invalid integer for " << key << ": " << it->second
		          << ". Use fallback " << fallback << "." << std::endl;
		return fallback;
	}
}

inline RuntimeConfig LoadRuntimeConfig() {
	const std::filesystem::path config_path = FindConfigPath();
	const auto values = ReadProperties(config_path);

	std::filesystem::path base_dir;
	auto base_it = values.find("base_dir");
	if (base_it != values.end() && !base_it->second.empty()) {
		std::filesystem::path configured(base_it->second);
		base_dir = configured.is_absolute() ? configured : (config_path.parent_path() / configured);
	} else if (std::filesystem::exists(config_path)) {
		base_dir = config_path.parent_path();
	} else {
		base_dir = std::filesystem::current_path();
	}
	base_dir = std::filesystem::absolute(base_dir).lexically_normal();

	auto read_path = [&](const std::string& key, const std::string& fallback) {
		auto it = values.find(key);
		return ResolvePath(base_dir, it == values.end() || it->second.empty() ? fallback : it->second);
	};

	RuntimeConfig cfg;
	cfg.base_dir = base_dir;
	cfg.client_index_path = read_path("client_index", "client/index.html");
	cfg.i3d_model_path = read_path("i3d_model", "models/a320_new_full.onnx");
	cfg.tridet_model_path = read_path("tridet_model", "models/tridet_a320.onnx");
	cfg.yolo_model_path = read_path("yolo_model", "models/best.onnx");
#ifdef _WIN32
	const char* default_ffmpeg_path = "runtime/ffmpeg/ffmpeg.exe";
#else
	const char* default_ffmpeg_path = "runtime/ffmpeg/ffmpeg";
#endif
	cfg.ffmpeg_path = read_path("ffmpeg_path", default_ffmpeg_path);
	cfg.output_dir = read_path("output_dir", "output");
	cfg.temp_dir = read_path("temp_dir", "temp");

	auto host_it = values.find("host");
	if (host_it != values.end() && !host_it->second.empty()) cfg.host = host_it->second;
	cfg.port = ReadInt(values, "port", cfg.port);
	cfg.gpu_device_id = ReadInt(values, "gpu_device_id", cfg.gpu_device_id);

	std::error_code ec;
	std::filesystem::create_directories(cfg.output_dir, ec);
	if (ec) std::cerr << "[Config] Failed to create output_dir: " << cfg.output_dir << " (" << ec.message() << ")" << std::endl;
	std::filesystem::create_directories(cfg.temp_dir, ec);
	if (ec) std::cerr << "[Config] Failed to create temp_dir: " << cfg.temp_dir << " (" << ec.message() << ")" << std::endl;

	std::cout << "[Config] base_dir=" << cfg.base_dir.string() << std::endl;
	if (std::filesystem::exists(config_path)) {
		std::cout << "[Config] loaded=" << config_path.string() << std::endl;
	} else {
		std::cout << "[Config] config file not found, using defaults under current directory." << std::endl;
	}
	return cfg;
}

inline const RuntimeConfig& Get() {
	static RuntimeConfig cfg = LoadRuntimeConfig();
	return cfg;
}

inline std::string PathString(const std::filesystem::path& path) {
	return path.string();
}

inline std::filesystem::path ReportPath(const std::string& camera_id) {
	return Get().output_dir / ("report_" + camera_id + ".json");
}

inline std::filesystem::path TempPath(const std::string& file_name) {
	return Get().temp_dir / file_name;
}

} // namespace PilotConfig
