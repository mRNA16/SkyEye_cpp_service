#include "WebServerUtils.hpp"
#include "service/json.hpp"
#include "service/config.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>

using json = nlohmann::json;

namespace WebServerUtils {
    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);

        std::tm tm_struct;
#ifdef _WIN32
        localtime_s(&tm_struct, &time);
#else
        localtime_r(&time, &tm_struct);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm_struct, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

    int check_head(const httplib::Request& req, httplib::Response& res) {
        if (!req.has_header("Content-Type") || req.get_header_value("Content-Type") != "application/json") {
            res.status = 400;
            json error;
            error["code"] = 400;
            error["msg"] = "Content-Type doesn't match 'application/json'";
            res.set_content(error.dump(), "application/json");
            return -1;
        }
        return 0;
    }

    int check_field(const httplib::Request& req, httplib::Response& res,std::vector<std::string> meta_fields) {
        json request;
        try {
            request = json::parse(req.body);
        }
        catch (json::parse_error& e) {
            res.status = 500;
            json error;
            error["code"] = 500;
            error["msg"] = "Invalid json format";
            error["details"] = e.what();
            res.set_content(error.dump(), "application/json");
            return -1;
        }
        for (const auto& field : meta_fields) {
            if (!request.contains(field)) {
                res.status = 500;
                json error;
                error["code"] = 500;
                error["msg"] = "Missing field :" + field;
                res.set_content(error.dump(), "application/json");
                return -1;
            }
        }
        return 0;
    }

    int async_post(const std::string& host,const std::string& path,
                    const json& body,
                    std::function<void(bool, const json&)> callback) {
        std::thread([=]() {
            httplib::Client cli(host);
            cli.set_connection_timeout(5); // 5�볬ʱ

            httplib::Headers headers = {
                {"Content-Type", "application/json"},
                {"Accept", "application/json"}
            };

            std::string body_str = body.dump();
            if (PRINT_DETAIL) std::cout << body_str << std::endl;
            auto res = cli.Post(path.c_str(), headers, body_str, "application/json");

            bool success = false;
            json response_json;
            if (res) {
                if (res->status == 200) {
                    try {
                        // std::cout << res->body << std::endl;
                        response_json = json::parse(res->body);
                        success = true;
                    }
                    catch (const json::parse_error& e) {
                        std::cerr << "Invalid JSON response: " << e.what() << "\n";
                        response_json = { {"error", "Invalid JSON response"} };
                    }
                }
                else {
                    try {
                        // std::cout << res->body << std::endl;
                        response_json = json::parse(res->body);
                        success = false;
                    }
                    catch (const json::parse_error& e) {
                        std::cerr << "Invalid JSON response: " << e.what() << "\n";
                        response_json = { {"error", "Invalid JSON response"} };
                    }
                }
            }
            else {
                response_json = { {"error",  httplib::to_string(res.error())} };
            }

            callback(success, response_json);
            }).detach();

        return 0;
    }
}