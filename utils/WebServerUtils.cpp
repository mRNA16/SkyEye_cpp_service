#include "WebServerUtils.hpp"
#include "service/json.hpp"
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
}