#pragma once
#include <string>
#include "service/json.hpp"
#include "service/httplib.h"

using json = nlohmann::json;

namespace WebServerUtils {
    std::string get_timestamp();
    int check_head(const httplib::Request& req, httplib::Response& res);
    int check_field(const httplib::Request& req, httplib::Response& res, std::vector<std::string> meta_fields);
    int async_post(const std::string& host, const std::string& path, const json& body, std::function<void(bool, const json&)> callback);
}