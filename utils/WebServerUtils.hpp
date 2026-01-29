#pragma once
#include <string>
#include "service/httplib.h"

namespace WebServerUtils {
    std::string get_timestamp();
    int check_head(const httplib::Request& req, httplib::Response& res);
    int check_field(const httplib::Request& req, httplib::Response& res, std::vector<std::string> meta_fields);
}