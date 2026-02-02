#include "service/service.hpp"
#include <iostream>

class TestablePilotWebServer : public PilotWebServer {
public:
    // 暴露公共接口来调用父类的私有函数
    int test_launch_camera(const std::string& id, const std::string& url) {
        camera_thread_manager.set(id, true);
        return this->launch_camera(id, url);
    }
};

int main(int argc, char** argv) {
    // 获取输入参数
    std::string test_url;
    if (argc > 1) {
        test_url = argv[1];
    }
    else {
        // 默认测试地址（这里是MediaMTX推送的本地视频文件）
        test_url = "rtsp://127.0.0.1:8554/live";
        std::cout << "Usage: ./test [rtsp_url]. Using default: " << test_url << std::endl;
    }

    // 实例化服务器对象
    TestablePilotWebServer test_server;
    std::string camera_id = "test_cam_001";

    std::cout << "--- Starting Camera Launch Test ---" << std::endl;

    int result = test_server.test_launch_camera(camera_id, test_url);

    if (result == 0) {
        std::cout << "Test ended successfully." << std::endl;
    }
    else {
        std::cerr << "Test failed with error code: " << result << std::endl;
    }

    return 0;
}