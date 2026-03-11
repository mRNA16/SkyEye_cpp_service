#include "../service/service.hpp"
#include <iostream>

class TestablePilotWebServer : public PilotWebServer {
public:
    int test_init() {
        return this->loadModels();
    }

    int test_launch_camera(const std::string& id, const std::string& url) {
        camera_thread_manager.set(id, true);
        return this->launch_camera(id, url);
    }
};

int main(int argc, char** argv) {
    std::string test_url;
    if (argc > 1) {
        test_url = argv[1];
    }
    else {
        test_url = "rtsp://127.0.0.1:8554/live";
        std::cout << "Usage: ./test [rtsp_url]. Using default: " << test_url << std::endl;
    }

    TestablePilotWebServer test_server;
    std::string camera_id = "test_cam_001";

    std::cout << "--- Initializing Models (I3D, ActionFormer) ---" << std::endl;
    if (test_server.test_init() != 0) {
        std::cerr << "Failed to load models!" << std::endl;
        return -1;
    }

    std::cout << "--- Starting Camera Launch Test (Producer-Consumer) ---" << std::endl;
    int result = test_server.test_launch_camera(camera_id, test_url);

    if (result == 0) {
        std::cout << "Test ended successfully." << std::endl;
    }
    else {
        std::cerr << "Test failed with error code: " << result << std::endl;
    }

    return 0;
}