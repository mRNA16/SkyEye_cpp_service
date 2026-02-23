#include "service/service.hpp"
#include <iostream>

/**
 * @brief 测试版服务端类，暴露保护方法以便进行单元测试
 */
class TestablePilotWebServer : public PilotWebServer {
public:
    // 模拟初始化
    int test_init() {
        return this->init();
    }

    // 模拟特征提取
    int test_extract_features(const std::string& id, const std::string& url) {
        // 在子线程管理中注册该摄像头，模拟其正在运行
        camera_thread_manager.set(id, true);
        return this->extract_features(id, url);
    }
};

int main(int argc, char** argv) {
    std::string test_url;
    if (argc > 1) {
        test_url = argv[1];
    }
    else {
        // 默认使用本地 RTSP 流或文件
        test_url = "rtsp://127.0.0.1:8554/live";
        std::cout << "提示: 未提供 RTSP 地址，使用默认地址: " << test_url << std::endl;
        std::cout << "用法: ./test_extractFeatures [rtsp_url]" << std::endl;
    }

    TestablePilotWebServer test_server;
    
    // 1. 初始化模型
    std::cout << "\n[1/2] --- 正在初始化 I3D 模型 ---" << std::endl;
    if (test_server.test_init() != 0) {
        std::cerr << "错误: I3D 模型初始化失败！请检查模型文件路径。" << std::endl;
        return -1;
    }
    std::cout << "I3D 模型初始化成功。" << std::endl;

    // 2. 测试特征提取
    std::string camera_id = "test_feature_cam_001";
    std::cout << "\n[2/2] --- 正在启动特征提取测试 ---" << std::endl;
    std::cout << "输入源: " << test_url << std::endl;
    
    // 该方法会进入循环读取视频帧，按 Ctrl+C 或断开流来停止
    int result = test_server.test_extract_features(camera_id, test_url);

    if (result == 0) {
        std::cout << "\n特征提取测试执行完毕。" << std::endl;
    }
    else {
        std::cerr << "\n测试过程中出现错误，返回代码: " << result << std::endl;
    }

    return 0;
}
