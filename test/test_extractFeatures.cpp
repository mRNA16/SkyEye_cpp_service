#include "../service/service.hpp"
#include <iostream>

/**
 * @brief 测试版服务端类，暴露保护方法以便进行单元测试
 */
class TestablePilotWebServer : public PilotWebServer {
public:
    // 模拟初始化模型
    int test_init() {
        return this->loadModels();
    }

    // 模拟摄像流拉取及算法处理（测试新架构下的生产者-消费者模式）
    int test_launch_and_extract(const std::string& id, const std::string& url) {
        // 在子线程管理中注册该摄像头，模拟其正在运行
        camera_thread_manager.set(id, true);
        // 调用重构后的主入口，里面包含了两条流水线：显示队列与算法队列
        return this->launch_camera(id, url);
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
    std::cout << "\n[1/2] --- 正在初始化模型（I3D & ActionFormer） ---" << std::endl;
    if (test_server.test_init() != 0) {
        std::cerr << "错误: 模型初始化失败！请检查模型文件路径配置。" << std::endl;
        return -1;
    }
    std::cout << "模型初始化成功。" << std::endl;

    // 2. 测试组合模块
    std::string camera_id = "test_combined_cam_001";
    std::cout << "\n[2/2] --- 正在启动混合架构测试 ---" << std::endl;
    std::cout << "输入源: " << test_url << std::endl;
    
    // 该方法会进入生产者循环，并开启显示线程和算法线程。可通过在显示窗口按 ESC 停止。
    int result = test_server.test_launch_and_extract(camera_id, test_url);

    if (result == 0) {
        std::cout << "\n混合测试执行完毕，算法线程已确保清空队列。" << std::endl;
    }
    else {
        std::cerr << "\n测试过程中出现错误，返回代码: " << result << std::endl;
    }

    return 0;
}
