#pragma once

#include <opencv2/opencv.hpp>
#include <functional>
#include <vector>
#include <cstdint>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

// 轻量级 H.264 编码器助手，将 OpenCV Mat 转换为 WebRTC 兼容的 Annex-B 流
class H264Encoder {
public:
    H264Encoder();
    ~H264Encoder();

    // 初始化编码器参数
    // width/height: 帧大小
    // fps: 帧率
    // bitrate: 比特率，默认 1.5Mbps
    bool Init(int width, int height, int fps, int bitrate = 1500000);

    // 执行编码并触发回调
    void Encode(const cv::Mat& mat, std::function<void(const uint8_t*, size_t)> callback);

private:
    void Cleanup();

    AVCodecContext* codec_ctx;
    AVFrame* frame;
    SwsContext* sws_ctx;
    int64_t pts;
};
