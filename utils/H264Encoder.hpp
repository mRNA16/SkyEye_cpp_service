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
#include <libavcodec/bsf.h>
}

// 轻量级 H.264 编码器助手，将 OpenCV Mat 转换为 WebRTC 兼容的 Annex-B 流。
// 内置 h264_mp4toannexb BSF 兜底，无论底层选中哪个编码器（libx264/nvenc/h264_mf）
// 最终输出均为 Annex-B 格式，确保 H264RtpPacketizer::StartSequence 能正确分包。
class H264Encoder {
public:
    H264Encoder();
    ~H264Encoder();

    // 初始化编码器
    // width/height: 输出分辨率；fps: 帧率；bitrate: 比特率(bps)
    bool Init(int width, int height, int fps, int bitrate = 1500000);

    // 编码一帧并通过回调返回 Annex-B 数据
    // pts 为帧的原始序号，对应 RTP 时间戳计算
    void Encode(const cv::Mat& mat, std::function<void(const uint8_t* data, size_t size, int64_t pts)> callback);

    // 强制下一帧为关键帧（用于响应 WebRTC 的 PLI 请求）
    void ForceKeyframe();

private:
    void Cleanup();

    AVCodecContext*  codec_ctx = nullptr;
    AVFrame*         frame     = nullptr;
    SwsContext*      sws_ctx   = nullptr;
    int64_t          pts       = 0;

    // Annex-B 兜底 BSF（h264_mp4toannexb）
    // 当编码器输出 AVCC 格式时自动转换；若已是 Annex-B 则透传
    AVBSFContext*    bsf_ctx   = nullptr;
    bool             need_bsf  = false; // 是否需要经过 BSF
    mutable size_t   last_encode_in_size_  = 0;
    mutable size_t   last_encode_out_size_ = 0;
    mutable int      last_encode_packets_  = 0; // 是否需要经过 BSF
};
