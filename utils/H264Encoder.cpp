#include "H264Encoder.hpp"
#include <iostream>

H264Encoder::H264Encoder() : codec_ctx(nullptr), frame(nullptr), sws_ctx(nullptr), pts(0) {}

H264Encoder::~H264Encoder() {
    Cleanup();
}

void H264Encoder::Cleanup() {
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (frame) av_frame_free(&frame);
    if (sws_ctx) sws_freeContext(sws_ctx);
    codec_ctx = nullptr;
    frame = nullptr;
    sws_ctx = nullptr;
}

bool H264Encoder::Init(int width, int height, int fps, int bitrate) {
    Cleanup();

    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) codec = avcodec_find_encoder_by_name("libx264");
    if (!codec) codec = avcodec_find_encoder_by_name("h264_nvenc"); // NVIDIA 硬件加速
    if (!codec) codec = avcodec_find_encoder_by_name("h264_mf");    // Windows Media Foundation
    if (!codec) codec = avcodec_find_encoder_by_name("h264");
    
    // 如果实在没有 H264 编码器，尝试 mpeg4 做最后保底（虽然 WebRTC 兼容性较差，但至少能运行）
    if (!codec) codec = avcodec_find_encoder(AV_CODEC_ID_MPEG4);

    if (!codec) {
        std::cerr << "[H264Encoder] CRITICAL: No suitable encoder found (tried h264, libx264, nvenc, mf, mpeg4)." << std::endl;
        return false;
    }

    std::cout << "[H264Encoder] Selected encoder: " << codec->name << std::endl;

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) return false;

    codec_ctx->width = width;
    codec_ctx->height = height;
    codec_ctx->time_base = { 1, fps };
    codec_ctx->framerate = { fps, 1 };
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx->bit_rate = bitrate;
    codec_ctx->gop_size = 10;
    codec_ctx->max_b_frames = 0;

    // 只有 libx264 支持这些特定的私有选项
    if (std::string(codec->name).find("x264") != std::string::npos || std::string(codec->name) == "h264") {
        av_opt_set(codec_ctx->priv_data, "preset", "ultrafast", 0);
        av_opt_set(codec_ctx->priv_data, "tune", "zerolatency", 0);
    } else if (std::string(codec->name).find("nvenc") != std::string::npos) {
        // NVIDIA NVENC 专用低延迟参数
        av_opt_set(codec_ctx->priv_data, "preset", "p1", 0); // p1 为最快
        av_opt_set(codec_ctx->priv_data, "tune", "ull", 0);  // ull 为超低延迟
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "[H264Encoder] Failed to open codec." << std::endl;
        return false;
    }

    frame = av_frame_alloc();
    frame->format = codec_ctx->pix_fmt;
    frame->width = width;
    frame->height = height;
    if (av_frame_get_buffer(frame, 32) < 0) return false;

    sws_ctx = sws_getContext(width, height, AV_PIX_FMT_BGR24,
        width, height, AV_PIX_FMT_YUV420P,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    
    pts = 0;
    return true;
}

void H264Encoder::Encode(const cv::Mat& mat, std::function<void(const uint8_t*, size_t)> callback) {
    if (!codec_ctx || !sws_ctx || mat.empty()) return;

    // 转换色彩空间 BGR -> YUV420P
    const int stride[1] = { static_cast<int>(mat.step) };
    const uint8_t* data[1] = { mat.data };
    sws_scale(sws_ctx, data, stride, 0, mat.rows, frame->data, frame->linesize);

    frame->pts = pts++;

    if (avcodec_send_frame(codec_ctx, frame) >= 0) {
        AVPacket* pkt = av_packet_alloc();
        while (avcodec_receive_packet(codec_ctx, pkt) >= 0) {
            callback(pkt->data, pkt->size);
            av_packet_unref(pkt);
        }
        av_packet_free(&pkt);
    }
}
