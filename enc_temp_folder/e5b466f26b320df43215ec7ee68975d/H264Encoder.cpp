#include "H264Encoder.hpp"
#include <iostream>
#include <string>

H264Encoder::H264Encoder() {}

H264Encoder::~H264Encoder() {
    Cleanup();
}

void H264Encoder::Cleanup() {
    if (bsf_ctx)   { av_bsf_free(&bsf_ctx);           bsf_ctx   = nullptr; }
    if (codec_ctx) { avcodec_free_context(&codec_ctx); codec_ctx = nullptr; }
    if (frame)     { av_frame_free(&frame);             frame     = nullptr; }
    if (sws_ctx)   { sws_freeContext(sws_ctx);          sws_ctx   = nullptr; }
    pts      = 0;
    need_bsf = false;
}

bool H264Encoder::Init(int width, int height, int fps, int bitrate) {
    Cleanup();

    // ── 编码器选择策略 ──────────────────────────────────────────────────────
    // 优先级：libx264（兼容性最好，输出稳定 Annex-B）
    //        → h264_nvenc（NVIDIA 显卡，需额外设置 annexb）
    //        → h264_mf   （Windows 媒体基金，默认 AVCC，依赖 BSF 转换）
    // 注意：avcodec_find_encoder(AV_CODEC_ID_H264) 在 Windows 上会选中 h264_mf，
    //       格式不确定，因此改为按名称显式指定顺序。
    const AVCodec* codec = nullptr;
    for (const char* name : {"libx264", "h264_nvenc", "h264_mf", "h264"}) {
        codec = avcodec_find_encoder_by_name(name);
        if (codec) break;
    }
    if (!codec) {
        std::cerr << "[H264Encoder] FATAL: No H.264 encoder found!" << std::endl;
        return false;
    }
    std::cout << "[H264Encoder] Selected encoder: " << codec->name << std::endl;

    // ── 编码器上下文 ─────────────────────────────────────────────────────────
    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) return false;

    codec_ctx->width       = width;
    codec_ctx->height      = height;
    codec_ctx->time_base   = { 1, fps };
    codec_ctx->framerate   = { fps, 1 };
    codec_ctx->pix_fmt     = AV_PIX_FMT_YUV420P;
    codec_ctx->bit_rate    = bitrate;
    codec_ctx->rc_max_rate = bitrate;
    codec_ctx->rc_buffer_size = static_cast<int>(bitrate / 2); // 较小的 buffer 实现实时流平滑
    codec_ctx->max_b_frames = 0;
    codec_ctx->level = 40; // 提升至 Level 4.0
    // 每秒一个 IDR 关键帧，保证客户端断线重连后快速恢复画面
    codec_ctx->gop_size    = fps;
    // 不使用全局头（SPS/PPS 内嵌在每个关键帧数据包中），适配 Annex-B / RTP
    codec_ctx->flags      &= ~AV_CODEC_FLAG_GLOBAL_HEADER;
    codec_ctx->flags      |= AV_CODEC_FLAG_LOW_DELAY;

    // ── 编码器私有参数 ────────────────────────────────────────────────────────
    const std::string cname(codec->name);
    if (cname == "libx264" || cname == "h264") {
        av_opt_set(codec_ctx->priv_data, "preset",      "veryfast", 0);
        av_opt_set(codec_ctx->priv_data, "tune",        "zerolatency", 0);
        av_opt_set(codec_ctx->priv_data, "crf",         "23", 0);
        // repeat-headers=1  每个 IDR 帧前都带 SPS/PPS
        // annexb=1          强制 Annex-B start-code 输出
        // sliced-threads=0  禁用切片线程以防在低比特率下产生过多 NALU
        // rc-lookahead=0    强制零帧前瞻，配合 zerolatency
        av_opt_set(codec_ctx->priv_data, "x264-params",
                   "repeat-headers=1:annexb=1:sliced-threads=0:rc-lookahead=0", 0);
        av_opt_set(codec_ctx->priv_data, "profile", "baseline", 0);
    } else if (cname.find("nvenc") != std::string::npos) {
        av_opt_set(codec_ctx->priv_data, "preset",          "p1",  0); // 最快
        av_opt_set(codec_ctx->priv_data, "tune",            "ull", 0); // 超低延迟
        av_opt_set(codec_ctx->priv_data, "annexb",          "1",   0); // 强制 Annex-B
        av_opt_set(codec_ctx->priv_data, "repeat-headers",  "1",   0); // IDR 带参数集
        av_opt_set(codec_ctx->priv_data, "rc-lookahead",    "0",   0);
    }
    // h264_mf 没有 annexb 私有选项，依赖后续 BSF 做格式转换

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "[H264Encoder] Failed to open codec." << std::endl;
        return false;
    }

    // ── Annex-B BSF 兜底 ──────────────────────────────────────────────────────
    // 判断编码器是否会输出非 Annex-B（AVCC）格式：
    //   h264_mf 和部分 GPU 编码器开启 global header 后会以 extradata 存储 SPS/PPS，
    //   数据包中使用 4 字节长度前缀（AVCC），必须经过 BSF 转换。
    // 这里无论如何都挂载 BSF，它对已是 Annex-B 的流是透传的，零额外开销。
    const AVBitStreamFilter* bsf_filter = av_bsf_get_by_name("h264_mp4toannexb");
    if (bsf_filter) {
        if (av_bsf_alloc(bsf_filter, &bsf_ctx) < 0) {
            std::cerr << "[H264Encoder] Warning: Failed to alloc BSF, raw output will be used." << std::endl;
            bsf_ctx = nullptr;
        } else {
            // 把编码器的参数（含 extradata/SPS/PPS）传给 BSF
            avcodec_parameters_from_context(bsf_ctx->par_in, codec_ctx);
            bsf_ctx->time_base_in = codec_ctx->time_base;
            if (av_bsf_init(bsf_ctx) < 0) {
                std::cerr << "[H264Encoder] Warning: BSF init failed, raw output will be used." << std::endl;
                av_bsf_free(&bsf_ctx);
                bsf_ctx = nullptr;
            } else {
                need_bsf = true;
                std::cout << "[H264Encoder] h264_mp4toannexb BSF attached (Annex-B guaranteed)." << std::endl;
            }
        }
    } else {
        std::cerr << "[H264Encoder] Warning: h264_mp4toannexb BSF not found in this FFmpeg build." << std::endl;
    }

    // ── SwsContext: BGR24 → YUV420P ──────────────────────────────────────────
    frame = av_frame_alloc();
    if (!frame) return false;
    frame->format = codec_ctx->pix_fmt;
    frame->width  = width;
    frame->height = height;
    if (av_frame_get_buffer(frame, 32) < 0) return false;

    sws_ctx = sws_getContext(
        width, height, AV_PIX_FMT_BGR24,
        width, height, AV_PIX_FMT_YUV420P,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr
    );
    if (!sws_ctx) return false;

    pts = 0;
    std::cout << "[H264Encoder] Init OK: " << width << "x" << height
              << " @ " << fps << "fps, " << bitrate / 1000 << "kbps" << std::endl;
    return true;
}

void H264Encoder::Encode(const cv::Mat& mat, std::function<void(const uint8_t* data, size_t size, int64_t pts)> callback) {
    if (!codec_ctx || !sws_ctx || mat.empty()) return;

    last_encode_in_size_ = static_cast<size_t>(mat.total() * mat.elemSize());
    last_encode_out_size_ = 0;
    last_encode_packets_ = 0;

    // BGR → YUV420P
    const int      stride[1] = { static_cast<int>(mat.step) };
    const uint8_t* src[1]    = { mat.data };
    sws_scale(sws_ctx, src, stride, 0, mat.rows, frame->data, frame->linesize);
    frame->pts = pts++;

    if (avcodec_send_frame(codec_ctx, frame) < 0) {
        std::cerr << "[H264Encoder] avcodec_send_frame failed" << std::endl;
        return;
    }

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) return;

    while (avcodec_receive_packet(codec_ctx, pkt) >= 0) {
        ++last_encode_packets_;
        if (need_bsf && bsf_ctx) {
            if (av_bsf_send_packet(bsf_ctx, pkt) >= 0) {
                AVPacket* bsf_pkt = av_packet_alloc();
                if (bsf_pkt) {
                    while (av_bsf_receive_packet(bsf_ctx, bsf_pkt) >= 0) {
                        last_encode_out_size_ += static_cast<size_t>(bsf_pkt->size);
                        callback(bsf_pkt->data, static_cast<size_t>(bsf_pkt->size), bsf_pkt->pts);
                        av_packet_unref(bsf_pkt);
                    }
                    av_packet_free(&bsf_pkt);
                }
            }
        } else {
            last_encode_out_size_ += static_cast<size_t>(pkt->size);
            callback(pkt->data, static_cast<size_t>(pkt->size), pkt->pts);
            av_packet_unref(pkt);
        }
    }
    av_packet_free(&pkt);

    if (pts % 15 == 0) {
        std::cout << "[H264Encoder] encode summary: in=" << last_encode_in_size_
                  << " out=" << last_encode_out_size_
                  << " packets=" << last_encode_packets_
                  << " pts=" << pts << std::endl;
    }
}

void H264Encoder::ForceKeyframe() {
    // 标记下一帧为 I 帧，确保客户端能立即获得画面
    if (frame) {
        frame->pict_type = AV_PICTURE_TYPE_I;
    }
}
