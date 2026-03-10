#pragma once
#include <string>

constexpr int PRINT_DETAIL = 1;

// GPU分配
constexpr int MIN_GPU_ID = 0;
constexpr int MAX_GPU_ID = 0;

// I3D设置
constexpr int CHUNK_SIZE = 16;
constexpr int INPUT_H = 224;
constexpr int INPUT_W = 224;
const std::string I3D_MODEL_PATH = R"(E:\pilot\i3d\models\a320.onnx)";

// ActionFormer设置
const std::string ACTIONFORMER_MODEL_PATH = R"(E:\pilot\actionformer_release\ckpt\thumos_i3d_tempNew\actionformer_a320.onnx)";