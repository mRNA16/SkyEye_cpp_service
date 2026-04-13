#pragma once
#include <string>

constexpr int PRINT_DETAIL = 1;

// GPU分配
constexpr int MIN_GPU_ID = 0;
constexpr int MAX_GPU_ID = 0;

// I3D设置
constexpr int BATCH_SIZE = 1;
constexpr int CHANNEL = 3;
constexpr int CHUNK_SIZE = 16;
constexpr int INPUT_H = 224;
constexpr int INPUT_W = 224;
const std::string I3D_MODEL_PATH = R"(E:\pilot\i3d\models\a320_new.onnx)";

// Tridet / 新动作检测分类数
constexpr int NUM_CLASSES = 17;
const std::string ACTIONFORMER_MODEL_PATH = R"(E:\pilot\actionformer_release\ckpt\thumos_i3d_tempNew\actionformer_a320.onnx)";
const std::string TRIDET_MODEL_PATH = R"(E:\pilot\algos\tridet_a320.onnx)";