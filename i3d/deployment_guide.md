# I3D 模型 C++ ONNX 部署指南

本指南详细介绍了如何将转换后的 `a320.onnx` 模型集成到 C++ 服务端中。

## 1. 依赖库
*   **ONNX Runtime C++ API**: 用于加载和运行 `.onnx` 模型。
*   **OpenCV**: 用于视频解码和图像预处理（Resize, Normalize）。

## 2. 核心代码实现

### 2.1 预处理逻辑 (关键)
I3D 模型的 C++ 预处理必须与 Python 端 (`extract_features.py`) 严格一致：
1.  **尺寸调整**：双线性插值缩放到 224x224。
2.  **归一化**：将 `[0, 255]` 映射到 `[-1, 1]`。
    *   公式：`value = (value / 255.0) * 2.0 - 1.0`。
3.  **维度转换**：将 OpenCV 的 `HWC` (BGR) 转换为模型的 `CTHW` (RGB)。

### 2.2 推理伪代码示例
```cpp
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>

void RunInference() {
    // 1. 初始化 Ort 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "I3D_Inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);

    // 2. 加载模型
    const wchar_t* model_path = L"models/a320.onnx";
    Ort::Session session(env, model_path, session_options);

    // 3. 准备数据 (假设我们有 16 帧 224x224 的图像)
    int64_t batchSize = 1;
    int64_t channels = 3;
    int64_t frames = 16;
    int64_t height = 224;
    int64_t width = 224;
    std::vector<int64_t> input_shape = {batchSize, channels, frames, height, width};
    
    // 输入 Tensor 大小: 1 * 3 * 16 * 224 * 224
    std::vector<float> input_tensor_values(batchSize * channels * frames * height * width);

    // [预处理循环]
    // 遍历 16 帧，每帧做：Resize -> RGB转换 -> 归一化 -> 填充到 input_tensor_values
    // 注意填充顺序: Channel(R)->Frame1->Frame2... 然后 Channel(G)...
    // 或者按 NCHWD (N,C,T,H,W) 顺序：
    // for (c) for (t) for (h) for (w):
    //    input_tensor_values[c*T*H*W + t*H*W + h*W + w] = NormalizedPixel;

    // 4. 创建 Ort Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), 
        input_shape.data(), input_shape.size()
    );

    // 5. 运行推理
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // 6. 获取输出
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    // 输出维度通常是 [1, 1024] (如果是特征模式)
}
```

## 3. 注意事项
1.  **输入形状**：由于我们在导出时设置了 `dynamic_axes`，您可以输入不同长度的视频帧（例如 16 帧、32 帧），但建议保持为 16 的倍数以获得最佳稳定性。
2.  **性能优化**：
    *   在 C++ 端，可以使用 **CUDA Execution Provider** 加速：`session_options.AppendExecutionProvider_CUDA(cuda_options);`。
    *   确保 OpenCV 开启了硬件解码能力。
3.  **坐标转换**：Python 的 `transpose([4, 1, 2, 3])` 在 C++ 中对应平展数组的填充步长计算，务必保证索引正确。

## 4. 常见问题排查
*   **结果不对**：检查归一化公式是否为 `(x/255)*2-1`，且 RGB 通道顺序是否正确。
*   **算子不支持**：如果遇到 `MaxPool3d` 或 `Conv3d` 报错，请升级 ONNX Runtime 版本至 1.10+。
