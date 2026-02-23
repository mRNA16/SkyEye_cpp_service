# I3D 模型部署至 C++ 服务端 (ONNX) 实施方案

该方案旨在将现有的 I3D 模型（`a320.pt`）导出为 ONNX 格式，并提供在 C++ 服务端使用 ONNX Runtime 进行推理的指导。

## 1. 环境准备
*   **Python 环境**：需要安装 `torch`, `onnx`, `onnxruntime`（可选，用于验证）。
*   **C++ 环境**：需要配置好 **ONNX Runtime (C++) API** 和 **OpenCV**（用于图像预处理）。

## 2. 模型导出 (Python)
将编写 `export_onnx.py` 脚本，执行以下步骤：
1.  根据 `pytorch_i3d.py` 中的定义初始化 `InceptionI3d` 模型。
2.  加载 `a320.pt` 的权重。注意根据 `extract_features.py` 中的逻辑处理 `state_dict`（如 DataParallel 的前缀处理）。
3.  定义输入形状：`[batch_size, 3, num_frames, 224, 224]`。
4.  使用 `torch.onnx.export` 导出模型。
    *   **动态轴设置**：允许 `batch_size` 和 `num_frames` 动态变化。
    *   **算子版本**：建议使用 `opset_version=14` 以支持 3D 卷积及池化算子。

## 3. C++ 服务端集成
在 C++ 服务端，部署流程如下：

### 3.1 推理环境初始化
1.  创建 `Ort::Env` 和 `Ort::SessionOptions`。
2.  加载 `.onnx` 模型文件，创建 `Ort::Session`。

### 3.2 预处理 (OpenCV)
I3D 模型要求输入为：
*   **尺寸**：224x224。
*   **通道**：RGB 顺序。
*   **归一化**：根据 `extract_features.py` 中的配置，像素值应映射到 `[-1, 1]`。
    *   公式：`data = (data * 2 / 255) - 1`。
*   **维度排列**：从 `[T, H, W, C]` 转换为 `[C, T, H, W]`（即 `[3, 16, 224, 224]`）。

### 3.3 推理执行
1.  将预处理后的视频帧序列（16 帧或其他数量）封装为 `Ort::Value` (Tensor)。
2.  调用 `session.Run()` 进行推理。
3.  获取并处理输出结果（特征向量或分类结果）。

## 4. 验证与测试
1.  比较 Python 原生推理结果与 C++ 推理结果的差异，确保精度一致。
