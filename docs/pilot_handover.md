# 飞行员模拟训练数字教员系统交接文档

本文档面向后续接手开发、部署和维护人员。内容基于当前仓库源码整理，重点说明系统职责、代码结构、运行链路、接口、配置、构建部署和常见维护点。

## 1. 系统概览

本项目是一个飞行员模拟训练数字教员后端系统，当前产品名在前端页面中体现为 `SkyEye`。系统接收实时 RTSP 视频流或本地视频文件，对驾驶舱/控制台操作进行动作检测，并在任务结束后生成动作检测报告。系统同时提供 Web 前端，用于启动分析、查看 WebRTC 实时画面、轮询在线预测结果和获取最终报告。

当前主程序是 C++17 服务端：

- HTTP 服务：基于 `cpp-httplib`，入口为 `service/entry.cpp`。
- 前端页面：单文件 `client/index.html`，由后端 `/` 路由直接返回，无单独构建步骤。
- 视频处理：OpenCV 读取/处理帧，FFmpeg 编码 H.264。
- 实时预览：libdatachannel + WebRTC，将后端叠框后的视频送到浏览器。
- 推理框架：ONNX Runtime，支持 CUDA Execution Provider。
- 模型链路：I3D 提特征，TriDet 做时序动作检测，YOLO Pose 用于画面叠框显示。

## 2. 代码目录说明

| 路径 | 作用 |
| --- | --- |
| `service/` | HTTP 服务、WebRTC 信令、任务调度、线程安全容器、配置读取。核心文件为 `service.cpp` / `service.hpp` / `config.hpp`。 |
| `feature/` | I3D 特征提取和 TriDet 动作检测封装。 |
| `yolo/` | YOLOv8-Pose ONNX 推理封装，用于检测框和关键点叠加显示。 |
| `utils/` | Web 工具函数和 H.264 编码器封装。 |
| `client/` | 前端单页 HTML，包含样式、控制逻辑、WebRTC 逻辑和接口调用。 |
| `config/` | Windows/Linux 运行配置模板。 |
| `docs/` | 部署文档、WebRTC 问题记录和本文档。 |
| `scripts/` | Windows/Linux 离线打包脚本。 |
| `test/` | RTSP、特征提取、YOLO 的测试入口。 |
| `i3d/`、`algos/`、`yolo/config/` | 打包脚本默认查找模型的位置。模型文件通常较大，部署时会复制到 `models/`。 |
| `out/`、`x64/`、`packages/` | 构建产物或依赖相关目录。 |

根目录下的 `implementation_plan.md`、`walkthrough.md`、`task.md` 是早期 I3D 开发记录，部分描述与当前实现不完全一致。后续维护请以当前源码、`docs/windows_deployment.md`、`docs/linux_deployment.md` 和本文档为准。

## 3. 主流程和数据流

### 3.1 服务启动

程序入口：

```cpp
// service/entry.cpp
int main() {
    PilotWebServer server;
    server.boot();
}
```

`PilotWebServer::boot()` 的主要步骤：

1. 读取运行配置，初始化输出目录和临时目录。
2. 加载 I3D 模型；尝试加载 YOLO 模型，YOLO 失败不会阻断主服务。
3. 注册 HTTP 路由、CORS 预检和 WebRTC offer 路由。
4. 返回 `client/index.html` 作为根页面。
5. 按配置监听 `host:port`，默认 `0.0.0.0:8080`。

### 3.2 实时 RTSP 分析

用户在前端 LIVE STREAM 页输入 `camera_id` 和 RTSP URL 后，前端调用：

```text
POST /launch_camera
```

后端为该 `camera_id` 启动后台线程。核心处理链路为：

1. OpenCV 打开 RTSP 视频源。
2. 原始帧进入显示队列，用于本地预览和 WebRTC 编码。
3. 原始帧缩放后进入算法队列 `HybridVideoQueue`。
4. `extract_features()` 每累计 `CHUNK_SIZE=16` 帧运行 I3D。
5. I3D 输出：
   - `features`：1024 维特征，L2 归一化后送入 TriDet。
   - `logits`：17 维分类概率，累加后用于最终 Score Fusion。
6. `tridet_predict()` 在线阶段每累计 20 个特征 chunk 做一次伪在线预测，供 `/get_live_prediction` 展示。
7. 用户调用 `/offline_camera` 或视频结束后，特征队列停止，TriDet 用全部特征执行离线推理并写报告。

最终报告写到：

```text
output/report_<camera_id>.json
```

### 3.3 本地视频分析

前端 LOCAL VIDEO 页既可以用浏览器本地播放视频，也可以把服务端可访问的视频路径提交给后端分析：

```text
POST /launch_local_video
```

本地视频分析复用与 RTSP 基本一致的算法链路。区别是视频源为 `file_path`，帧率来自视频文件元数据，读取到文件末尾后自动进入 finalizing 并生成报告。

### 3.4 实时视频显示

`PilotWebServer::live()` 负责显示和 WebRTC 编码发送：

- 显示队列由生产者推入原始帧。
- YOLO 线程每 5 帧抽一次做姿态检测，结果缓存为最新叠框信息。
- 编码线程把叠框帧缩放到 `1280x720 @ 15fps`，用 `H264Encoder` 输出 Annex-B H.264。
- WebRTC 会话通过 `/webrtc/offer` 建立，后端使用 libdatachannel 将 H.264 RTP 发给浏览器。
- `enable_display_` 只控制本地 OpenCV 窗口预览，浏览器 WebRTC 画面仍可工作。

## 4. 模型与算法封装

### 4.1 I3D 特征提取

文件：

- `feature/feature.hpp`
- `feature/feature.cpp`

当前输入配置：

- 输入节点：`input`
- 输出节点：`output` 和 `logits`
- 输入形状：`[1, 3, 16, 224, 224]`
- 预处理：BGR 转 RGB，Resize 到 `224x224`，ImageNet mean/std 归一化，按 `NCHWD` 排列。
- 输出：
  - `features`：TriDet 输入特征。
  - `logits`：全局动作分类概率融合依据。

注意：根目录早期文档中写过 `(x/255)*2-1`，当前源码实际是 ImageNet mean/std 归一化。模型或训练代码变更时必须核对这里。

### 4.2 TriDet 时序动作检测

文件：

- `feature/tridet.hpp`
- `feature/tridet.cpp`

当前配置：

- 输入节点：`features`
- 输出节点：`cls_logits`、`offsets`、`lb_logits`、`rb_logits`、`points`
- 输入维度：`[1, 1024, 2304]`
- 类别数：`NUM_CLASSES=17`
- 每个 session 单独创建 TriDet 实例，避免共享状态污染。

在线预测使用 `Run()`，离线报告使用 `RunOffline()`。最终报告还会在 `service.cpp` 中做一次同类相邻片段合并和阈值过滤。

动作名称在 `service.cpp` 的 `ACTION_NAMES` 中维护：

```text
Yoke, ThrottleLever, LandingGear, SpeedBrakes, Flap, Computer,
TrimWheel, EngineSwitch, EFISControl, SpeedSel, HeadingSel,
AltitudeSel, VerticalSpeedSel, AutoPilot, LightControl, AlartLight, Others
```

如更换模型类别顺序，必须同步更新 `NUM_CLASSES` 和 `ACTION_NAMES`。

### 4.3 YOLO Pose 叠框

文件：

- `yolo/yolo_detector.hpp`
- `yolo/yolo_detector.cpp`

当前用于 WebRTC / 本地窗口的视频叠框，不参与最终动作报告。模型假设：

- 输入节点：`images`
- 输出节点：`output0`
- 输入尺寸：`640x640`
- 类别数：6
- 关键点数：1
- 置信度阈值：`YOLO_CONF_THRESHOLD=0.25`
- NMS 阈值：`YOLO_NMS_THRESHOLD=0.45`

叠框显示标签在 `PilotWebServer::draw_yolo_detections()` 中维护：

```text
p1_normal, p1_grip, p1_point, p2_normal, p2_grip, p2_point
```

## 5. HTTP 接口

默认服务地址：

```text
http://127.0.0.1:8080/
```

接口统一返回 JSON，前端当前使用 `Content-Type: text/plain` 发送 JSON 字符串，后端没有强制校验 `application/json`。

### 5.1 启动实时流

```http
POST /launch_camera
```

请求：

```json
{
  "camera_id": "CAM_TEST_001",
  "video_url": "rtsp://localhost:8554/live"
}
```

说明：如果同名任务已在运行，会返回 `running`，不会重复启动。

### 5.2 停止实时流并生成报告

```http
POST /offline_camera
```

请求：

```json
{
  "camera_id": "CAM_TEST_001"
}
```

说明：该接口将任务标记为停止，后端会等待算法线程收尾并生成最终报告。

### 5.3 启动本地视频分析

```http
POST /launch_local_video
```

请求：

```json
{
  "session_id": "LOCAL_001",
  "file_path": "D:\\videos\\flight.mp4"
}
```

说明：`file_path` 必须是后端进程所在机器可访问的路径，不是浏览器上传文件路径。

### 5.4 查询任务状态

```http
POST /get_task_status
```

请求：

```json
{
  "camera_id": "CAM_TEST_001"
}
```

典型状态：

- `starting`
- `running`
- `finalizing`
- `completed`
- `failed`
- `unknown`

返回中包含 `has_report`，用于前端判断报告是否可读取。

### 5.5 获取最终报告

```http
POST /get_report
```

请求：

```json
{
  "camera_id": "CAM_TEST_001"
}
```

成功返回：

```json
{
  "code": 200,
  "data": {
    "camera_id": "CAM_TEST_001",
    "summary": "Action Detection Report",
    "actions": [
      {
        "start": 1.23,
        "end": 2.34,
        "score": 0.56,
        "action": "Yoke"
      }
    ]
  }
}
```

### 5.6 获取伪在线预测

```http
POST /get_live_prediction
```

请求：

```json
{
  "camera_id": "CAM_TEST_001"
}
```

说明：只用于运行期间日志展示，不等价于最终报告。任务结束后缓存会被清理。

### 5.7 WebRTC 信令

```http
POST /webrtc/offer
```

请求：

```json
{
  "camera_id": "CAM_TEST_001",
  "sdp": "<browser offer sdp>"
}
```

后端要求浏览器 offer 中包含 H.264。返回 answer SDP。当前浏览器侧禁用了外部 STUN，主要面向本机或局域网调试。

### 5.8 本地 OpenCV 窗口开关

```http
POST /toggle_display
```

说明：切换 `cv::imshow` 本地预览窗口，主要用于 Windows 桌面调试。Linux release preset 默认关闭本地窗口编译选项。

## 6. 配置文件

Windows 默认配置：

```text
config/pilot_deploy.properties
```

Linux 默认配置：

```text
config/pilot_deploy.linux.properties
```

Linux 优先读取 `.linux.properties`，找不到时回退到通用配置。配置查找位置包括可执行文件目录、上级目录、当前工作目录和当前工作目录上级。

关键配置项：

| 键 | 说明 |
| --- | --- |
| `base_dir` | 部署根目录，默认相对 `config/` 指向上一层。 |
| `client_index` | 前端 HTML 路径。 |
| `i3d_model` | I3D ONNX 模型路径。 |
| `tridet_model` | TriDet ONNX 模型路径。 |
| `yolo_model` | YOLO ONNX 模型路径。 |
| `ffmpeg_path` | FFmpeg 可执行文件路径。 |
| `output_dir` | 报告输出目录。 |
| `temp_dir` | 临时算法缓存目录。 |
| `host` / `port` | HTTP 服务监听地址和端口。 |
| `gpu_device_id` | ONNX Runtime CUDA 设备编号；CPU 模式可配合编译选项关闭 CUDA EP。 |

当前默认模型部署名：

```text
models/a320_new_full.onnx
models/tridet_a320.onnx
models/best.onnx
```

## 7. 构建和运行

### 7.1 依赖

核心依赖：

- CMake 3.10+
- C++17 编译器，Windows 下为 MSVC，Linux 下为 GCC/G++
- Ninja
- FFmpeg 开发库和可执行文件
- OpenCV
- LibTorch
- ONNX Runtime，GPU 版需要 CUDA/cuDNN 匹配
- libdatachannel
- Python3 development
- pthread / Windows 线程库

### 7.2 Windows 构建

当前 `CMakePresets.json` 默认路径假设：

```text
D:/vcpkg
D:/onnxruntime-win-x64-gpu-1.24.1
```

常用命令：

```powershell
cmake --preset windows-release
cmake --build --preset pilot-release
```

调试构建：

```powershell
cmake --preset windows-base
cmake --build --preset pilot-debug
```

构建产物默认位于：

```text
out/build/windows-release/pilot.exe
out/build/windows-base/pilot.exe
```

### 7.3 Linux 构建

详见 `docs/linux_deployment.md`。典型命令：

```bash
cmake --preset linux-release \
  -DORT_DIR=/opt/onnxruntime \
  -DCMAKE_PREFIX_PATH="/opt/libtorch;/opt/vcpkg/installed/x64-linux" \
  -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake

cmake --build --preset pilot-linux-release
```

CPU 模式：

```bash
cmake --preset linux-release \
  -DORT_DIR=/opt/onnxruntime \
  -DCMAKE_PREFIX_PATH="/opt/libtorch;/opt/vcpkg/installed/x64-linux" \
  -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
  -DPILOT_USE_ORT_CUDA=OFF
```

### 7.4 开发环境运行

从源码根目录启动构建产物，然后访问：

```text
http://127.0.0.1:8080/
```

如果模型或前端路径找不到，优先检查：

- 当前工作目录
- `config/pilot_deploy*.properties`
- `base_dir`
- 模型文件是否存在

## 8. 离线部署包

### 8.1 Windows

生成部署包：

```powershell
.\scripts\package_windows.ps1
```

或指定构建目录：

```powershell
.\scripts\package_windows.ps1 -PreferredBuildDir "out\build\windows-release"
```

输出目录：

```text
dist/pilot_windows/
```

目标机运行：

```powershell
.\bin\pilot.exe
```

详见 `docs/windows_deployment.md`。

### 8.2 Linux

生成部署包：

```bash
scripts/package_linux.sh
```

输出目录：

```text
dist/pilot_linux/
```

目标机运行：

```bash
./run.sh
```

详见 `docs/linux_deployment.md`。

### 8.3 部署目录结构

推荐部署结构：

```text
pilot_deploy/
  bin/
    pilot(.exe)
  client/
    index.html
  config/
    pilot_deploy.properties
    pilot_deploy.linux.properties
  models/
    a320_new_full.onnx
    tridet_a320.onnx
    best.onnx
  runtime/
    ffmpeg/
      ffmpeg(.exe)
    *.dll / lib/*.so*
  output/
  temp/
```

## 9. 测试和调试入口

当前 CMake 会构建以下目标：

| Target | 输出名 | 说明 |
| --- | --- | --- |
| `PilotApp` | `pilot` / `pilot.exe` | 主服务。 |
| `Tst` | `test_runner` | RTSP 播放相关测试。 |
| `TestFeatures` | `test_extractFeatures` | I3D 特征提取测试。 |
| `TestYolo` | `test_yolo` | YOLO 推理测试。 |

建议的最小验证顺序：

1. 启动 `pilot`，确认日志打印模型路径且 I3D 初始化成功。
2. 浏览器访问 `http://127.0.0.1:8080/`，确认页面可打开。
3. 用 LOCAL VIDEO 提交一个后端可访问的视频路径，验证能生成 `output/report_<id>.json`。
4. 用 LIVE STREAM 连接 RTSP，验证启动、WebRTC 画面、停止和报告生成。
5. 若调 WebRTC，优先查看后端 `[WebRTC]`、`[H264]`、`[EncodeThread]` 日志。

Git 注意事项：当前仓库在沙箱用户下会触发 `dubious ownership`，如果后续需要使用 git 命令，需要在实际开发用户环境中配置 safe.directory：

```powershell
git config --global --add safe.directory E:/pilot
```

## 10. 常见维护点

### 10.1 更换模型

需要同步核对：

- 配置文件中的模型路径。
- ONNX 输入/输出节点名。
- 输入尺寸、归一化方式、张量布局。
- 类别数 `NUM_CLASSES`。
- 报告动作名 `ACTION_NAMES`。
- YOLO 类别名和关键点数。

### 10.2 修改动作类别

至少修改：

- `service/config.hpp` 中 `NUM_CLASSES`。
- `service.cpp` 中 `ACTION_NAMES`。
- 训练/导出后的 TriDet 和 I3D logits 输出维度。
- 前端报告展示通常不需要改，因为它展示后端返回的 `action` 字符串。

### 10.3 端口或跨机器访问

修改 `config/pilot_deploy*.properties`：

```properties
host=0.0.0.0
port=8080
```

跨机器访问时确认防火墙、端口和浏览器访问地址。WebRTC 当前没有配置外部 STUN/TURN，复杂网络环境下可能需要补充 ICE 服务器或改用同网段部署。

### 10.4 WebRTC 黑屏或无画面

优先检查：

- 浏览器 offer 是否包含 H.264。
- 后端 `/webrtc/offer` 是否返回 answer。
- `H264Encoder` 是否找到 H.264 encoder。
- 日志中是否有 `h264_mp4toannexb BSF attached`。
- 是否有 WebRTC session，但编码线程没有收到帧。
- 防火墙或虚拟网卡导致 ICE candidate 不可达。

已有问题记录见 `docs/webrtc_problems.md`。

### 10.5 报告未生成

检查：

- 任务是否已经停止或视频是否读到结尾。
- `/get_task_status` 是否为 `finalizing` 或 `failed`。
- `output_dir` 是否可写。
- TriDet 初始化是否成功。
- `feature_queue.stop()` 后 `tridet_predict()` 是否正常退出。

### 10.6 内存和磁盘临时文件

算法帧队列使用 `HybridVideoQueue`：

- 内存中最多缓存一部分帧。
- 超出后落盘到 `temp_dir` 下的二进制文件。
- 队列析构时删除临时文件。

长视频或高并发任务会放大磁盘 IO 和临时目录占用，需要关注 `temp_dir` 所在磁盘空间。

## 11. 后续开发建议

1. 将 `service.cpp` 拆分。当前文件同时承担 HTTP 路由、WebRTC、视频读取、算法调度和报告生成，后续建议按职责拆为 `api`、`webrtc`、`pipeline`、`report` 等模块。
2. 补充接口级自动化测试。目前测试更偏单模块，HTTP 路由和完整 pipeline 缺少可重复测试。
3. 为模型配置增加版本字段。模型、类别名和预处理强耦合，建议在配置中显式记录模型版本和类别映射文件。
4. 完善错误码。当前很多输入错误返回 HTTP 200 或 500，前端主要看 `code` 字段，后续可以统一错误协议。
5. 统一前端资源管理。`client/index.html` 是单文件，便于部署但维护成本会逐渐升高；如果 UI 继续扩展，可考虑拆分或引入轻量构建流程。
6. 明确 GPU 并发策略。当前 `gpu_device_id` 默认单卡，`distribute_GPU()` 逻辑保留但配置范围为 0 到 0；多卡部署前需要实测和完善资源管理。

## 12. 新接手人员快速上手路线

1. 先读 `service/config.hpp`，理解运行配置和全局常量。
2. 再读 `service/service.hpp`，了解 `PilotWebServer` 持有哪些状态。
3. 按 `boot()` -> `set_camera_interface()` -> `launch_camera()` / `launch_local_video()` -> `extract_features()` -> `tridet_predict()` 的顺序读 `service/service.cpp`。
4. 对照 `client/index.html` 中的 `handleLaunch()`、`handleOffline()`、`startWebRTC()` 理解前后端交互。
5. 需要改模型时读 `feature/feature.cpp`、`feature/tridet.cpp`、`yolo/yolo_detector.cpp`。
6. 需要部署时直接按 `docs/windows_deployment.md` 或 `docs/linux_deployment.md` 操作。
