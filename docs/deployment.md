# 部署与配置

先使用 README 的 Mock 启动流程确认界面可访问，再配置模型与设备。以下约定来自当前运行代码；相机、语音模型和底盘接口需要按自己的设备准备。

## 模型服务

Web 入口默认读取根目录 `model_config.json`。它只允许非敏感的模型名称、接口地址和协议；API Key 放在环境变量中。环境变量覆盖同名配置，`JAKA_MODEL_CONFIG` 可指定另一份 JSON。项目不自动加载 `.env`。

| 配置 | 用途 |
| --- | --- |
| `JAKA_AGENT_BASE_URL` / `JAKA_AGENT_MODEL` | Agent 的 OpenAI 兼容服务地址与服务端模型名称 |
| `JAKA_AGENT_PROTOCOL` | `json` 为文本 JSON 工具协议，`native` 为原生工具调用；需匹配服务能力 |
| `DASHSCOPE_BASE_URL` | 视觉与规划客户端的 OpenAI 兼容地址 |
| `QWEN_PLAN_MODEL` / `QWEN_VISION_MODEL` | 任务规划与视觉分析使用的服务端模型名称 |
| `JAKA_AGENT_API_KEY` / `DASHSCOPE_API_KEY` | 对应服务的凭据；本地无鉴权服务可使用占位值 |

`model_config.example.json` 是字段示例。实际服务提供的模型名未必是本地权重目录，请按部署端返回的名称设置。仓库不包含模型权重或推理服务器安装程序。

Linux 示例，先复制示例为自己的配置并填入实际模型名：

```bash
cp model_config.example.json model_config.local.json
export JAKA_MODEL_CONFIG="$PWD/model_config.local.json"
export JAKA_AGENT_API_KEY=not-needed
export DASHSCOPE_API_KEY=not-needed
python robot_web.py --host 127.0.0.1 --port 8080
```

PowerShell 使用 `$env:JAKA_MODEL_CONFIG = '配置文件的绝对路径'` 等方式设置变量。个人配置 `model_config.local.json` 已被 Git 忽略。

## 底盘与相机

| 配置 | 用途 |
| --- | --- |
| `JAKA_HOST` | 底盘地址，代码默认 `192.168.10.10` |
| `JAKA_PORT` / `JAKA_HTTP_PORT` | 底盘 TCP / HTTP 端口，默认 `31001` / `9001` |
| `JAKA_HEAD_CAM_SN` / `JAKA_HAND_CAM_SN` | 自己设备的头部与手部相机序列号 |
| `JAKA_WEB_GRAPH` | 场景图路径，默认 `zmq_scene_graph.json` |
| `JAKA_WEB_SLAM_IMAGE` | 本地 SLAM 图片路径 |
| `JAKA_SLAM_IMAGE_URL` / `JAKA_SLAM_YAML_URL` | 底盘地图图片与元数据地址 |
| `JAKA_MAPPING_SERVER` | 建图 ZeroMQ 地址，默认 `tcp://127.0.0.1:5560` |

相机采集模块还需要 `numpy`、`opencv-python`、`dashscope` 和适配设备/系统的 `pyorbbecsdk`。建图需要 `pyzmq` 及独立建图服务。SDK 应依据自己的 Orbbec 设备与系统架构安装；这些设备依赖不包含在最小网页依赖中。

辅助采集脚本 `jaka_step.py` 仍依赖设备侧采集程序，需通过 `--lumi-root` 指定包含 `jaka_step.py`、`agv.py` 和 `get_pos.py` 的 SDK 示例目录。它不是启动网页和运行离线测试的前提。地图绘图工具还需自行安装 `matplotlib`。

默认相机序列号、地图图片和底盘地址是开发设备配置，不能直接代表另一台机器人。先检查底盘状态、相机取帧和地图坐标，再确认执行任务。

## 地图标定

根目录 `zmq_scene_graph.json` 与 `slam.png` 用于示例界面。在新场地中替换成匹配的场景图和 SLAM 地图，确认分辨率、原点、旋转及每个导航点的可达性。

标定可在网页调整，保存为本地 `slam_calibration.json`。`examples/slam_calibration.example.json` 仅展示字段，其数值不适合直接控制实机。不要把旧场地标定复制到新场地运行；缺少元数据时的界面默认值只用于初始显示。

默认关闭启动时自动拉取底盘地图。需要时显式设置 `JAKA_SLAM_AUTO_FETCH=1`，并确认地图地址与现场一致。

## 语音与手机访问

`voice.py` 需要 `numpy`、`sounddevice`、`sherpa_onnx` 及系统音频组件。当前模型目录固定为：

- `/home/pi/voice/asr-paraformer`：Paraformer 中英识别模型。
- `/home/pi/voice/vits-piper-zh_CN-huayan-medium`：Piper 中文语音合成模型。

需要自行准备对应模型；若修改目录，调整 `voice.py` 的 `BASE`。手机录音还需要 FFmpeg，将上传音频转为 16 kHz 单声道 PCM；单次最多 30 秒、8 MB。浏览器录音需要 HTTPS 或本机 localhost，远程访问可使用受控 HTTPS 代理或安全隧道。

服务没有公网登录机制。建议监听 `127.0.0.1` 并置于受控访问入口后，或限制在受信任网络内。

## 运行数据与发布

会话数据库默认位于 `conversation_data/conversations.sqlite3`，可通过 `JAKA_CONVERSATION_DB` 更改；媒体根目录可通过 `JAKA_MEDIA_ROOT` 更改。任务录像默认启用，可设置 `JAKA_RECORD_TASK_VIDEO=0` 关闭。现场照片、录像和数据库均不纳入版本管理。

```bash
python tools/build_pi_release.py
```

输出 `dist/jaka-pi-runtime.tar.gz` 与 `dist/runtime-sha256.json`。运行包按 `deploy/raspberrypi/runtime-files.json` 的清单收集源码、网页、示例地图及图标，不包括模型权重、Python 环境、测试或演示视频。

更新设备前停止服务并备份现有运行目录及数据，解压运行包到部署目录，保留本机标定与业务数据，安装依赖后重启。检查 `/api/health`、地图和模型连接，再做现场任务验收。
