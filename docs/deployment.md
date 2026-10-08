# 部署与配置

先使用 README 的 Mock 启动流程确认界面可访问，再配置模型与设备。以下约定来自当前运行代码；相机、语音模型和底盘接口需要按自己的设备准备。

## 模型服务

Web 和规划命令按以下顺序选择配置：显式 `JAKA_MODEL_CONFIG` → 工作目录的 `configs/model_config.local.json` → 已有根目录 `model_config.json` → 包内默认配置。它只允许非敏感的模型名称、接口地址和协议；API Key 放在环境变量中。环境变量覆盖同名配置，`JAKA_MODEL_CONFIG` 可指定另一份 JSON。项目不自动加载 `.env`。

| 配置 | 用途 |
| --- | --- |
| `JAKA_AGENT_BASE_URL` / `JAKA_AGENT_MODEL` | Agent 的 OpenAI 兼容服务地址与服务端模型名称 |
| `JAKA_AGENT_PROTOCOL` | `json` 为文本 JSON 工具协议，`native` 为原生工具调用；需匹配服务能力 |
| `DASHSCOPE_BASE_URL` | 视觉与规划客户端的 OpenAI 兼容地址 |
| `QWEN_PLAN_MODEL` / `QWEN_VISION_MODEL` | 任务规划与视觉分析使用的服务端模型名称 |
| `JAKA_AGENT_API_KEY` / `DASHSCOPE_API_KEY` | 对应服务的凭据；本地无鉴权服务可使用占位值 |

完整的模型下载、推理环境安装、启动、SSH 转发和接口验收见[模型部署](models.md)。`deploy/server/model_deploy.py` 提供当前双模型的安装与启动代码，权重另行下载。`configs/model_config.example.json` 是字段示例；请按自己的服务配置模型名称，或使用部署脚本生成与权重路径一致的配置。

Linux 示例，先复制示例为自己的配置并填入实际模型名：

```bash
cp configs/model_config.example.json configs/model_config.local.json
export JAKA_MODEL_CONFIG="$PWD/configs/model_config.local.json"
export JAKA_AGENT_API_KEY=not-needed
export DASHSCOPE_API_KEY=not-needed
jaka-agent --host 127.0.0.1 --port 8080
```

PowerShell 使用 `$env:JAKA_MODEL_CONFIG = '配置文件的绝对路径'` 等方式设置变量。个人配置 `configs/model_config.local.json` 已被 Git 忽略。

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

相机采集模块可安装 `python -m pip install -e ".[vision]"`，另行安装适配设备/系统的 `pyorbbecsdk`。建图可安装 `.[mapping]`，还需独立建图服务。SDK 应依据自己的 Orbbec 设备与系统架构安装；这些设备依赖不包含在最小网页依赖中。双相机、机械臂 SDK 和上电准备见[硬件准备](hardware.md)。

采集命令 `jaka-collect`（`mapping/collector.py`）仍依赖设备侧采集程序，需通过 `--lumi-root` 指定包含 `jaka_step.py`、`agv.py` 和 `get_pos.py` 的 SDK 示例目录。它不是启动网页和运行离线测试的前提。地图绘图工具还需自行安装 `matplotlib`。

默认相机序列号、地图图片和底盘地址是开发设备配置，不能直接代表另一台机器人。先检查底盘状态、相机取帧和地图坐标，再确认执行任务。

## 地图标定

包内 `resources/maps/zmq_scene_graph.json` 与 `slam.png` 用于示例界面。现场地图放入 `$JAKA_DATA_DIR/maps/`（默认 `data/maps/`），或用 `JAKA_WEB_GRAPH` 与 `JAKA_WEB_SLAM_IMAGE` 指向实际文件，确认分辨率、原点、旋转及每个导航点的可达性。

标定可在网页调整，保存为数据目录下的 `slam_calibration.json`（已有根目录文件会继续复用）。`examples/slam_calibration.example.json` 仅展示字段，其数值不适合直接控制实机。不要把旧场地标定复制到新场地运行；缺少元数据时的界面默认值只用于初始显示。

默认关闭启动时自动拉取底盘地图。需要时显式设置 `JAKA_SLAM_AUTO_FETCH=1`，并确认地图地址与现场一致。

## 语音与手机访问

`src/jaka_agent/hardware/voice.py` 需要 `numpy`、`sounddevice`、`sherpa_onnx` 及系统音频组件。未设置环境变量时沿用旧设备目录：

- `/home/pi/voice/asr-paraformer`：Paraformer 中英识别模型。
- `/home/pi/voice/vits-piper-zh_CN-huayan-medium`：Piper 中文语音合成模型。

语音依赖可安装 `.[audio]`；用 `JAKA_VOICE_BASE` 或分别用 `JAKA_ASR_DIR`、`JAKA_TTS_DIR` 覆盖目录。模型下载与文件检查见[可选语音模型](models.md#6-可选语音模型)。手机录音还需要 FFmpeg，将上传音频转为 16 kHz 单声道 PCM；单次最多 30 秒、8 MB。浏览器录音需要 HTTPS 或本机 localhost，远程访问可使用受控 HTTPS 代理或安全隧道。

服务没有公网登录机制。建议监听 `127.0.0.1` 并置于受控访问入口后，或限制在受信任网络内。

## 运行数据与发布

`JAKA_DATA_DIR` 指定可写数据目录；默认是启动命令所在工作目录下的 `data/`。不要把数据写进安装包的 `src/` 或 `site-packages`。

| 数据 | 新安装默认位置 |
| --- | --- |
| 会话数据库与媒体 | `data/conversation_data/` |
| 现场照片与任务录像 | `data/web_captures/`、`data/web_videos/` |
| 新建地图与上传的 SLAM 图片 | `data/maps/` |
| 建图过程与轨迹 | `data/mapping_runs/`、`data/robot_tracks.json` |
| 标定与日志 | `data/slam_calibration.json`、`data/logs/robot_web.log` |

**已有本地数据的迁移**：不设置 `JAKA_DATA_DIR` 时，继续使用工作目录中已存在的 `conversation_data/`、`web_captures/`、`web_videos/`、`mapping_runs/`、轨迹和标定文件。已有根目录地图也会继续被发现。设置新的数据目录后，应在停服时把相关旧数据一起复制过去，并将现场地图放入其 `maps/` 子目录；原文件不会自动移动。

`JAKA_CONVERSATION_DB` 和 `JAKA_MEDIA_ROOT` 仍可分别覆盖数据库与媒体位置。任务录像默认启用，`JAKA_RECORD_TASK_VIDEO=0` 可关闭。

```bash
python tools/build_pi_release.py
```

输出 `dist/jaka-pi-runtime.tar.gz` 与 `dist/runtime-sha256.json`。运行包按 `deploy/raspberrypi/runtime-files.json` 的清单收集源码、网页、示例地图及图标，不包括模型权重、Python 环境、测试或演示视频。

更新设备前停止服务并备份现有运行目录及数据，解压运行包到部署目录，保留本机标定与业务数据，运行 `python -m pip install .` 安装运行包后，用 `jaka-agent` 重启。解压目录也可使用 `python robot_web.py` 兼容入口。检查 `/api/health`、地图和模型连接，再做现场任务验收。

## 命令入口

探索和采集命令需要先安装 `python -m pip install -e ".[vision,mapping]"`，再按硬件说明准备相机 SDK。最小安装只用于网页、Mock 和不依赖设备的规划入口。

| 命令 | 用途 |
| --- | --- |
| `jaka-agent --mock` / `python -m jaka_agent --mock` | 启动界面体验模式 |
| `jaka-plan cmds` | 打印底盘协议示例，不联网、不移动 |
| `jaka-plan` | 模型规划与 Mock 底盘执行演示，需要已配置的模型服务 |
| `jaka-explore --help` | 自动探索参数 |
| `jaka-mapping-bridge --help` | 建图桥接参数 |
| `jaka-collect --help` | 设备侧采集参数 |

`jaka-plan run`、`markers`、`voice` 及设备诊断子命令可能访问真实设备，使用前核对环境变量和参数。旧的 `python qwen_planner.py cmds` 仍可用。
