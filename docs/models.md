# 模型部署

本项目把推理留在 Linux GPU 服务器，机器人只运行网页、任务执行、相机和可选语音。当前配置为 **Qwen3.5-9B + MiniCPM-V-4.6，两个服务均使用 Transformers Serve**；Agent 使用文本 JSON 工具协议。旧的 Qwen vLLM 部署记录不作为本教程的默认方案。

公开代码不包含服务器地址、登录账号、密码、密钥或模型权重。文中的 `jaka-model-server` 是你在本机 SSH 配置中定义的别名。

## 需要哪些模型

| 模型 / 权重来源 | 用途 | 部署位置 | 是否必需 |
| --- | --- | --- | --- |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Agent 决策、地图查询与技能选择 | GPU，默认端口 8001 | 真实 Agent 必需 |
| [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) | 现场图片、参考图比对、巡逻与部分任务规划 | GPU，默认端口 8000 | 真实视觉任务必需 |
| [流式 Paraformer 中英模型](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/paraformer-models.html) | 麦克风指令与手机录音转写 | 机器人 CPU，sherpa-onnx | 语音功能可选 |
| [Piper 中文 huayan medium](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models) | 中文语音播报 | 机器人 CPU，sherpa-onnx | 语音功能可选 |
| [Cubify Anything](https://github.com/apple-aiml-research/ml-cubifyanything) 的 `cutr_rgbd.pth`、[CLIP ViT-H-14](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | 外部 BoxFusion 建图后端 | 独立 GPU 环境 | 在线建图可选；已有地图无需部署 |

Mock 网页不需要这些模型。权重与第三方软件分别遵循其上游许可；本仓库提供部署和接入代码，不重新分发权重。

## 1. 准备 GPU 服务器

使用 Linux x86_64、Python 3.11/3.12、可用的 NVIDIA 驱动、`python3-venv` 与 Git。先执行 `nvidia-smi` 检查可用 GPU、空闲显存及驱动，再按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 选择匹配的安装源。两个模型默认使用不同 GPU；没有第二张卡时可以分别启动验收，或在确认显存充足后调整分配。这里没有经过测量的最低显存保证，图片数量、分辨率、上下文和并发都会增加占用。

```bash
git clone https://github.com/wang1299/JAKA-Robot-Agent.git
cd JAKA-Robot-Agent
cp deploy/server/models.env.example configs/server.local.env
```

编辑 `configs/server.local.env` 中的权重根目录、GPU 编号、端口与可选 PyTorch 安装源。所有目录必须为绝对路径，支持 `$HOME`。该文件被 Git 忽略；模型部署脚本按普通 `KEY=VALUE` 解析它，不执行 shell 表达式。进程环境中的同名变量优先。

| 配置 | 默认 / 含义 |
| --- | --- |
| `JAKA_MODEL_HOME` | `$HOME/jaka-models`，权重与两个虚拟环境的父目录 |
| `JAKA_QWEN_MODEL` / `JAKA_MINICPM_MODEL` | 默认在 `$JAKA_MODEL_HOME/models/` 下分别存储 |
| `JAKA_QWEN_ENV` / `JAKA_MINICPM_ENV` | 默认在 `$JAKA_MODEL_HOME/envs/` 下，互相独立 |
| `JAKA_QWEN_GPU` / `JAKA_MINICPM_GPU` | `1` / `0`；通过 `CUDA_VISIBLE_DEVICES` 选择物理 GPU |
| `JAKA_QWEN_PORT` / `JAKA_MINICPM_PORT` | `8001` / `8000`；必须不同 |
| `JAKA_MODEL_DTYPE` | `bfloat16`；需 GPU 支持，可按设备调整 |
| `JAKA_TORCH_INDEX_URL` | 可选官方 PyTorch wheel 源；未设置时由 pip 默认源解析 |
| `JAKA_TRANSFORMERS_SPEC` | `transformers[serving]==5.12.1` |

5.12.1 来自当前客户端中记录的兼容版本；启动参数已对照[该版本官方 CLI 源码](https://github.com/huggingface/transformers/blob/v5.12.1/src/transformers/cli/serve.py)。当前仓库未保存原 GPU 服务器完整依赖锁，也未在本次整理中重新运行 GPU 推理。因此这里是可复现配置的起点，部署后必须完成第 5 步验收。不要把机器人环境的最小依赖文件用于推理服务器。

## 2. 安装环境、下载权重

以下命令在服务器仓库目录执行。安装脚本为两个模型分别创建环境，安装 PyTorch、Transformers Serving、Accelerate、PyAV、Pillow 与 Hugging Face Hub，运行 `pip check` 并检查 CUDA：

```bash
python3 deploy/server/model_deploy.py install --python python3.11
```

每个环境会保存 `requirements.resolved.txt`，用于记录实际安装版本。若修改了 `JAKA_MODEL_HOME` 或环境覆盖路径，下面的 Python 路径也要对应修改。

```bash
$HOME/jaka-models/envs/minicpm/bin/python deploy/server/model_deploy.py download
```

下载器先把 Hugging Face 模型版本解析为 commit SHA，再下载该版本，并在权重目录写入 `download-manifest.json`。首次下载可能耗时较长；网络中断后可重跑。若要使用确定的模型修订，分别执行：

```bash
$HOME/jaka-models/envs/minicpm/bin/python deploy/server/model_deploy.py download --role minicpm --revision <模型提交SHA>
$HOME/jaka-models/envs/qwen/bin/python deploy/server/model_deploy.py download --role qwen --revision <模型提交SHA>
```

下载需要访问官方模型源。如源要求登录，在服务器本地通过 `hf auth login` 配置，凭据不写进仓库或教程。可先部署单个模型：安装和下载命令均支持 `--role qwen` 或 `--role minicpm`。

已有权重时，设置 `JAKA_QWEN_MODEL`、`JAKA_MINICPM_MODEL` 指向完整目录即可，无需重复下载。目录中必须包括配置、tokenizer / processor 和权重分片；仅复制一个权重文件无法启动。

## 3. 启动两个模型服务

先查看将要运行的命令，此操作不加载模型：

```bash
python3 deploy/server/model_deploy.py command --role qwen
python3 deploy/server/model_deploy.py command --role minicpm
```

在两个终端分别执行：

```bash
python3 deploy/server/model_deploy.py serve --role qwen
```

```bash
python3 deploy/server/model_deploy.py serve --role minicpm
```

启动器绑定 `127.0.0.1`，固定使用本地权重目录，选择 `sdpa`，关闭编译、连续批处理与 reasoning。选定 GPU 后，进程内设备编号为 `cuda:0`。运行时设置 `HF_HUB_OFFLINE=1`，避免误填模型名后临时下载其他模型；首次加载仍可能较慢。

Transformers Serve 提供 `/health`、`/v1/models`、`/v1/chat/completions`。固定模型模式将请求交给预加载模型；客户端仍应填写相应服务端权重路径，方便与已有客户端保持一致。`/v1/models` 的缓存列表未必包含自定义目录，验收以实际聊天和图片请求为准。

需要长期运行时，编辑 `deploy/server/jaka-model@.service.example` 的用户、仓库路径和配置路径，另存为 `/etc/systemd/system/jaka-model@.service`。将本地配置复制到 `/etc/jaka/server.local.env`，仅允许服务用户读取，然后执行：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now jaka-model@qwen jaka-model@minicpm
sudo systemctl status jaka-model@qwen jaka-model@minicpm
journalctl -u jaka-model@qwen -n 80 --no-pager
journalctl -u jaka-model@minicpm -n 80 --no-pager
```

虚拟环境和权重目录必须让所选服务用户可读。模板不创建系统账号。没有 systemd 的容器可用终端或已有进程管理器运行前台命令。

## 4. 通过 SSH 连接机器人

先在**服务器**生成与部署路径一致的客户端配置：

```bash
python3 deploy/server/model_deploy.py client-config --output configs/model_config.generated.local.json
```

该命令拒绝覆盖已有文件。通过自己的安全传输方式将文件复制到**机器人**的 `configs/model_config.local.json`。其中模型名是服务器权重目录，不是机器人目录。文件中只有回环接口、模型名与协议；若改了本地转发端口，还需修改对应 `BASE_URL`。

在机器人用户的 `~/.ssh/config` 定义别名 `jaka-model-server`，由你在本机填写 `HostName`、`User`、`Port`、`IdentityFile`。实际值只保留本地。使用 SSH 密钥或代理，并先交互执行一次 `ssh jaka-model-server`，核对服务器主机指纹。隧道脚本使用 `BatchMode=yes` 和严格主机密钥检查，不读取或保存登录密码。

```bash
cp configs/tunnel.env.example configs/tunnel.local.env
bash tunnel.sh foreground
```

此命令将本地 `127.0.0.1:8000/8001` 转发到服务器对应的回环端口。默认别名已配置时无需填写真实主机地址。修改端口时，服务端端口、转发目标和客户端 JSON 必须一致。

另一个终端运行第 5 步检查。前台验证通过后可用 `bash tunnel.sh start` 启动自动重连；`status` 只代表管理进程存在，模型能否正常回答仍需接口检查。停止使用 `bash tunnel.sh stop`。日志默认写入本机 `$HOME/.jaka-model-tunnel.log`，不要直接上传包含实际连接信息的日志。

需要开机启动时，按设备用户名和实际路径编辑 `deploy/raspberrypi/jaka-model-tunnel.service.example`，安装为 `/etc/systemd/system/jaka-model-tunnel.service`，然后 `sudo systemctl enable --now jaka-model-tunnel`。安装该服务后，`tunnel.sh` 的管理命令会调用 systemd，避免重复占用端口。

机器人网页使用自己的应用环境：

```bash
python -m pip install -e .
export JAKA_MODEL_CONFIG="$PWD/configs/model_config.local.json"
export JAKA_AGENT_API_KEY=not-needed
export DASHSCOPE_API_KEY=not-needed
jaka-agent --host 127.0.0.1 --port 8080
```

`not-needed` 仅适用于本教程的本地无鉴权服务；有鉴权代理时从本机环境提供真实 API Key。客户端变量与优先级见[部署与配置](deployment.md#模型服务)。不要使用 `--mock` 验收真实模型调用。

## 5. 验收模型连接

在机器人仓库目录、安装了应用依赖的环境中执行：

```bash
python tools/check_models.py --config configs/model_config.local.json --health-only
python tools/check_models.py --config configs/model_config.local.json --output tests/artifacts/model-smoke.json
```

第二条会真实调用模型：Qwen 输出指定 JSON，MiniCPM 识别合成的纯红单图和红 / 蓝双图。图片为同尺寸正方形，与客户端多图预处理约定一致。脚本仅发送合成输入，不连接底盘或相机；结果文件不记录服务器 URL、密钥、模型目录或模型原始回答。可用 `--role qwen` / `--role minicpm` 单独检查，冷启动时可增加 `--timeout`。

这些检查只验证接口、JSON 与基本图片输入，不能证明寻物或迎宾准确率。随后应按[实机验收](testing.md#实机验收)测试真实场地和图片，并记录耗时、错误匹配和无法判断的情况。

## 6. 可选语音模型

在 Linux 机器人上安装音频依赖、PortAudio、ALSA 播放工具与 FFmpeg。Debian / Raspberry Pi OS 示例：

```bash
sudo apt-get install libportaudio2 libasound2-dev alsa-utils ffmpeg
python -m pip install -e ".[audio]"
export JAKA_VOICE_BASE="$HOME/voice"
python deploy/raspberrypi/download_voice_models.py --base "$JAKA_VOICE_BASE"
python deploy/raspberrypi/download_voice_models.py --base "$JAKA_VOICE_BASE" --check
```

下载器使用 sherpa-onnx 官方 Release，校验必需文件后才放入最终目录；已有完整模型会跳过下载，已有不完整目录会报错而不覆盖。会记录压缩包 SHA256；这个本地计算值用于追踪下载，只有使用独立可信的 `--sha256` 值校验时才提供额外的来源完整性验证。

```text
$JAKA_VOICE_BASE/
├── asr-paraformer/
│   ├── encoder.int8.onnx
│   ├── decoder.int8.onnx
│   └── tokens.txt
└── vits-piper-zh_CN-huayan-medium/
    ├── *.onnx
    ├── tokens.txt
    └── espeak-ng-data/
```

`JAKA_VOICE_BASE` 必须在启动应用的同一环境中设置；systemd 中可用 `Environment=JAKA_VOICE_BASE=实际绝对目录`。`JAKA_ASR_DIR` 与 `JAKA_TTS_DIR` 可分别覆盖两个子目录。未设置时保留旧设备的 `/home/pi/voice` 默认值。`--check` 只检查文件，不代表麦克风、音箱或识别已经验收。

```bash
arecord -l
aplay -l
jaka-plan voice
```

最后一条会启用真实语音与机器人规划路径，需要先完成硬件准备与任务执行条件检查。只体验网页和键盘输入时无需运行它。手机录音需 HTTPS 或本机 localhost；单次录音限制和音频转码说明见[部署与配置](deployment.md#语音与手机访问)。

## 7. 可选在线建图后端

现有语义地图可直接使用；只有要从 RGB-D 数据重建地图时才需要外部 BoxFusion 代码与模型。此仓库提供 `mapping/bridge.py`、采集与场景图适配代码，不把本地另一份完整 BoxFusion 工程当成本项目源码发布。

准备流程：

1. 在独立环境安装自己的 BoxFusion 后端及其上游依赖。该后端涉及 PyTorch、CUDA 扩展、Open3D 与 CLIP，不要与上面的 Transformers 5.12.1 环境混装。
2. 按 [Cubify Anything 官方说明](https://github.com/apple-aiml-research/ml-cubifyanything)取得 RGB-D 检测权重 `cutr_rgbd.pth`；下载 [LAION CLIP 模型](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)的 `open_clip_pytorch_model.bin`，放到后端配置指定目录。保留上游许可和版本记录。
3. 配置相机内参、深度单位、权重路径、输出目录，再独立验证后端处理一组 RGB-D 帧。不同后端版本的 CLI 与依赖应以该版本源码为准。
4. 用本仓库的桥接服务连接后端。真实后端必须理解会话与输出约定：`BOXFUSION_SERVICE`、`BOXFUSION_SESSION_ID`、`BOXFUSION_OUTPUT_DIR`、`BOXFUSION_PREVIEW_DIR`；原版离线 demo 未必实现这些接口。

先在独立测试环境验证桥接通信：

```bash
python -m pip install -e ".[mapping]"
jaka-mapping-bridge server --help
```

真实后端的接入形式：

```bash
jaka-mapping-bridge server --listen tcp://127.0.0.1:5560 \
  --boxfusion-dir /path/to/BoxFusion \
  --boxfusion-command "python <已适配的在线入口.py> <该版本启动参数>"
```

上面的入口是需要替换的接入位置。桥接器会管理会话和输出；部署端还需启用并核对 `deploy/server/map_graph_hook.py` 的场景图回传适配，操作前使用 `patch_map_graph_hook.py --help` 查看支持的参数。此部分尚不能承诺任意上游 BoxFusion checkout 开箱即用。机器人采集端的相机 SDK、位姿和 `JAKA_MAPPING_SERVER` 见[硬件与建图配置](deployment.md#底盘与相机)。

## 常见问题

| 现象 | 检查方向 |
| --- | --- |
| SSH 无法建立、主机密钥或权限错误 | 先在机器人用户下交互 `ssh jaka-model-server` 核对指纹和密钥；后台模式不会询问密码 |
| 本地端口已占用 | 查明已有模型或隧道进程，避免手工隧道与 systemd 同时运行 |
| `/health` 失败 | 分别检查模型进程、服务器回环端口、隧道与机器人回环端口 |
| 健康检查通过但聊天失败 | 检查完整权重、客户端模型路由、CUDA、实际服务日志；不能只看模型列表 |
| CUDA 不可用或显存不足 | 检查 torch 构建与驱动、GPU 编号、空闲显存，先降低图片 / 上下文 / 并发，分别启动两个模型 |
| 无法识别模型架构、CLI 选项不存在 | 核对所在虚拟环境和 `requirements.resolved.txt`；修改 Transformers 版本后重新验证所有启动参数 |
| JSON 测试失败或内容为空 | 确认 reasoning 关闭、令牌预算和超时足够；先验证文本请求，再验证 Agent 工具对话 |
| 多图出现形状错误 | 使用同尺寸正方形图片，检查是否使用当前客户端预处理；单图通过不代表双图通过 |
| 语音模型文件存在但不能识别或播放 | 检查 ONNX 运行环境、sherpa-onnx wheel 的系统架构、PortAudio、ALSA 和设备权限 |

## 发布前检查

```bash
python tools/check_deployment_privacy.py
python tools/run_offline_checks.py --require-node
bash -n tunnel.sh
```

隐私检查覆盖待发布的部署目录、配置模板和本文，阻止其中出现非回环 IPv4 地址及常见明文凭据，并拒绝纳入个人配置和权重。它是防误提交检查，不能替代人工核对，也不扫描或清除 Git 历史。实际服务器日志、SSH 配置、API Key、个人模型配置和下载权重始终保存在本地。
