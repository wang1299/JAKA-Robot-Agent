# 测试与验收

离线测试检查任务契约、状态和接口行为，不调用真实模型、不驱动机器人。模型回答质量、现场识别效果和真实导航结果需要单独验收。

## 运行离线检查

验证环境为 Python 3.13。先安装开发依赖，并将 Node.js 加入 PATH：

```bash
python -m pip install -e ".[dev]"
python tools/run_offline_checks.py --require-node
```

也可通过 `--node /path/to/node` 指定 Node.js。未使用 `--require-node` 时，缺少 Node.js 会明确跳过前端检查；CI 要求两部分都通过。日志和 JSON 结果位于 `tests/artifacts/offline-checks/`，不纳入 Git。

当前公开测试集包含 **374 项 Python 测试与 4 组前端检查**：本地 Windows 环境中 370 项 Python 测试通过，3 项服务器协调器集成检查因缺少可选的服务器存档夹具而跳过，1 项 Linux SSH 进程检查因平台不同而跳过；4 组前端检查通过。CI 在 Linux 上运行 SSH 检查，使用假的 SSH 程序，不连接服务器。存档路径为本地 `.codex-tmp/map-update-evidence/`，不纳入公开仓库。

入口会去除因跨模块导入造成的重复测试，并在 Python 测试期间阻止非回环网络连接。测试使用模型替身与临时数据。

| 范围 | 检查内容 |
| --- | --- |
| Agent 与技能 | 工具协议、参数校验、技能选择、缺失输入与澄清 |
| 地图与导航 | 目标语义、房间/门口关系、地图版本、返程与朝向 |
| 执行状态 | 用户确认、取消、会话隔离、过期计划与重复执行 |
| 视觉任务 | 寻物恢复、巡逻比较、迎宾误报与网页照片确认 |
| 会话与媒体 | SQLite 持久化、会话恢复、媒体关联与访问约束 |
| 手机音频 | 上传格式、录音长度与转码接口 |
| 模型部署 | 配置优先级、GPU / 端口隔离、语音下载与解压、合成输入 HTTP 检查、隧道参数与脱敏输出 |
| 前端交互 | 会话与媒体界面、地图门口、巡逻点选择、迎宾照片确认 |

新增安装与迁移检查覆盖：从其他工作目录启动、新旧入口、网页静态资源、数据目录选择、现场地图和配置优先级。

GitHub Actions 在 Python 3.11 和 3.13 上运行相同入口，构建 wheel 和设备运行包，再使用安装后的包启动 Mock 与任务回放服务，核对合成任务卡和随包网页资源。

```bash
python -m pip wheel . --no-deps -w dist
python tools/smoke_web.py
python tools/smoke_web.py --legacy
# 安装 wheel 到独立环境后，再用该环境验证（不使用源码路径）
python tools/smoke_web.py --installed --python /path/to/environment/python
```

## Agent 与机器人任务契约

```bash
python tools/evaluate_agent.py --mode replay
python tools/smoke_web.py --replay
```

固定案例涵盖寻物观察反馈、迎宾照片确认、导航失败、缺少参考图和运行中取消。本地结果为 5/5；其成绩表示软件契约通过，预设决策和合成反馈不能证明模型或实机能力。`tests/test_agent_robot_replay.py` 另覆盖拒绝照片、过期确认、错误会话、地图更新、异常清理与反馈状态。完整操作和真实决策模式见[任务回放与评估](replay.md)。

## 手工模型评估

下列脚本不属于离线检查，会访问模型或真实输入，应先检查其配置和参数：

- `tools/check_models.py`：当前双模型的文本 JSON、单图与双图冒烟检查，步骤见[模型部署](models.md#5-验收模型连接)。
- `tests/test_service.py`：模型服务连通与功能检查。
- `tests/test_reference_match.py`、`tests/test_minicpm_reference_exact.py`：参考图比对评估。
- `tests/eval_*.py`：对话、导航、视觉或会话相关场景评估。
- `tests/check_mobile_live.py`：已部署网页与手机音频入口检查。

部分评估脚本需要自行准备图片、服务地址或现场数据。离线测试通过不代表这些评估已完成。

## 实机验收

先核对地图与标定、底盘状态和相机画面，再在可观察、可停止的条件下验证：

1. 地点导航：确认目标正确、到达后停留，只有要求返程时才返回出发点。
2. 寻物：使用现场目标与干扰物，记录找到、未找到和错误匹配的结果。
3. 迎宾：候选人物出现后必须等待主人网页照片确认；拒绝应继续等待，取消应结束任务。
4. 巡逻与查看：比较相同点位画面，无法看清时保留不确定性，不将历史图片当作当前现场。
5. 中断与恢复：运行中取消、切换地图或会话、重启服务后恢复历史，不重新执行旧任务。

现有四段演示为早期实机录像，迎宾含语音确认；最新版的网页确认与持久化功能以当前代码和对应测试为准，仍需要新版现场验收。
