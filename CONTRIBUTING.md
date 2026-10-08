# 参与开发

建议先运行[任务回放](docs/replay.md)，了解 Agent 计划与机器人执行的边界，再选择技能、硬件适配或评估方面的改进。

## 开发与检查

需要 Python 3.11+；前端检查需要 Node.js。

```bash
python -m pip install -e ".[dev]"
jaka-agent --replay
python tools/run_offline_checks.py --require-node
python tools/evaluate_agent.py --mode replay
```

离线检查不得连接外部模型或真实机器人。设备 SDK、音频与建图依赖按使用场景单独安装，见部署文档。

## 新增机器人技能

以“到点查看”为最小参考：

1. 在 `src/jaka_agent/agent/skills.py` 定义 `Skill`：输入结构、资源要求、步骤、完成标准和失败规则。`skill_tools` 据此生成模型可调用的 `plan_*` 工具。
2. 在 `prepare_skill` 中验证输入并转换为执行器认识的计划；地图目标必须来自查询结果，不能把模型生成的名称或房号直接用作 `ann_id`。
3. 经 `tasks/cards.py` 的 `plan_skill` 创建待确认任务。新增步骤需要同步实现 `tasks/executor.py` 或独立业务模块中的执行分支。
4. 动作、观察和长请求接入 `tasks/runtime.py` 的取消机制，保留所有者检查、地图版本校验与结果持久化。
5. 增加缺输入、确认前不执行、失败停止、取消后不再观察等有实际意义的检查；可在回放中加入演示案例。

当前技能注册是代码内显式注册，尚不支持从外部目录自动加载插件。只增加模型提示词或技能卡片不会产生新的执行能力。

## 接入其他机器人或相机

底盘语义接口为 `hardware/navigation.py` 的 `NavDriver`。参考 `JakaTCPDriver`，实现移动、取消、位姿、状态和等待导航终态，并在 `tasks/manager.py` 的任务驱动构造与状态查询处接入；使用命令行执行时同步修改 `cli/planner.py`。返回的导航状态必须能被执行器正确识别；不可把“命令已发送”当成“到达成功”。

相机采集入口位于 `hardware/camera.py` 与 `hardware/service.py`；保持采集路径、媒体归属、时间信息和设备释放行为。新设备先用替身验证接口，再在独立场地做导航、停止、观察与确认验收。当前没有通用外部驱动插件发现机制。

## Issue 与 PR

Issue 请描述运行模式、Python/系统版本、复现步骤、预期与实际结果。涉及实机问题时补充任务状态和已脱敏日志。

PR 说明解决的问题、最终行为和验证结果；涉及模型或实机的内容分别记录离线验证与现场验证。服务器地址、密码、API Key、私钥、用户照片和现场运行数据留在本地忽略目录。保留第三方许可与必要署名；项目级许可证确定前，不要替项目自行添加许可声明。
