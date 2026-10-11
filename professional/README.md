# Microduck 专业路线

第一次使用项目先阅读[团队上手操作手册](../docs/团队上手操作手册.md)，再执行本页的专业训练步骤。

本分支以 **Pollen Robotics 官方训练仓库**为训练基础。官方完整源码固定于 `ceca01706143d729305baa94cbd8ab990979e325`，放在 `upstream/microduck_rl/`，保留原许可、依赖锁与原始算法。`upstream.lock.json`记录254个文件的哈希。当前教程仍保留，作为51维部署接口的历史参考。

## 开始使用

在专业分支根目录：

```bash
python3 professional/run.py verify
python3 professional/run.py catalog
python3 professional/run.py prepare
cd professional/runs/official
uv sync --frozen
```

装好后回到分支根目录执行`python3 professional/run.py cpu-check`，检查官方61维观测、14动作与3次CPU物理步进。每次在独立运行目录保留日志；它不训练策略，也不能证明步态收敛。

`prepare`建立独立Git训练目录，避免官方HF打包器把教程根目录当成训练项目。它复制固定Git对象，不追随远端develop，不覆盖已有修改。

回到本分支根目录，先生成64环境、5轮的烟雾测试命令：

```bash
python3 professional/run.py train
python3 professional/run.py train --backend hf --namespace YOUR_ACCOUNT
```

默认仅显示命令。装好独立环境后加`--execute`才实际运行；HF需要账户授权与余额，会产生费用，默认60分钟超时。初次创建分支时没有提交收费任务；后续获批的一小时GPU链路已完成，见[GPU首轮结果](gpu-result-20261010.md)。日志使用本地TensorBoard，不向W&B上传。

小规模检查通过，再运行正式基线（默认1024环境、4000轮）。先测显存与吞吐，再决定是否提高到4096环境：

```bash
python3 professional/run.py train --profile baseline --seed 42
python3 professional/run.py train --profile baseline --seed 43
python3 professional/run.py train --profile baseline --seed 44
```

统一入口已覆盖固定官方注册的全部 **33 个任务：18 个基础任务和 15 个 Backlash 变体**。原来的六个别名保持兼容。用 `catalog` 查看完整任务 ID、是否需要轮脚、是否依赖专家检查点。每个命令分别记录任务、种子和实验名。

| 任务族 | 基础别名 | 可追加的后缀 |
|---|---|---|
| 行走 | `walk` | `-rough`、`-backlash`、`-rough-backlash` |
| 行走与恢复 | `recovery` | `-rough`、`-backlash`、`-rough-backlash` |
| 起身 | `standup` | `-rough`、`-backlash`、`-rough-backlash` |
| 坐下与站起 | `sitstand` | `-rough`、`-backlash`、`-rough-backlash` |
| 嘴尖触地后站起 | `ground-pick` | `-rough`、`-backlash`、`-rough-backlash` |
| 踢球 | `kick` | `-backlash` |
| 前滚翻 | `roulade` | 无 |
| 轮脚速度跟踪 | `rollers` | `-backlash` |
| 轮脚对称滑行 | `swizzle` | `-backlash` |
| 轮脚下蹲滑行 | `roller-crouch` | `-backlash` |
| 轮脚坡道滑行 | `roller-slope` | `-backlash` |
| 轮脚起身 | `roller-standup` | 无 |
| 轮脚原地旋转 | `spin` | 无 |

`rollers`、`swizzle`、`roller-*`、`spin` 使用轮脚模型；普通脚机器人不能直接使用这些策略。`ground-pick` 训练嘴尖触地动作，不是视觉识别后抓取任意物体；`kick` 的 actor 不读取球位置。

```bash
python3 professional/run.py train --task ground-pick
python3 professional/run.py train --task roulade
python3 professional/run.py train --task walk-rough-backlash --backend hf --namespace YOUR_ACCOUNT
```

这些命令默认仍只显示训练命令，任务已纳入入口不代表任务已训练或验收。源仓库官方导出路径位于训练目录的`scripts/export.py`，导出必须包含观测归一化：

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file /absolute/path/model_100.pt --device cpu --onnx-file output.onnx
uv run scripts/infer_policy.py --walking output.onnx --new-cmd-obs
```

## 恢复任务的教师资产前提

官方 VelStand 已启用 **PPO + 专家行为克隆（BC）**，不是纯 PPO 从零发现全部技能。`tasks/distill.py` 用冻结起身教师指导跌倒观测，用冻结行走教师约束直立观测，减少学会起身后遗忘行走；两次更新使用学生刚采集的观测。它保留 61 维 actor 接口，属于技能蒸馏，不等于依赖历史观测估计隐变量的 RMA。

默认教师来自官方 W&B：起身 `pollen-robotics/mjlab_microduck/69u48n8l` 的 `model_9750.pt`，行走 `pollen-robotics/mjlab_microduck/441tzs6d` 的 `model_3750.pt`。开跑 `recovery*` 前，需要确认这些检查点可访问，或在独立实验配置的 `bc_cfg` 中指定匹配架构与归一化器的本地 `checkpoint_path` 和 `anchor_checkpoint_path`。不得拿不明来源 ONNX 当作可恢复训练的检查点。

`--no-wandb` 只关闭训练日志上报，不会取消上述教师下载。当前统一入口也没有封装教师路径、warm start 和 resume 的全部参数；需要时在 `professional/runs/official` 按官方 `microduck_velstand_env_cfg.py` 的训练说明操作。跨任务 warm start 需明确使用 `MICRODUCK_WARM_START=1` 重置课程计数，并保证观测归一化兼容。

`hf_hour.py` / `gpu_pipeline.py` 是首次获批 L4 链路的固定行走烟雾测试（64 环境、5 轮），不是所有任务的完整自动评估器。新任务通过 `run.py train` 使用官方训练入口；任务专用的成功率、触地方式、冲击与恢复指标仍需独立验收。

## 能力演进与证据

阅读[来源目录与路线](../docs/professional/README.md)。训练能力来源于固定官方源码；相关仓库、帖子与论文只在被验证后才进入默认配置。采用状态应区分：已导入源码、已通过本地检查、已训练、已完成独立评估、已验证实机。

本分支与官方的源码、封装和验收差异见[官方差异核对](../docs/professional/upstream-comparison.md)。2026-10-11 核对时官方 `develop` 与固定提交一致；无需重复移植已纳入的任务、Backlash、BAM 或 BC。

建议第一批对照：官方原版行走、齿轮间隙变体、低速从静止启动、电压/摩擦/延迟鲁棒性、行走与恢复。每次只改变一个因素；种子42/43/44使用相同场景，保存每项奖励贡献、跌倒率、速度误差、侧漂、偏航、峰值电流、视频与模型来源。

## 部署边界

- 官方策略是61输入/14输出/50Hz；教程是51输入/14输出，不能直接替换`microduck/src/agents/walk.onnx`。
- 官方运行仓库目前只是固定版本参考，尚未引入教程运行程序；61维实际部署应使用匹配的官方运行栈，或经过单独验证的接口适配。
- `verify`证明源码完整；命令与配置检查不是GPU训练或实机能力证明。
- Python/依赖以官方独立`uv.lock`为准；不能把教程的虚拟环境当成专业路线已经安装完成。
- 本机已完成独立`uv sync --frozen`、训练CLI参数核对、初始8项工作流/接口回归、官方任务CPU初始化与3步物理检查；记录见`local-validation.json`。后续12项回归和L4 CUDA训练链路通过；稳定步态和实机仍未验证。
- 代码Apache-2.0，官方3D模型CC-BY-NC-SA-4.0，原始许可文件保留在固定源码内。
