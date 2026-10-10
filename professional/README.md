# Microduck 专业路线

本分支以 **Pollen Robotics 官方训练仓库**为训练基础。官方完整源码固定于 `ceca01706143d729305baa94cbd8ab990979e325`，放在 `upstream/microduck_rl/`，保留原许可、依赖锁与原始算法。`upstream.lock.json`记录254个文件的哈希。当前教程仍保留，作为51维部署接口的历史参考。

## 开始使用

在专业分支根目录：

```bash
python3 professional/run.py verify
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

任务别名：`walk`、`walk-backlash`、`recovery`（行走+跌倒恢复）、`standup`、`sitstand`、`kick`。每个命令分别记录任务、种子和实验名。源仓库官方导出路径位于训练目录的`scripts/export.py`，导出必须包含观测归一化：

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file /absolute/path/model_100.pt --device cpu --onnx-file output.onnx
uv run scripts/infer_policy.py --walking output.onnx --new-cmd-obs
```

## 能力演进与证据

阅读[来源目录与路线](../docs/professional/README.md)。训练能力来源于固定官方源码；相关仓库、帖子与论文只在被验证后才进入默认配置。采用状态应区分：已导入源码、已通过本地检查、已训练、已完成独立评估、已验证实机。

建议第一批对照：官方原版行走、齿轮间隙变体、低速从静止启动、电压/摩擦/延迟鲁棒性、行走与恢复。每次只改变一个因素；种子42/43/44使用相同场景，保存每项奖励贡献、跌倒率、速度误差、侧漂、偏航、峰值电流、视频与模型来源。

## 部署边界

- 官方策略是61输入/14输出/50Hz；教程是51输入/14输出，不能直接替换`microduck/src/agents/walk.onnx`。
- 官方运行仓库目前只是固定版本参考，尚未引入教程运行程序；61维实际部署应使用匹配的官方运行栈，或经过单独验证的接口适配。
- `verify`证明源码完整；命令与配置检查不是GPU训练或实机能力证明。
- Python/依赖以官方独立`uv.lock`为准；不能把教程的虚拟环境当成专业路线已经安装完成。
- 本机已完成独立`uv sync --frozen`、训练CLI参数核对、初始8项工作流/接口回归、官方任务CPU初始化与3步物理检查；记录见`local-validation.json`。后续12项回归和L4 CUDA训练链路通过；稳定步态和实机仍未验证。
- 代码Apache-2.0，官方3D模型CC-BY-NC-SA-4.0，原始许可文件保留在固定源码内。
