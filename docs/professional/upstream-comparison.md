# 专业分支与官方训练仓库的差异核对

核对日期：2026-10-11。官方来源为 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)。本文区分源码纳入、入口覆盖和机器人能力验收，避免把“有任务代码”当作“已有成熟模型”。

## 1. 我们的训练仓库是什么

团队仓库是 [karekin/Microduck-build-tutorial](https://github.com/karekin/Microduck-build-tutorial)，专业路线位于 `codex/microduck-professional` 分支。它不是另写的一套训练引擎：`professional/upstream/microduck_rl` 完整保留官方训练源码，外层增加版本与哈希校验、独立训练工作目录、中文入口、CPU 前检、限定预算的 GPU 链路和团队文档。

教程历史训练目录保留学习价值，但专业训练使用官方独立环境与官方 PPO/BAM 配方；教程 51 维部署接口和官方 61 维接口不能混用。

## 2. 版本与文件核验

| 检查 | 结果 |
|---|---|
| 固定官方提交 | `ceca01706143d729305baa94cbd8ab990979e325` |
| 2026-10-11 官方 `develop` | 同一提交 |
| GitHub compare | `identical`；ahead=0、behind=0、total_commits=0 |
| 固定文件集合与 SHA-256 | 254 / 254 通过，无缺失、额外文件或哈希差异 |
| 官方源码修改 | 本次未修改 vendor |
| 统一训练入口覆盖 | 33 / 33 官方注册任务；原六个别名兼容 |

证据入口：[`develop` 提交 API](https://api.github.com/repos/pollen-robotics/microduck_rl/commits/develop)、[固定提交与 develop 对比 API](https://api.github.com/repos/pollen-robotics/microduck_rl/compare/ceca01706143d729305baa94cbd8ab990979e325...develop)、[`upstream.lock.json`](../../professional/upstream.lock.json)。执行 `python3 professional/run.py verify` 可复核本地文件。官方分支会继续变化，以上结论有日期边界；今后升级须重新固定版本和核验。

## 3. 差异分层与本次补齐

| 项目 | 固定官方源码 | 专业分支现状 |
|---|---|---|
| 13 个任务族、18 个基础注册任务 | 已有 | 完整纳入 |
| 15 个 Backlash 注册变体 | 已有 | 完整纳入；不是每个任务族都有变体 |
| BAM M6 电机模型与域随机化 | 已有 | 原样纳入，尚需实机参数校准 |
| HF Jobs 提交、检查点上传 | 已有 | 原样纳入；额外限定预算的首次链路已运行 |
| PPO + 专家 BC、对称性配置 | 已有 | 原样纳入；对称性默认是否开启按各任务配置判断 |
| 统一中文任务别名 | 无此团队外层 | 原来仅 6 个；本次补齐全部 33 个并增加 `catalog` |
| 官方升级与复现控制 | 上游持续开发 | 固定提交 + 254 文件 SHA-256 + 独立 checkout |
| 各任务成熟策略 | 上游有历史训练与部署说明 | 团队尚未独立训练并验收这些技能 |
| 与匹配运行栈的实机部署 | 在官方 `microduck` 仓库 | 尚未完成团队实机集成与验收 |

本次补齐的是入口覆盖、资产提示、回归测试和差异说明，不是又移植一份已经完整纳入的官方源码，也没有启动新的收费训练。

完整映射见 [专业路线 README](../../professional/README.md)，也可运行 `python3 professional/run.py catalog`。覆盖测试直接从固定 `tasks/__init__.py` 的注册声明读取任务集合，确认封装没有遗漏或虚构任务；无需导入 CUDA 训练依赖。

## 4. 不能遗漏的算法与硬件边界

### 专家蒸馏已经存在

[`tasks/distill.py`](../../professional/upstream/microduck_rl/src/mjlab_microduck/tasks/distill.py) 中 `PpoWithExpertBc` 继承 PPO：每次 PPO 更新后，对学生采集的观测再做行为克隆。跌倒帧由冻结起身专家指导，直立帧由冻结行走专家保护原技能；中间区域与无法由专家教授的命令交给 PPO。VelStand 默认启用该机制。因此“官方只有纯 PPO、没有教师学生”是不准确的。

默认教师检查点是官方 W&B 的 `69u48n8l/model_9750.pt` 与 `441tzs6d/model_3750.pt`，也支持本地匹配检查点。可访问性、模型许可、结构和归一化一致性须在训练前确认。外层 `--no-wandb` 仅关闭日志，不取消教师资产读取。

这类技能蒸馏不是 RMA。固定行走 actor 仍是 61 维观测输入、512/256/128 的前馈 MLP；没有在这些注册策略中发现已接入的 SAC、TD3、AMP 或历史观测适应模块。源码的动作历史缓冲用于延迟等机制，不能据此宣称 actor 已变为历史策略。

### 其他模型不等于其他强化学习算法

固定仓库还包含 `pickup/model.py` 的短时序 CNN：输入最近 50 帧、每帧 63 维特征，判断机器人是否被拿起。它是辅助分类模型，不是 63 维步态策略，也不是替代 PPO 的强化学习算法。不能因搜到时序网络就说机器人已用 RMA 或循环策略行走。

### 任务与真实能力有边界

- `GroundPick` 是嘴尖触地再回到站姿，不能据此宣称完成视觉抓取任意物体。
- `BallKick` 的 actor 不读取球位置；不是视觉找球与导航踢球系统。
- `rollers`、`swizzle`、`roller-*`、`spin` 使用被动轮脚模型；普通脚硬件不能直接部署。
- 只有官方实际注册的 15 个 Backlash 变体纳入入口。不能自行制造 `roulade-backlash`、`spin-backlash` 或 `roller-standup-backlash` 的任务 ID。
- ONNX 61 → 14 和归一化一致只是接口要求；不同任务还有命令、接触、阶段或姿态标志的语义要求。

## 5. 仍需推进的工作

1. **训练资产准备**：恢复任务先验证教师检查点与 warm start 流程，记录来源、校验值和归一化器；避免为了绕过下载而悄悄禁用 BC。
2. **逐任务烟雾与正式训练**：33 个入口可生成命令不等于 33 个都运行过。新增任务先做配置/NaN/导出检查，再按预算开展真实训练。
3. **独立评估**：固定未参与训练的场景和种子，测行走误差、跌倒/恢复成功率、冲击、电流、终态和视频；不同任务设不同标准。
4. **任务通用产物链路**：`gpu_pipeline.py` 与 `hf_hour.py` 仍固定为首次行走烟雾任务。全任务自动导出、评估和回收需后续适配，不能把首次流水线报告当作踢球或恢复验收。
5. **硬件验证**：先使用匹配官方运行栈进行 CPU ONNX 部署排练，再做吊架/低速/受控跌倒/电机温度与电流检查。稳定步态和实机安全仍未验证。

团队已有的 GPU 结果只证明 64 环境、5 轮 PPO 的 CUDA 训练 → 检查点 → 归一化 ONNX → 物理步进链路跑通，见 [GPU 首轮结果](../../professional/gpu-result-20261010.md)。这不证明正式步态收敛，也不证明上表中的复杂技能已被团队实现。
