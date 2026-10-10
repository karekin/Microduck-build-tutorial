# Microduck 专业分支：证据、基线与能力演进

核查日期：2026-10-10（Asia/Shanghai）。这是第一轮研究与可执行实验规划，不宣称已覆盖全部 GitHub，也不把别人的视频、训练曲线或论文结果当成本项目能力。

专业路线以 `professional/upstream/microduck_rl` 中固定的官方快照为训练基础，快照来源见 `professional/upstream.lock.json`。原教程仍供历史对照。先复现官方基线，再通过可测实验吸收社区观点；优先让低速起步、行走、恢复可靠，然后扩展动作和适应能力。

## 状态如何判断

- `research`：来源已阅读并整理，尚未接入。
- `proposed`：选为下一步实现/实验，尚未验证。
- `verified_integration`：只有完成接入与相应本地验证才能升级；CPU配置检查只证明配置，不证明GPU训练收敛或实机有效。

本目录的来源全部为 research/proposed。官方已实现某项功能，也不等于本机已训练或实机已验收该能力。实验执行和结果应写入独立运行目录，保存失败原始证据。

## 首先守住两个接口

当前教程策略通常为 `[1,51] → [1,14]`；官方新策略族采用 `[1,61] → [1,14]`，新增头部/躯干命令槽。不能直接把官方ONNX拷贝覆盖教程的walk模型。不要以填10个零的方式假定完成迁移：必须逐项核对顺序、单位、坐标系、关节映射、零位、动作比例、传感器编码视角和50Hz控制周期。

专业训练的export应走官方入口，将观测归一化嵌入ONNX。先用官方CPU推理程序做部署仿真，再决定是否适配旧教程runtime。历史网络或循环网络还需要明确历史缓存/隐状态的复位和策略切换行为，不能悄悄改变61维运行协议。

## 推进顺序

| 阶段 | 先获得什么 | 吸收哪些来源 | 放行条件 |
|---|---|---|---|
| P0 | 官方版本与观测/动作契约可追溯 | S01、S02、S05、S06、S24 | CPU检查、短CUDA冒烟、导出数值一致性 |
| P1 | 平地速度基线、无推扰低速起步 | S15、S23 | 从静止起步独立评估，多seed不退化 |
| P2 | BAM随机化与齿轮间隙A/B | S03、S22、S24 | 记录实际参数分布；基线/Backlash只改变一个因素 |
| P3 | 速度+恢复，再逐个站起/坐站/拾取/踢球 | S01、S17 | 完成动作后站稳，切回行走无异常 |
| P4 | 粗糙地形与更强扰动 | S07、S16 | 检查课程实际生效且train/play配置独立 |
| P5 | 历史信息、教师学生、RMA/A-RMA | S06、S10、S12、S19、S20 | 确认部署缓存/隐状态契约及推理时延 |
| P6 | AMP等动作先验 | S11、S21 | 有合法可执行参考动作，稳定性不退化 |
| P7 | SAC/TD3和第二仿真器 | S08、S09、S13、S18 | 同任务、同预算、同评价标准才比较 |

[实验矩阵](experiments.md)规定独立变量、指标和预算。[来源账本](sources.json)记录日期、许可、能力证据、采用状态及接口风险。BAM/Backlash在官方已经实现；专业分支应先验证和校准，不重新拼一套不一致的实现。

## 三个近期问题值得先验证

1. [低速起步 #59](https://github.com/pollen-robotics/microduck_rl/issues/59)：作者报告外部推扰能掩盖从静止无法低速起步。建议把无推扰起步加入固定评价，再分别测试低速采样与奖励。这是历史快照的社区报告，尚未在本分支复现。
2. [rough课程/配置污染 #65](https://github.com/pollen-robotics/microduck_rl/issues/65)：包含无需GPU的配置复现方法。先检查固定官方快照是否已修复；只看到terrain level升高不足以证明难度课程生效。
3. [踢球后倒下 #67](https://github.com/pollen-robotics/microduck_rl/issues/67)：高踢球奖励可能掩盖身体倒下。动作验收应包含动作后站稳率。社区FastSAC成功报告不能证明SAC必然比PPO优越，先排查任务奖励与终止条件。

## 实验与费用边界

在Mac上做来源/配置检查、CPU推理和结果分析；需要CUDA的训练在可用Linux NVIDIA环境执行。每个候选先用64个环境/5次迭代检查配置，再按显存/吞吐决定并行规模。不要把小规模冒烟的吞吐直接外推为收敛时间。云训练只在显式启动命令后发生；本轮没有购买GPU、提交HF Jobs或上传模型。

先固定任务、seed、总交互步数、训练环境数量、评价环境、归一化、控制频率与基线快照。算法对照同时报告GPU小时和样本数。用训练总reward选模型不足以证明能力：至少查看跌倒率、速度误差、起步时间、动作后站稳、足滑和部署推理时间。

每次实验先填写最大GPU小时和单价，费用预算=计费时间×单价+存储/传输。单价以实例创建页为准，本目录不把浮动云价写死。预算到达仍未通过就保存结果并停止该实验，不能以延长训练替代诊断。筛选seed建议42，候选晋级后用42/43/44重复；评估seed用独立集合，不能按结果挑seed。

## 来源许可

官方训练仓库的代码为Apache-2.0，STL模型为CC BY-NC-SA-4.0；固定快照需保留原许可与署名。Microban、RMA关联代码和旧AMP代码有不同许可，不因进入来源列表而自动复制到项目。第三方仓库声称模型均为Apache时，也以原模型发布方声明为准。论文与issue正文许可未核对的项标为unknown，当前只写摘要/策略建议，不搬运代码、模型或大数据。

## 初始来源目录（24项）

| ID | 可点来源 | 可吸收内容 | 状态 |
|---|---|---|---|
| S01 | [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) | 固定官方训练基线，优先平地速度与恢复 | proposed |
| S02 | [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) | 先对齐模型清单、接口与部署仿真 | research |
| S03 | [Rhoban/bam](https://github.com/Rhoban/bam) | 校准XL330摩擦、电压与延迟，不替换为理想PD | proposed |
| S04 | [Rhoban/microban](https://github.com/Rhoban/microban) | 学习小型Dynamixel机械设计与系统辨识；不复用关节策略 | research |
| S05 | [mujocolab/mjlab](https://github.com/mujocolab/mjlab) | 固定框架版本，复用配置检查与独立评估 | proposed |
| S06 | [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) | 固定训练器；以后增加教师学生或历史网络对照 | research |
| S07 | [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym) | 借鉴课程、噪声、推扰、执行器与评估；不引入旧IsaacGym栈 | research |
| S08 | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | 需要复杂传感器/场景时才启动仿真迁移 | research |
| S09 | [Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) | 独立sim2sim验证；先比较关节和接触一致性 | research |
| S10 | [Improbable-AI/walk-these-ways](https://github.com/Improbable-AI/walk-these-ways) | 借鉴命令条件化行为与课程，不复制四足权重 | research |
| S11 | [escontra/AMP_for_hardware](https://github.com/escontra/AMP_for_hardware) | 先获取Microduck可执行参考动作，再接判别器与动作先验 | research |
| S12 | [antonilo/rl_locomotion](https://github.com/antonilo/rl_locomotion) | 先历史观测，再教师特权信息/适应模块 | research |
| S13 | [Rishitha2628/microduck-kick](https://github.com/Rishitha2628/microduck-kick) | 优先复用独立评估设计，后来才对照SAC | research |
| S14 | [mshooter/microduck_academy](https://github.com/mshooter/microduck_academy) | 学习无实机的部署接口回归，保留第三方模型许可核对 | research |
| S15 | [Walking policy won't start slow walk from stand still](https://github.com/pollen-robotics/microduck_rl/issues/59) | 建立无推扰从静止起步回归，再单独改变低速采样 | proposed |
| S16 | [Rough terrain: generator runs in random mode, and play cfgs mutate the shared terrain config used for training](https://github.com/pollen-robotics/microduck_rl/issues/65) | 在固定快照上CPU复现，先检查当前代码是否已修复 | proposed |
| S17 | [BallKick: PPO can converge to "kick then collapse" at ~40-57° tilt, under the 70° fell_over threshold](https://github.com/pollen-robotics/microduck_rl/issues/67) | 评估踢完站稳/高度/倾角，先修奖励与终止再讨论换算法 | proposed |
| S18 | [BAM actuator models for IsaacLab](https://github.com/Rhoban/bam/issues/19) | 迁移前核查BAM是否已有可用接口，不能假定支持 | research |
| S19 | [RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034) | 教师特权信息与历史观测适应；先验证双足适用性 | research |
| S20 | [Adapting Rapid Motor Adaptation for Bipedal Robots](https://arxiv.org/abs/2205.15299) | 蒸馏后以实际估计器输入微调基础策略；不要照搬四足RMA | research |
| S21 | [AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control](https://arxiv.org/abs/2104.02180) | 针对Microduck创建动作数据与重定向，再测自然性/稳定性 | research |
| S22 | [Extended Friction Models for the Physics Simulation of Servo Actuators](https://arxiv.org/abs/2410.08650) | 用XL330实测轨迹挑选模型与校准DR范围 | research |
| S23 | [Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://arxiv.org/abs/1804.10332) | 保持紧凑可测观测，优先修复实际物理差异 | research |
| S24 | [Microduck RL AGENTS.md sim2real playbook](https://github.com/pollen-robotics/microduck_rl/blob/develop/AGENTS.md) | 把约束写入实验入口和回归；以固定vendor版本为准 | proposed |

## 增量维护

后续每次研究优先阅读官方新提交、具体issue/PR与有原始实验的社区仓库。新来源补齐URL、核查日期、版本、许可、适用机器人、可测假设和成本；未核实内容写unknown。先去重并识别与当前任务的关系，再加入候选实验。只有完成本项目对照实验才能宣称能力改善。此目录是首轮可维护账本，不创建定期爬取或收费训练任务。
