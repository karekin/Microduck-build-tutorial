# 专业实验矩阵与放行规则

日期：2026-10-10。以下全部为proposed实验；预算是建议上限，不是训练时长承诺或已启动任务。

## 共同基线

运行清单至少包含：分支/commit、官方快照hash、lock文件hash、GPU/驱动、软件版本、任务、模型文件hash、checkpoint来源、seed、num_envs、控制频率、训练交互步数、GPU小时、完整配置、训练/评估日志、视频及导出数值检查。把参数和结果保存到运行目录，不覆写原始失败结果。

筛选seed=42，晋级seed=42/43/44；评估seed建议1001/1002/1003。固定种子不能消除GPU非确定性，应报告重复实验波动。先把评估集冻结，再看候选结果。

| ID | 一次只改变什么 | 对照和评价 | 初始GPU预算上限 | 放行标准/主要风险 | 来源 |
|---|---|---|---|---|---|
| E00 | 不改官方任务 | CPU配置、关节顺序、DR重置、模型许可；64env/5iter CUDA冒烟 | 0.5h | 无NaN，参数分布真实变化，checkpoint/export可读；不要求走稳 | S01/S24 |
| E01 | 官方平地Velocity基线 | 固定命令网格、无推扰起步与带推扰独立报告 | 筛选2h，复验共6h | 先建立实测基线；上游1–2h估计不是当前GPU承诺 | S01/S15 |
| E02 | 低速起步采样比例 | 从静止分别输入0.05/0.10/0.15/0.20/0.30m/s，推扰关闭 | 筛选2h | 起步时间/30s速度MAE改善；跌倒率无明显增加 | S15 |
| E03 | 低速奖励宽度 | 保持E02其他配置；每个速度档报告结果 | 筛选2h | 避免靠站着获取较高奖励；高速及原地站立不退化 | S15/S23 |
| E04 | BAM friction_scale范围 | 相同摩擦标称值；验证每环境实际scale及reset后不累加 | 筛选2h | 验证DR不是对零做倍乘；CPU部署仿真用同BAM | S03/S22/S24 |
| E05 | Backlash模型 | 官方基础/Backlash双任务，其余参数相同 | A/B共4h | 相同encoder视角、servo选择、reward；记录训练代价与held-out误差 | S01/S24 |
| E06 | 单一电压/延迟DR范围 | 按实机测得范围，分别评价低电压或延迟变化 | 单因素2h | 防止无限扩大DR导致全部任务退化 | S03/S23 |
| E07 | 切换VelStand官方任务 | 与Velocity对照行走，固定前趴/后趴/坐姿恢复集 | 筛选3h | 恢复成功率、恢复时间、行走退化、舵机碰撞；训练任务变化不是单一算法优势 | S01 |
| E08 | rough训练配置修复（如需） | 先CPU复现#65，再记录原/修配置地形分布 | CPU0h；训练A/B共4h | train/play不共享可变配置；难度与课程参数实际对应 | S16 |
| E09 | 踢球奖励站稳门控或终止条件（二选一） | 512回合，无/有推扰分开；倾角<30°且高度>站立80%作为初始站稳定义 | A/B共4h | 站稳率与球速共同比较，阈值需验证合理性 | S17/S13 |
| E10 | 一种新增动作任务 | 先坐站，再地面拾取；每次仅一个任务 | 每任务3h | 完成动作、回到站立、再切走路；phase/posture命令不能当普通恒定命令发布 | S01/S02 |
| E11 | 5帧历史MLP | 相同PPO与任务；增加仅历史缓存 | 筛选3h | 延迟held-out改善；ONNX缓存/reset/切换、CPU推理p99需验证 | S06/S19 |
| E12 | 教师学生适应 | 先冻结教师再训练学生，评估未知摩擦/载荷 | 6h筛选上限 | 部署禁止读取仿真特权状态；估计误差影响基础策略，参考A-RMA | S12/S19/S20 |
| E13 | AMP奖励项 | 有Microduck可执行参考动作后再开始 | 6h筛选上限 | 数据许可和重定向先通过；自然性评分与稳定性不能互相替代 | S11/S21 |
| E14 | PPO与FastSAC | 相同任务/观测/reward；分别按GPU小时与样本数对照 | 每算法4h | 至少3seed晋级；无同预算同指标不宣称优势；replay显存需测 | S13/S17 |
| E15 | 第二仿真器 | 同一模型静态、摆动、关节阶跃、接触测试后才跑策略 | CPU/配置先行；GPU最多2h试跑 | BAM/Backlash接口可用，关节语义一致；sim2sim不能替代实机 | S08/S09/S18 |

上限是各实验独立预算；不代表应一次全部跑。E00→E01优先，首轮只选择E02或E04一个问题。晋级复验另填写预算，不自动扩大。训练reward曲线只用于诊断，不能作为唯一成功标准。

## 评价与实机门槛

先评价官方固定基线，随后冻结每项指标的容忍区间。初始建议：候选至少改善主目标，在独立评估中跌倒率不增加超过2个百分点、速度MAE不恶化超过5%，动作后站稳率独立报告。百分比是待基线校准的工程建议，不是论文结论。多个seed报告均值、范围及每次结果，保留置信度局限。

部署前必须保存导出前/后同一输入输出差异、输入格式、控制周期和CPU推理p50/p95/p99。循环策略必须增加reset和策略切换回归。实机验证分为支撑站立、低速平地、恢复/动态动作，需具备停止方式并记录电压/电流/温度、关节与IMU日志；纯CPU或GPU仿真结果不能标注为实机PASS。

## 结果模板

```json
{
  "experiment_id": "E02",
  "status": "proposed",
  "source_ids": ["S15"],
  "baseline_commit": null,
  "candidate_commit": null,
  "changed_factor": "low_speed_command_sampling",
  "training_seed": 42,
  "evaluation_seeds": [1001, 1002, 1003],
  "gpu_hour_cap": 2.0,
  "gpu_hours_used": null,
  "cloud_hourly_price": null,
  "interaction_steps": null,
  "metrics": null,
  "evidence_paths": [],
  "decision": "not_run",
  "limitations": ["No GPU experiment or real robot validation yet"]
}
```

状态只有proposed→running→evaluated→accepted/rejected；accepted表示对当前明确范围达标，不宣称所有速度/地面/硬件都有效。源码接入通过测试可标verified_integration，能力收益另由实验结果判定。
