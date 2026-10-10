# Microduck 学习分支：在 Mac 上跑通最小 PPO 闭环

本分支的目标是**亲手体验真实 PPO 训练、保存、导出和仿真**。不需要 GPU、云账号或机器人硬件。学习入口独立注册为 `Mjlab-Learning-Stand-Microduck`；旧速度任务、原始 `.pt` 和 `walk.onnx` 保留。

这里的默认 72 个训练控制步只能证明流程运行和参数更新，**不能证明机器人已学会站立或走路**。所有新模型以 `LEARNING_ONLY_NOT_FOR_HARDWARE` 命名并携带禁止实机的标签。入口不会调用硬件程序，也不会覆盖部署目录。

## 两条命令

在此分支目录执行：

```sh
./learn.sh
```

它会依次完成：创建 CPU MuJoCo 环境 → 真正运行 RSL-RL PPO 的采样与梯度更新 → 保存并重载新 `.pt` → 导出 ONNX → 检查 ONNX → 用仿真真实观测比较 PyTorch/ONNX 输出 → 用**刚训练的新 ONNX**驱动物理仿真 48 步。默认无窗口。

查看刚生成的学习策略：

```sh
./learn.sh --view
```

打开终端显示的本地浏览器地址（通常 `http://127.0.0.1:8080`），使用查看器播放控件；结束时在终端按 Ctrl-C。浏览器入口复用 MjLab 的 Viser 查看器，模型是刚保存的学习 `.pt`，不是原附带步态。只有查看器需要保持进程运行，训练命令会自行退出。

首次启动可能花时间加载／编译物理计算内核；这个教学任务使用的是 MuJoCo Warp 的 CPU 执行设备，不能用它的速度推算云 GPU 训练速度。

## 训练里只有什么

| 项目 | 学习配置 |
| --- | --- |
| 任务 | 一个平地、固定零速度的站立任务 |
| 环境数量 | 1 |
| Actor / Critic | 分别一个 `51 → 64 → 32` 前馈网络；Actor 输出14个动作，Critic输出1个价值 |
| 观测 | IMU角速度3 + 重力3 + 关节位置14 + 关节速度14 + 上次动作14 + 固定命令3 |
| 三项奖励 | 直立 `+2`；零速跟踪 `+2`；动作变化平方 `-0.1` |
| PPO | 每轮24步，默认3轮，每轮2个优化epoch，1个mini-batch |
| 物理 | 复用已有机器人和BAM执行器；固定电压7.4V，无随机电压下降和命令延迟 |
| 复位 | 固定位置、固定关节初态；跌倒或超时自动复位 |
| 简化 | 无课程、推扰、领域随机化、观测噪声、历史堆叠或多技能；Critic使用相同观测 |

仍保留观察量归一化，导出时包含归一化计算。因此 ONNX 输入是原始51维观测，不应再重复归一化。PPO探索动作仍会随机采样，这属于学习算法本身；固定种子使实验可复查。

## 输出怎么读

每次运行在 `learning_artifacts/<UTC时间>/` 创建独立目录（Git忽略）：

- `LEARNING_ONLY_NOT_FOR_HARDWARE.pt`：新策略、价值网络、优化器及归一化状态的训练检查点。
- `LEARNING_ONLY_NOT_FOR_HARDWARE.onnx`：新策略的推理图，含禁止实机元数据。
- `result.json`：版本、种子、奖励、网络、实际训练步数、Actor权重改变量、导出检查、PyTorch/ONNX最大误差、真实物理步数、复位数和耗时。
- 失败时保留 `failure.txt` 和 `status: FAIL`，不把失败目录登记为最新成功结果。

`learning_artifacts/latest.json` 指向最近一次成功闭环，查看器据此加载。`status: PASS` 的意思是**学习闭环验证通过**，不是步态验收。

想多体验几轮，可以执行 `./learn.sh --iterations 10 --rollout-steps 100`。增加轮数并不保证更稳，也不会自动切换到专业任务。查看指定模型：`./learn.sh --view --checkpoint /绝对路径/LEARNING_ONLY_NOT_FOR_HARDWARE.pt`；只加载自己生成且可信的学习检查点。

## Python 环境

`learn.sh` 优先使用本目录 `mjlab_microduck/.venv/bin/python`；没有时复用旁边原教程的已安装环境 `../Microduck-build-tutorial/mjlab_microduck/.venv/bin/python`。它强制把本分支 `mjlab_microduck/src` 放到导入路径最前，避免误用原教程的editable代码。

也可明确指定：

```sh
TRAINING_PYTHON=/已安装MjLab环境/bin/python ./learn.sh
```

新机器按原训练项目的锁文件安装已有依赖：

```sh
cd mjlab_microduck
uv sync --locked
cd ..
./learn.sh
```

不需要安装新的RL库；继续使用项目已有MjLab、RSL-RL、PyTorch、MuJoCo、ONNX及BAM依赖。Mac CPU是流程演示路径；要达到专业步态训练规模，应使用独立专业分支和实际GPU基准。

## 验证与边界

入口会自动要求：Actor MLP权重确实变化、参数有限、训练实际步数匹配、保存的检查点可重载、ONNX checker通过、每个真实观测上的ONNX/PyTorch误差在容差内、ONNX驱动后物理状态有限且改变、原策略资产哈希不变。配置隔离检查位于 `mjlab_microduck/tests/test_learning_config.py`。

本地验证记录见 `docs/learning-verification.md`。尚未完成长期站立、多种子收敛、GPU训练速度或实机验证；这些不属于学习分支默认任务。
