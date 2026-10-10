# 第一轮一小时GPU链路

2026-10-10，用户授权Hugging Face L4 24GB，最多1小时算力、费用不超过$0.80。账户单次充值$10已单独获批并完成；自动充值关闭。本流程不额外开启端口或付费缓存。

## 已准备的链路

`gpu_pipeline.py`使用固定官方Microduck源码、64环境、5轮PPO、seed42，完成实际CUDA训练→完整.pt保存/重载→含归一化的61→14 ONNX→真实观测输出比较→新play环境48步物理评估。默认CUDA不可用即失败，不能将CPU自验算作GPU训练。每次保存源版本、软件/GPU身份、耗时、参数变化、样本吞吐、有限状态、倾角/高度/复位、模型哈希、日志和配置。

`hf_hour.py`默认只打包；`--submit`才调用一次HF Jobs。它固定L4、3600秒超时，使用实时API单价并拒绝超过0.80美元/小时的价格；源码和结果仓库为私有。未知提交结果不能盲目重试。任务完成或失败直接退出，停止算力使用。

```bash
# 在已装好的官方环境中使用缓存登录；不在聊天或报告中记录token。
professional/runs/official/.venv/bin/python professional/hf_hour.py
professional/runs/official/.venv/bin/python professional/hf_hour.py --submit

# 对已有任务查询或下载，不能重复提交
professional/runs/official/.venv/bin/python professional/hf_hour.py --receipt /absolute/path/receipt.json
professional/runs/official/.venv/bin/python professional/hf_hour.py --receipt /absolute/path/receipt.json --download
```

报告与模型下载必须通过必需文件、SHA256、路径范围和固定来源检查。只有`status=PASS`且`gpu_training_completed=true`，才能标记GPU链路通过。几轮PPO用于链路验证，不证明稳定行走；实机始终另行验收。

本地预检已完成：显式CPU1环境/1轮/48步完整链路PASS，约11秒；12项回归通过，并经独立审查修复小时价格舍入的预算问题。云任务进度与最终证据保存在本轮`professional/runs/hf-hour-*/receipt.json`及下载目录。

## 持续迭代方式

先建立固定GPU基线，再从`docs/professional/experiments.md`选一个因素。每轮登记预算、源码/模型来源、训练seed和独立评估seed；保存失败原件，比较低速起步、速度误差、跌倒率与动作后站稳。候选有可重复收益才晋级为产品默认策略。后续每次付费训练仍需明确该轮预算，不把首次一小时授权扩展为持续扣费。
