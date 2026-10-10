# 学习分支本地验证记录

验证日期：2026-10-10（中国时间）。环境为本机macOS ARM64，无CUDA GPU；复用原教程已安装Python，导入代码来自学习分支。

执行 `./learn.sh`，默认1环境、3轮×24控制步、48步导出策略评估。最终版本完整闭环通过，用时 **11.94秒**；PPO采样及更新阶段 **2.00秒**。这是已有依赖与内核缓存的本机测量，不是冷安装时间或稳定步态训练时长。

| 检查 | 实测结果 |
| --- | --- |
| 真实训练 | 72个环境控制步；RSL-RL PPO进行了梯度更新 |
| Actor参数 | 三个线性层的权重与偏置均改变，最大绝对变化约0.005796 |
| 模型结构 | Actor 51→64→32→14；Critic 51→64→32→1 |
| 检查点 | 新 `.pt` 保存并重载成功，独立学习任务标签 |
| ONNX | Checker通过，输入1×51、输出1×14 |
| 导出一致性 | 48个真实仿真观测上比较，最大绝对误差2.980232e-8 |
| 新ONNX驱动仿真 | 48控制步，0.96秒仿真时间；动作、奖励、qpos有限，轨迹中qpos最大变化0.564001 |
| 行为限制 | 评估发生1次自动复位，不能据此声称站立／行走成功 |
| 原模型 | 原 `velocity.pt` 与 `walk.onnx` 的SHA256均不变 |
| 隔离检查 | 2个配置测试通过：旧速度任务不受学习配置修改影响；无隐藏前进采样或特权Critic |
| 静态检查 | Python编译、shell语法、Git空白检查通过 |
| 查看器 | `./learn.sh --view` 可加载新检查点，Viser完成初始化，127.0.0.1本地HTTP页面可读；测试进程随后关闭 |

软件版本：MjLab 1.3.0、RSL-RL 5.0.1、PyTorch 2.14.1、ONNX 1.23.2、BAM 1.0.0。已有锁文件的干净安装尚未另行复验；版本以每次 `result.json` 实际记录为准。

最终闭环原始报告在 `learning_artifacts/20261010T005941965531Z/result.json`，最新成功索引在 `learning_artifacts/latest.json`，二者作为本地实验数据被Git忽略。失败试跑保留在各自时间目录，未覆盖原报告，也未写入最新成功索引。

尚未验证：浏览器交互与画面验收、长期站立与多随机种子收敛、云GPU吞吐、实机。未安装独立Ruff或类型检查器，本次静态验证使用Python编译和shell/Git检查，没有添加依赖。

## 本分支独立安装与查看器404专项复验

2026-10-10补验：在学习分支自身新建 `.venv`，使用原锁文件执行 `uv sync --frozen --python 3.12` 后，`./learn.sh` 的真实72步PPO闭环通过，父代理记录本轮总耗时 **67.57秒**。此测量与上面的复用环境11.94秒分开保留，不能把两者当成相同安装／缓存条件。

首次查看器检查使用系统Xcode CPython 3.9.6，代码为 `urlopen('http://127.0.0.1:8080', timeout=5)`，曾返回 `HTTP Error 404: NOT FOUND`。该失败事实保留，不能只凭终端显示服务启动就计为页面验证通过。

专项复验确认：**该404来自系统Python继承的代理与无末尾斜杠的URL组合，查看器客户端文件没有缺失**。没有为此修改学习入口、MjLab、Viser、锁文件或系统代理设置。

| 请求条件 | 实测结果 |
| --- | --- |
| Xcode Python 3.9.6，默认代理，URL为 `http://127.0.0.1:8080` | 重现404，响应服务为Python 3.12 / websockets 16.1.1 |
| 同一解释器、同一URL，显式 `ProxyHandler({})` | 200，HTML 818656字节 |
| 同一解释器、默认代理，URL加末尾 `/` | 200，HTML 818656字节 |
| 学习分支Python 3.12，默认／禁代理／浏览器User-Agent | 根路径、`/index.html`、带查询参数根路径均200 |
| 新环境独立端口18081，裸Viser 1.1.1 | 根页面和gzip响应均可读 |

系统Python的代理“存在”且127.0.0.1“未绕过”；检查没有输出代理地址或值。另在独立端口18082使用临时HTTP探针观察接收路径：系统默认代理＋无末尾 `/` 的请求到达时已是非根路径；禁用该次请求的代理或添加 `/`，接收路径都为 `/`。Viser会将非根路径按客户端文件路径查找，找不到时返回404，符合捕获结果。请求头、HTTP服务器根目录、UV文件链接及MjLab/mjviser代码检查没有发现查看器资源问题。

新环境的查看器HTML SHA256为 `8c05b8bfea1ed54773a7616d1d868ebf0b45055ff8da8fc1603ccb2a7858f079`。Chrome可打开页面并显示 `Connected`；该浏览器同时显示 `WebGL unavailable`，因此本轮仍不声明3D画面或播放交互验收通过，也没有修改用户的浏览器加速设置。

建议始终使用终端显示的完整本地地址（包含末尾 `/`），并使用训练环境的Python检查。若要排除本地代理影响，仅对这一次检查显式禁用代理：

```sh
mjlab_microduck/.venv/bin/python - <<'PY'
from urllib.request import build_opener, ProxyHandler
opener = build_opener(ProxyHandler({}))
with opener.open('http://127.0.0.1:8080/', timeout=5) as response:
    page = response.read()
    assert response.status == 200 and b'<!doctype html>' in page[:100].lower()
    print(response.status, response.headers.get('Content-Type'), len(page))
PY
```

这条检查应在 `./learn.sh --view` 服务仍运行时执行。HTTP页面可读只证明客户端入口服务正常；3D渲染与播放应另行验证。所有本次自建临时服务均在诊断完成后关闭，旧模型未改。
