# 后续计划

> 已完成（2026-11-18 至 11-20）。MolmoAct2 LoRA、g05 CoT、WebUI 发起训练 PASS；flux3 在当前 `main` 上 BLOCKED，用上游 PR #5213 PASS；推 Hub 不做；多机待后续。结果见 `smoke-report.md` 第 8–10 节。下文保留原计划。

供确认后再执行。已完成的 smoke、resume、8 卡、DataLoader、π0 compile、OSMesa 评估不再重跑。

仍用 DOUni 上的 `lerobot-rocm:88a8506e6197`。不改 LeRobot 源码，不重建镜像，不提交。容器启动参数与之前相同。单项超时 5400 秒。失败记入 `results.txt` 后继续下一项。日志追加到 `run-followup.log`。每项结束后删除该任务的大权重。空闲低于 80 GB 时先清已完成权重。不删 DeepSeek、`/opt/rocm-7.0.2`、镜像和 Hugging Face 原始 blob。

安装额外包只发生在当次 `--rm` 容器里。安装后若 `torch.version.hip` 消失，或 torchvision 不再带 `+rocm`，立刻停掉该项。

## 1. MolmoAct2 LoRA

单卡，再视结果决定 DDP。数据与上次 fft 成功时相同：SO100 episode 0，相机 top→cam0、wrist→cam1。本地配置继续剥掉镜像不认识的字段，但 `train_mode_vlm=lora`，不再写成 fft。

容器启动后先执行 `uv pip install 'torchao>0.16'`。安装完检查 torch 仍是 ROCm 构建，再跑 `steps=2`。

通过：退出码 0，有 loss，checkpoint 在，日志显示走了 LoRA/peft，而不是 fft。单卡失败则不做 DDP。torchao 把 torch 换成 CUDA 构建，或 peft 仍报版本不符，就记 FAIL 并停止，不升级镜像里的其他包。

## 2. flux3

先在空容器里执行 `python -c "from flash_attn_interface import flash_attn_func"`。

- 导入成功：再跑单卡 2 步。数据用该 checkpoint 自己的动作维，不套 SO100。`steps=2`，`push_to_hub=false`。通过标准与其他 smoke 相同。单卡通过后再考虑 2 卡，本项默认只做单卡。
- 导入失败：尝试在当次容器里安装不替换 torch 的 flash-attn。90 分钟内装不上或编译失败，记 BLOCKED 并停止。不从源码反复试编译参数，不做多卡。

## 3. g05 打开 CoT

上次 g05 通过是因为关掉了 language recipe。这次只补数据，不改策略代码。

从 `lerobot/svla_so100_pickplace` 的 episode 0 做一份本地副本，给样本加上常量字段 `atomic_task`。然后用上次通过的 `g05_so101` 参数，但恢复默认的 `use_language_recipe=true`、`predict_cot=true`。1 卡，`steps=2`。

通过：不再出现 `atomic_task` 的 KeyError，退出码 0，有 loss，checkpoint 在。tokenizer 仍缺别的字段，记 FAIL 并停止，不继续猜字段。

## 4. 推送到 Hub

这一项等你给出私有仓库名再跑。未给仓库名就跳过，不新建公开仓库。

用 DiT、aloha episode 0、`steps=2`，只把这一次的 `--policy.push_to_hub=true` 指到你给的私有 repo。token 继续只走缓存文件，不写进命令、日志和总结。

通过：训练退出码 0，Hub 上该 repo 能看到这次 checkpoint。上传完成后删除本地大权重。

## 5. 多机

先探测，探测不过就不训练。

看 DORobot 和 DOUni 是否互通、是否有可用的 RCCL/RDMA 路径、两边是否都能启动同一份 `lerobot-rocm`。没有跨机高速链路，或两台机器的 Docker 网络打不通，记 BLOCKED。

探测通过才跑一个 2 进程任务：DOUni 1 卡 + DORobot 1 卡，策略用 DiT，`steps=2`。通过：退出码 0，`dp_world_size=2`，两台机器都参与了计算。

## 6. WebUI 真正发起一次训练

只做一条最小路径，仍在 `~/Repo/lerobot-webui`。

本机页面点一次开始，经 SSH 在 DOUni 上跑与已通过的 DiT 单卡相同的 2 步命令。页面能看到日志，停止等于 `docker rm` 那个容器。没有暂停。token 不出现在命令预览、页面和 localStorage。默认不推 Hub。

通过：从页面启动的容器退出码 0，checkpoint 写在既有输出目录，页面能看到 loss 日志。不把 WebUI 放进 LeRobot 仓库，也不提交。

## 顺序

1. MolmoAct2 LoRA
2. flux3
3. g05 CoT
4. 推 Hub（有仓库名才跑）
5. 多机探测；通过才训练
6. WebUI 发起一次 DiT

每项结束更新 `tmp/rocm-train/smoke-report.md`。
