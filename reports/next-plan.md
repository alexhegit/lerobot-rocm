# 下一批 ROCm 训练验证计划

> 已完成（2026-11-18）。全部 PASS，结果见 `smoke-report.md` 第 7 节。下文保留原计划。

供确认后再执行。不改 LeRobot 源码，不重建 `lerobot-rocm`，不提交。仍在 DOUni 上用现有镜像 `88a8506e6197`。

本计划不包含：g05 CoT、MolmoAct2 LoRA、flux3、推 Hub、WebUI 发起训练、多机。

## 共同规则

- 容器启动方式与 smoke 相同：`--ipc=host --shm-size 16g`，设备 `/dev/kfd` `/dev/dri`，`--group-add 44 --group-add 992`，`MIOPEN_FIND_MODE=2`，挂载 outputs 和 Hugging Face 缓存。
- `policy.device=cuda`，`push_to_hub=false`，`trackio.enable=false`，`episodes=[0]`，`batch_size=1`，`log_freq=1`。
- 单项超时 5400 秒。超时记 TIMEOUT，删掉该容器，继续下一项。
- 失败记入结果后继续下一项。恢复训练的种子失败时，跳过恢复，不拿别的 checkpoint 顶替。
- 日志追加到 `run-next.log`，结果追加到 `results.txt`。不覆盖旧日志。
- 每项结束后删除该任务的 `*.safetensors`、`*.distcp`、`training_state`。第 1 项的种子 checkpoint 保留到恢复任务结束。
- 开始前和每项之前看磁盘。空闲低于 80 GB 时，先删已经完成任务的大权重。不删 DeepSeek、`/opt/rocm-7.0.2`、`lerobot-rocm`、Hugging Face 原始 blob。
- 队列串行。8 卡任务占用 `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。

预计 2–3 小时。π0 编译若吃满 90 分钟，总时长会到约 4 小时。

## 1. 从 checkpoint 恢复

策略：DiT。数据：`lerobot/aloha_sim_transfer_cube_human`。1 卡。

1. `dit-resume-seed`：`steps=2`，`save_freq=2`，`env_eval_freq=0`，`num_workers=0`。保留 checkpoint。
2. `dit-resume`：同一输出目录，`--resume=true`，`--config_path` 指向 `checkpoints/000002/pretrained_model/train_config.json`，`--steps=4`。

通过：

- 种子退出码 0，且存在 `checkpoints/000002`。
- 恢复退出码 0，日志出现 step 3 和 step 4，并写出 `checkpoints/000004`。
- 恢复后的 step 从 3 开始，而不是从 1 重新计数。

## 2. 8 卡

先 DDP，再 FSDP。两张都用 aloha episode 0，`steps=2`，`save_freq=2`，`env_eval_freq=0`，`num_workers=0`。

1. `diffusion-ddp8`：`accelerate launch --num_processes=8`，不加 `--parallelism.*`。
2. `act-fsdp8`：同一 launcher，加 `--parallelism.dp_shard=8`。ACT 2 卡 FSDP 已经通过，这一项只提高分片数。

通过：退出码 0，有 loss，checkpoint 在；`training_step.json` 里 `dp_world_size=8`。DDP 的 `dp_replicate=8`，FSDP 的 `dp_shard=8`。

lingbot 不做 8 卡 DDP。它单卡峰值约 110 GB，8 份副本放不下。

## 3. 训练中的环境评估

策略：DiT。1 卡。数据仍是 aloha episode 0。FSDP 不能在训练中做环境评估，所以这一项只做单卡。

`steps=2`，`save_freq=2`，`env_eval_freq=2`，并加上：

- `--env.type=aloha`
- `--eval.n_episodes=1`
- `--eval.batch_size=1`
- `--eval.max_episode_steps=2`

默认 `n_episodes=50`、Aloha 每局最多 400 步，这里把评估收成 1 局 2 步。看的是评估循环能跑完，不看成功率。

通过：退出码 0，训练 loss 仍在，日志里出现评估结果，进程没有停在评估阶段。镜像若缺 Aloha 环境依赖，记 FAIL 并停止这一项，不重建镜像。

## 4. DataLoader 与更长步数

策略：DiT。1 卡。同一 aloha episode。

`steps=20`，`save_freq=20`，`log_freq=5`，`num_workers=2`，`env_eval_freq=0`。worker 启动方式保持代码默认的 `spawn`。

通过：退出码 0，日志有 step 5、10、15、20，存在 `checkpoints/000020`，worker 没有挂死。只记录 loss 变化，不设下降门槛。

## 5. π0 打开 torch.compile

策略：π0。1 卡。数据：`lerobot/svla_so100_pickplace` episode 0。其余参数与已经通过的 `pi0-1` 相同，只把 `--policy.compile_model=true`。

`steps=2`，`save_freq=2`，`env_eval_freq=0`，`num_workers=0`。

通过：退出码 0，有 loss，checkpoint 在。第一步会包含编译，允许它占用大部分 90 分钟。编译报错就记 FAIL，不改编译参数再试，也不继续做编译后的 DDP。

## 执行顺序

1. `dit-resume-seed` → `dit-resume`
2. `diffusion-ddp8` → `act-fsdp8`
3. `dit-eval`
4. `dit-workers`
5. `pi0-compile-1`

每项结束更新 `tmp/rocm-train-smoke-plan.md` 的下一节，并删掉该项大权重。队列结束后确认没有容器残留。
