# LeRobot 0.7 ROCm 训练 smoke 报告

更新：2026-11-23。数据来自 DOUni `/data/lerobot/outputs/rocm-train/results.txt` 和各批次日志。文中的 `tmp/rocm-train/` 是验证时在 LeRobot 检出里的工作目录；本仓库里报告在 `reports/`（仿真评估的 `eval_info.json` 在 `reports/eval/`），脚本在 `scripts/`（分布式和夜间部分在 `scripts/dist/`），上游原文在 `upstream/`，日志在 `logs/`。主机地址、内部路径已替换为占位。

## 当前状态

训练都在 DOUni 的 8× MI300X 上完成，容器是按 `docker/Dockerfile.rocm` 构建的 `lerobot-rocm`（镜像 `88a8506e6197`）。没有改 LeRobot 源码，也没有重建镜像。挡住训练的问题都用启动参数、本地配置或一次性运行时补丁绕过，见第 6 节和各节说明。

通过标准：单卡（或标注的多卡）`steps=2`，有 loss，第 2 步 checkpoint 写完。不评估策略质量。

| 类别 | 模型 | 单卡 | 多卡 | 节 |
|---|---|---|---|---|
| 策略 | Diffusion | PASS | DDP 2 / DDP 8 / FSDP 2 | 5、7、13 |
| 策略 | ACT | PASS | DDP 2 / FSDP 2 / FSDP 8，DDP 和 FSDP resume、换分片 resume | 5、7、13 |
| 策略 | DiT（`multi_task_dit`） | PASS | DDP 2 / 4、FSDP 2 / 4、HSDP 2×2 / 4×2，特性测试见第 13 节 | 5、7、13 |
| 策略 | SmolVLA | PASS | DDP 2 / 4 / 8、FSDP 2、HSDP 2×2 | 5、13 |
| 策略 | X-VLA、EVO1、VLA-JEPA、VQ-BeT、TD-MPC | PASS | DDP 2 / FSDP 2 | 5、11.2、13 |
| 策略 | π0 | PASS（含 `compile_model`） | DDP 2 / FSDP 2 / FSDP 8 / HSDP 2×4 | 5、7、13 |
| 策略 | π0.5、π0-FAST、EO1 | PASS | DDP 2 / FSDP 2 | 5、13 |
| 策略 | GR00T N1.7 | PASS | DDP 2 / DDP 8 / FSDP 2 | 11.2、13 |
| 策略 | MolmoAct2（fft 与 LoRA） | PASS | DDP 2 / DDP 8 / FSDP 2 | 5、9、13 |
| 策略 | DM05、LingBot-VA（LoRA） | PASS | DDP 2 / FSDP 2 | 11.2、13 |
| 策略 | WALL-OSS 0.5（`wall_oss_05`）、FastWAM、Cosmos3 | PASS | DDP 2 / FSDP 2；FSDP 8（WALL-OSS 0.5、FastWAM、Cosmos3 Edge） | 5、13 |
| 策略 | LaWAM、LingBot-VLA v2 | PASS | DDP 2 / FSDP 2 | 5 |
| 策略 | G05（含 CoT）、FineART-VLA、WALL-OSS flow（`wall_x`）、Long-WAM | PASS | DDP 2；FSDP 失败（层内混合 dtype 或绕过 `forward` 调用子模块，见 13.3） | 5、8、11.2、13 |
| 策略 | FLUX3（LoRA） | 当前 `main` BLOCKED；上游 PR #5213 PASS | PR #5213 DDP 2 / FSDP 2 | 10、11.2、13 |
| 奖励模型 | SARM、reward_classifier | PASS（训练） | — | 11 |
| 奖励模型 | Robometer、TOPReward | 零样本推理 PASS；RA-BC 打分 + SmolVLA 加权训练 PASS | — | 11.1 |
| RL | HIL-SERL（Gaussian actor + SAC） | PASS | — | 11 |

训练特性：从 checkpoint 恢复、训练中仿真评估（MuJoCo，OSMesa 渲染）、`num_workers=2`、`torch.compile`、bf16 混合精度、PEFT LoRA 都已通过，见第 4、7 节。分布式特性（梯度累积、HSDP、fp16 / bf16 / FSDP 混合精度策略、CPU offload、DDP 选项、分布式 resume、换分片 resume、DCP checkpoint 格式、多卡评估）见第 13 节。

性能与效果（第 14 节）：多卡训练务必设 `AMD_CPU_AFFINITY=0`，否则 HIP 把所有 rank 和 DataLoader worker 限制在 8 个 CPU 上（ACT 8 卡 706 → 4122 smp/s）；卷积策略挂持久 MIOpen 缓存（首步 464 → 27.5 s）。ROCm 上按官方配方训的 ACT 仿真成功率 66%，官方检查点 76%，区间重叠。

未完成：多机训练（两台机器都没有 InfiniBand，待后续）。推 Hub 不做（smoke checkpoint 不值得上传）。

上游：已提交 PR [#5432](https://github.com/huggingface/lerobot/pull/5432)、[#5433](https://github.com/huggingface/lerobot/pull/5433)，在 [#5213](https://github.com/huggingface/lerobot/pull/5213) 留言 ROCm 结果，提交 NATTEN issue [SHI-Labs/NATTEN#354](https://github.com/SHI-Labs/NATTEN/issues/354)。第二批：PR [#5471](https://github.com/huggingface/lerobot/pull/5471)、[#5472](https://github.com/huggingface/lerobot/pull/5472)、[#5473](https://github.com/huggingface/lerobot/pull/5473)，issue [#5474](https://github.com/huggingface/lerobot/issues/5474)、[#5475](https://github.com/huggingface/lerobot/issues/5475)、[#5476](https://github.com/huggingface/lerobot/issues/5476)，在 [ROCm/rocm-systems#6712](https://github.com/ROCm/rocm-systems/pull/6712#issuecomment-7088991627) 留言。见第 12、14.3 节。

数值：ACT、SmolVLA 的 GPU fp32 推理与 CPU 一致（max_abs < 1e-5），LIBERO 成功率偏低不是 ROCm 数值问题（14.4）。

## 1. 硬件环境

| 角色 | 机器 | 用途 |
|---|---|---|
| 跳板机 | 本机 `<jump-host-ip>` | SSH 发起端。WebUI 预览也在这台机器上，没有参与本次训练 |
| 单卡机 | `ssh DORobot` | 最早的 ACT 小模型单卡 smoke |
| 8 卡机 | `ssh DOUni`，主机名 `<gpu-node-hostname>` | 本报告里的全部批量 smoke |

DOUni 加速器：8× AMD Instinct MI300X VF，GFX `gfx942`，SKU `M3000108`。系统盘 `/dev/vda1` 2.0 TB。报告收尾时已用约 1.9 TB，空闲约 107 GB。

训练时没有单独的 ROCm 设备名。容器里仍使用 `--policy.device=cuda`，用 `HIP_VISIBLE_DEVICES` 选卡。多卡 smoke 使用 2 张卡。另有一次更早的 ACT 8 卡 DDP，输出在 `/data/lerobot/outputs/smoke-rocm-ddp8`，不在这份 `results.txt` 里。

## 2. 软件环境

### 代码

- 本地检出：`v0.7.0` 之后一个提交（版本字符串 0.7.1），分支 `rocm-train`，从本地 `main` 拉出。LeRobot 源码没有改动，分支上只有 `tmp/rocm-train/` 下的报告和脚本。
- DOUni 上的检出 `/data/lerobot` 仍是 `main`。运行中的代码以镜像内 `/lerobot` 为准，因为 Dockerfile 使用 `COPY . .`。主机上换分支不会改变已运行的镜像。

### 镜像

按仓库默认方式构建，没有改 Dockerfile：

```bash
docker build -f docker/Dockerfile.rocm -t lerobot-rocm .
```

| 项 | 值 |
|---|---|
| 标签 | `lerobot-rocm:latest` |
| 镜像 ID | `88a8506e6197` |
| 构建时间 | 2026-11-09 |
| 体积 | 81.3 GB |
| 基座 | `rocm/primus:v26.4`（面向 gfx942 / gfx950） |
| 安装范围 | `LEROBOT_EXTRAS=all` |
| Python | 3.12 |
| torch | `2.12.0+rocm7.14.0a20260608`，HIP `7.14.60850` |
| torchvision | `0.27.0+rocm7.14.0a20260608` |
| numpy | `2.4.6` |

Dockerfile 会把基座里的 torch、torchvision、torchaudio、numpy、triton 版本写成约束，避免 `uv` 从 PyPI 换上 CUDA wheel。构建末尾会检查 `torch.version.hip` 存在，且 torchvision 版本串含 `+rocm`。

镜像内与本次相关、但没有升级的包：`torchao 0.15.0`、`peft 0.19.1`。PEFT LoRA 要求 `torchao > 0.16`，所以凡是用 LoRA 的任务（MolmoAct2 LoRA、LingBot-VA、FLUX3）都在当次 `--rm` 容器里执行 `uv pip install 'torchao>0.16'`（装到 0.18.0），装完检查 `torch.version.hip` 仍在。

### 容器启动

每次任务一台新容器，`--rm`，输出和 Hugging Face 缓存挂进容器：

```bash
docker run --rm --ipc=host --shm-size 16g \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 44 --group-add 992 \
  -e HIP_VISIBLE_DEVICES=... \
  -e MIOPEN_FIND_MODE=2 \
  -v /data/lerobot/outputs:/outputs \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  lerobot-rocm \
  bash /outputs/rocm-train/jobs/<name>.sh
```

- `44`、`992` 是当时主机上 `video`、`render` 的数字 gid。`--group-add render` 在这台机器上不能用。
- `MIOPEN_FIND_MODE` 使用数字 `2`。字符串 `FAST` 不能阻止查找。
- Hugging Face token 只放在挂载的缓存文件里，不写进脚本、日志或命令行。
- 单个任务超时 5400 秒。超时后删除该容器并继续下一项。

## 3. 测试用例

目标是确认「能开始训练、能算 loss、能把第 2 步 checkpoint 写完」，不是测最终策略质量。

统一参数：

- `steps=2`，`save_freq=2`，`log_freq=1`
- `batch_size=1`，`num_workers=0`
- `env_eval_freq=0`（不做环境评估）
- `policy.push_to_hub=false`，`trackio.enable=false`
- 数据集只取 `episodes=[0]`
- π 系列 `compile_model=false`

通过标准：

- 进程退出码 0
- 日志里有 step loss
- 存在 `checkpoints/000002/pretrained_model/model.safetensors`，或 FSDP 的 `*.distcp` / `training_step.json`
- 多卡时 `training_step.json` 的 `dp_world_size` 等于进程数（本次为 2）
- 单卡通过后才跑对应的 2 卡任务

数据集：

| 数据 | 用在 |
|---|---|
| `lerobot/aloha_sim_transfer_cube_human` episode 0 | 第一批视觉策略（diffusion、ACT、DiT） |
| `lerobot/svla_so100_pickplace` episode 0 | SmolVLA 以及多数语言策略。状态/动作维 6。相机 `observation.images.top`、`observation.images.wrist` |
| `lerobot/libero` episode 0 | fastwam。状态 8 维，动作 7 维 |
| 本地补写的 q01/q99 | 写进上述 SO100 数据集的 `meta/stats.json`，没有上传 Hub。π0.5、g05、wall-oss 使用 |

并行：

- 单卡：直接 `lerobot-train`，任务名后缀 `-1`
- DDP：`accelerate launch --num_processes=2 $(which lerobot-train)`，不加 `--parallelism.*`，后缀 `-ddp2`
- FSDP：同一 launcher，加 `--parallelism.dp_shard=2`，后缀 `-fsdp2`

Accelerate 只作为启动器。没有使用 accelerate YAML。未设置 `LEROBOT_ALLOW_ACCELERATE_ENV`。

## 4. 覆盖的 LeRobot 训练特性

本次实际走到的路径：

- `lerobot-train` 从 Hub 预训练权重或本地目录加载策略
- 单卡训练，`policy.device=cuda`（ROCm 上即 HIP）
- 2 卡 DDP（`dp_replicate=2`）
- 2 卡 FSDP（`dp_shard=2`），包括带 `_fsdp_wrap_modules` 的策略
- bf16 mixed precision，以及「fp32 主权重 + bf16 autocast」
- 第 2 步保存模型权重和 optimizer，并写出 `training_step.json`
- `push_to_hub=false`
- 只训练一个 episode
- 图像观测（数据集里的视频帧）
- `rename_map`、相机槽位、embodiment / state_slots / action_slots
- QUANTILES 归一化（用本地 q01/q99）
- 关闭语言 recipe / CoT 后继续训练（g05）
- 从本地改过的 `config.json` 加载（eo1、MolmoAct2、lingbot）
- 从第 2 步 checkpoint 恢复，并继续训练到第 4 步（DiT）
- 8 卡 DDP（diffusion，`dp_replicate=8`）和 8 卡 FSDP（ACT，`dp_shard=8`）
- `num_workers=2` 跑 20 步（DiT，worker 启动方式 `spawn`）
- π0 打开 `compile_model`
- 训练中的环境评估（DiT，Aloha，1 局、每局最多 2 步；无头渲染用 OSMesa）
- PEFT LoRA（MolmoAct2、LingBot-VA、FLUX3）
- G05 的 language recipe / CoT
- 奖励模型训练（SARM、reward_classifier）和 RA-BC 加权训练（`--sample_weighting.type=rabc`）
- 本地 WebUI 远程发起一次 DiT 训练

## 5. 测试结果

时间均为 UTC+8。下表是每个任务在 `results.txt` 里的最后一条。39 个训练任务全部 PASS。`prepare-fix2` 只是准备本地配置，没有 GPU 训练。

跑完的大权重已删除（`*.safetensors`、`*.distcp`、`training_state`）。日志、`results.txt`、config 和 tokenizer 元数据保留。

### 第一批（2026-11-10 19:39–20:10）

视觉策略，aloha episode 0；SmolVLA 用 SO100。

| 任务 | GPU | 结果 | 用时 | step1 → step2 loss |
|---|---:|---|---:|---|
| diffusion-1 | 1 | PASS | 665 s | 1.230 → 1.184 |
| diffusion-ddp2 | 2 | PASS | 664 s | 1.218 → 1.180 |
| act-fsdp2 | 2 | PASS | 246 s | 69.666 → 67.579 |
| dit-1 | 1 | PASS | 54 s | 1.204 → 0.986 |
| dit-ddp2 | 2 | PASS | 63 s | 1.209 → 0.988 |
| smolvla-1 | 1 | PASS | 80 s | 0.258 → 0.899 |
| smolvla-ddp2 | 2 | PASS | 91 s | 0.403 → 2.571 |

此前已通过、不在这张表里：DORobot 上的 ACT 小模型单卡；DOUni 上的 ACT 8 卡 DDP。

### 语言策略与后续批次

这些任务都用过失败重跑。表中是最后一次成功的用时和 loss。失败原因见附录。

| 任务 | GPU | 用时 | step1 → step2 loss | 备注 |
|---|---:|---:|---|---|
| xvla-1 / ddp2 | 1 / 2 | 143 / 125 s | 第一轮即通过 | `lerobot/xvla-base` |
| pi0fast-1 / ddp2 | 1 / 2 | 164 / 162 s | 20.096→12.618 / 17.546→11.633 | 补上 HF 登录后通过 |
| pi0-1 / ddp2 | 1 / 2 | 153 / 170 s | 0.011→0.532 / 0.052→0.671 | 同上 |
| pi05-1 / ddp2 | 1 / 2 | 154 / 175 s | 0.128→0.275 / 0.155→0.079 | 本地 q01/q99 |
| eo1-1 / ddp2 | 1 / 2 | 124 / 139 s | 0.093→23.963 / 0.087→41.871 | 本地去掉未知字段 |
| g05-1 / ddp2 | 1 / 2 | 82 / 93 s | 约 10.5 | `g05_so101`，关闭 CoT |
| wall-1 / ddp2 / fsdp2 | 1 / 2 / 2 | 125 / 111 / 129 s | 0.752→0.117 / 0.482→0.229 / 0.483→0.229 | FSDP 用 fp32 + bf16 |
| fastwam-1 / ddp2 / fsdp2 | 1 / 2 / 2 | 264 / 249 / 268 s | 约 4.4 → 13–15 | `lerobot/libero` |
| cosmos3edge-1 / ddp2 / fsdp2 | 1 / 2 / 2 | 193 / 183 / 181 s | 约 11–15 | 第一轮即通过 |
| cosmos3nano-1 / ddp2 / fsdp2 | 1 / 2 / 2 | 399 / 305 / 341 s | 13.572→14.176 / 11.988→16.221 / 11.979→16.224 | DDP 改 bf16 后通过 |
| lawam-1 / ddp2 / fsdp2 | 1 / 2 / 2 | 111 / 80 / 98 s | 0.595→0.502 / 0.416→0.814 / 0.416→0.813 | FSDP 加载后转 fp32 |
| molmo-1 / ddp2 | 1 / 2 | 82 / 105 s | 1.137→0.584 / 0.830→1.186 | `train_mode_vlm=fft`。没有做 FSDP |
| lingbot-1 / ddp2 / fsdp2 | 1 / 2 / 2 | 177 / 183 / 196 s | 0.525→0.405 / 0.494→0.267 / 0.467→0.265 | 空 `align_params`，权重用符号链接 |

多卡通过时，`dp_world_size=2`。DDP 为 `dp_replicate=2`，FSDP 为 `dp_shard=2`。

## 6. 第 5 节的问题与处理方式

下面只记挡住第 5 节某次 smoke 的问题。处理都在容器外的脚本、挂载目录或一次性 Python 补丁里。镜像 `88a8506e6197` 没有重建。后面各节的问题写在各节里。

### A. 容器和 ROCm 运行环境

**`--group-add render` 失败。** 这台主机上该组不能按名字加入。改用数字 gid：`video=44`，`render=992`。

**`MIOPEN_FIND_MODE=FAST` 没有停掉卷积查找。** 环境变量改成数字 `2`。新架构的第一步仍可能要几十秒，这是查找和 JIT，不是训练挂死。

**aiter 在没有对应 GPU 的构建阶段会报警，并回退到原生 attention。** `flash_attn_func` 不在这份 aiter 里。使用 SDPA 的策略可以继续。FLUX3 默认要 `flash_attn_interface`，改用 `--policy.attn_mode=torch` 后不再依赖它，见第 10 节。

**磁盘被 smoke 权重写满。** 镜像本身 81.3 GB。lingbot、MolmoAct2、lawam 的本地副本再加上 optimizer，曾把空闲空间打到几十 GB。已完成任务的 `*.safetensors`、FSDP `*.distcp` 和 `training_state` 按授权删除。没有删除 DeepSeek 权重、`/opt/rocm-7.0.2`、`lerobot-rocm` 镜像和 Hugging Face 原始 blob。收尾时空闲约 107 GB。

### B. 登录与数据集

**π0 / π0-fast / π0.5 拉取 `google/paligemma-3b-pt-224` 返回 401。** DORobot 上那份 825 字节 token 签名无效。改用跳板机上可用的 Hugging Face token，放到 DOUni `/root/.cache/huggingface/token`（权限 600），由缓存挂载进容器。token 没有写进脚本。之后 π 系列单卡和 DDP 通过。

**π0.5、g05、wall-oss 的 QUANTILES 需要 q01/q99。** 公式是 `x' = 2 * (x - q01) / (q99 - q01) - 1`。`svla_so100_pickplace` 原来没有这两项。仓库脚本 `augment_dataset_quantile_stats.py` 算完会 `push_to_hub()`。本次只在容器里调用计算和本地 `write_stats`，把结果写入该数据集的 `meta/stats.json`，没有推 Hub。统计的宽度是机器人自由度 6，不是策略内部 padding 后的 20 或 32。

### C. 单个策略

**g05 维度和相机。** `g05_base` 是 galaxea，原始 14 维；`embodiment=so100` 的槽位是 6 维，两者不能混用。成功路径是 `g05_so101`，embodiment so100，原始 6 维，策略维 20，槽位 10–15。可选相机 `wrist_left` 必须留在 `camera_order` 里，不能改名，否则子集检查失败。最终相机顺序是 top、wrist_left、wrist。

**g05 语言字段。** 分位数补上之后，默认 recipe 仍要样本字段 `atomic_task`，tokenizer 会 `KeyError`。smoke 使用 `--policy.use_language_recipe=false --policy.predict_cot=false --policy.runtime_system=system1`。这只说明「关掉 CoT 后训练能跑」，没有验证 CoT 数据路径。

**eo1 配置比镜像新。** Hub `lerobot/eo1-base` 的 `config.json` 含 `use_language_recipe`，镜像里的 `EO1Config` 拒绝未知字段。把 snapshot 拷到本地，删掉该字段，`--policy.path` 指向本地目录，dtype bfloat16，attention 用 sdpa。`type` 字段保留。

**fastwam 动作维。** `fastwam_base` 的 action 是 7、本体感受是 8。SO100 是 6/6，会报 action shape 必须是 7。改用 `lerobot/libero`，dtype float32，attention 用 sdpa。单卡、DDP、FSDP 随后通过。

**wall-oss FSDP 混 dtype。** 权重里同时有 bf16 和 fp32 时，FSDP 在 `_init_mp_dtypes` 断言失败。通过的组合是 `--policy.dtype=float32` 加 `--accelerator.mixed_precision=bf16`，再加 `--parallelism.dp_shard=2`。一旦单独设置 `accelerator.fsdp.mixed_precision_policy.*`，它会替换由 `accelerator.mixed_precision` 推导出的策略，所以这次没有另设那组字段。

**cosmos3 nano DDP 显存。** 单卡和 FSDP 在 fp32 下通过。2 卡 DDP 在 fp32 下大约用到 184/192 GiB 后 OOM。同一任务改为 `--policy.dtype=bfloat16 --accelerator.mixed_precision=bf16` 后通过。edge 的单卡、DDP、FSDP 没有改 dtype 也通过。

**lawam FSDP 混 dtype。** 单卡和 DDP 原本就通过。FSDP 有两层失败：

1. 直接 FSDP：断言参数 dtype 不是单一的。
2. `--policy.dtype=float32` 加 bf16 autocast：过了初始化，LAM 的 VQ `LayerNorm`（`lam_core/core/vq.py`）报 `expected scalar type Float but found BFloat16`。
3. 把 checkpoint 文件里 493 个 bf16 张量改成 fp32（约 10.2 GB）仍不够。Qwen backbone 在 `from_pretrained` 时又按 bf16 建出来。

最终做法是不改权重文件，在进程里包一层 `LaWAMPolicy._load_as_safetensor`：原加载返回后，把所有浮点参数 `.float()`，打印 `cast_params=493` 且 dtype 只剩 float32，然后照常进入 `lerobot-train`。FSDP 不再开 bf16 autocast。该脚本是挂载进容器的一次性文件，没有写回仓库。通过用时 98 秒，`dp_world_size=2`。

**MolmoAct2。** Hub 配置多了 `enable_lora_vlm`、`enable_lora_action_expert`、`train_action_expert_only`、`model_dtype`，镜像里的 `MolmoAct2Config` 不接受。剥掉这些字段后，默认 `train_mode_vlm=lora` 仍会调用 peft，而 peft 要求 `torchao > 0.16`，镜像是 0.15。本地配置和命令行都改成 `train_mode_vlm=fft`。相机按该 checkpoint 的命名，把 top 映到 cam0、wrist 映到 cam1。没有升级 torchao，也没有做 FSDP。单卡 82 秒、DDP 105 秒通过。

**lingbot 教师和权重。**

1. checkpoint 里的 `align_params` 指向 MoGe。文档约定空字典表示不做蒸馏。命令行 `--policy.align_params='{}'` 不会清空已有字典，draccus 是把 CLI 字典并进 checkpoint 字典。必须改本地 `config.json`。
2. 教师文件 `Ruicheng/moge-2-vitb-normal` 的 `model.pt` 缺少端口期望的键。空 `align_params` 后不再下载教师。`from_pretrained` 默认 `strict=False`，checkpoint 里多出来的 align 张量可以忽略。
3. 第一次本地目录没有 `model.safetensors`（约 24 GB，缓存里是指向 blob 的符号链接）。补上真实文件后，两步 loss 为 0.525 → 0.405，模型权重写出，optimizer 因磁盘满失败。
4. 成功的一次不再复制这 24 GB：小文件用 `cp -L` 变成普通文件，权重只做符号链接。`cp -a` 会把 `config.json` 的相对符号链接原样拷走，换目录后指向不存在的 `../../blobs/...`，所以必须解引用。单卡、DDP、FSDP 随后在 177 / 183 / 196 秒通过，optimizer 也写完了。FSDP 只用 `dp_shard=2`，没有再开 bf16 autocast。

### D. 没有当成缺陷修进仓库的事项

- 上述兼容问题是配置、数据或依赖版本的不匹配，没有形成 LeRobot 补丁。后来提交上游的两个 bug 见第 12 节。
- `results.txt` 保留了每一次 FAIL。看单个任务是否通过时，以同名最后一条为准。

## 7. 恢复、8 卡、评估、DataLoader、编译

2026-11-18 08:39–09:24（UTC+8）。日志：`run-next.log`。镜像未重建。大权重在每项结束后删除。

| 任务 | GPU | 结果 | 用时 | 说明 |
|---|---:|---|---:|---|
| dit-resume-seed | 1 | PASS | 50 s | loss 1.204 → 0.986，写出 step 2 |
| dit-resume | 1 | PASS | 46 s | `--resume=true`，`--steps=4`。日志从 step 3 开始：1.255 → 1.144，写出 `checkpoints/000004` |
| diffusion-ddp8 | 8 | PASS | 1539 s | loss 1.242 → 1.225。`dp_replicate=8`，`dp_world_size=8` |
| act-fsdp8 | 8 | PASS | 491 s | loss 70.290 → 71.516。`dp_shard=8`，`dp_world_size=8` |
| dit-eval | 1 | PASS | 53 s | 前两次渲染失败，见下方。OSMesa 这次退出码 0。训练 loss 1.204 → 0.986。评估 1 局 2 步，`pc_success=0`，`eval_s=0.52`。成功率不作为通过条件 |
| dit-workers | 1 | PASS | 60 s | `num_workers=2`，20 步。step 5/10/15/20 的 loss 为 1.165 / 1.055 / 0.987 / 0.863，写出 `checkpoints/000020` |
| pi0-compile-1 | 1 | PASS | 410 s | `--policy.compile_model=true`。loss 0.011 → 0.515 |

**dit-eval 的渲染。** 训练本身三次都完成了。第一次评估走 GLFW/X11，容器没有 `DISPLAY`，MuJoCo 报 `gladLoadGL error`。第二次设 `MUJOCO_GL=egl`，驱动报 `can't create a graphics context on a compute chip`（MI300X VF 没有图形上下文），同时缺 `libGLU.so.0`。第三次在当次容器里安装 `libosmesa6` 和 `libglu1-mesa`（没有重建镜像），`MUJOCO_GL=osmesa`，评估循环跑完。

队列已结束，没有残留容器。`results.txt` 里 `dit-eval` 和 `dit-eval-egl` 的 FAIL 仍保留；有效结果是 `dit-eval-osmesa PASS`。

## 8. 后续计划

2026-11-18 19:44 起。日志：`run-followup.log`。规则与 `followup-plan.md` 相同。

| 任务 | 状态 | 说明 |
|---|---|---|
| molmo-lora-1 | FAIL→PASS | 这一轮失败：`torchao 0.18.0` 装上后 HIP 仍在。61 秒时严格加载失败，checkpoint 没有 LoRA 键。`strict=False` 之后又缺 `policy_postprocessor.json`。补了一份生成的后处理器，数据批次里出现 `None`，53 秒失败。第 9 节修好后单卡和 DDP 都通过 |
| flux3-1 | BLOCKED 7 s | `flash_attn_interface` 不存在。安装 `flash-attn` 时隔离构建找不到 torch |
| flux3-torch | 当前 main BLOCKED；PR #5213 PASS | 当前 `main` 上：LeRobot 的后端参数没有传到 NATTEN，NATTEN 在 ROCm 上禁用编译版 flex，强制打开后 3D block mask 内核又启动失败。见 10.1。用上游 PR #5213（不装 NATTEN，纯 PyTorch 邻域注意力）单卡 2 步 PASS，见 10.2 |
| g05-cot-1 | PASS 81 s | 本地副本加了常量 `atomic_task`。模板含 `<atomic_task_text>`。loss 10.536 → 11.212，`ce_loss` 与 `fm_loss` 都有 |
| hub-push | 不做 | smoke checkpoint 不值得上传 Hub，不再等私有仓库名 |
| multi-node | 待后续 | DOUni `<DOUni-ip>` 与 DORobot `<DORobot-ip>` 都没有 InfiniBand。DORobot 上没有 `lerobot-rocm`。不跨公网训练。等有多机互联条件再做 |
| webui-dit | PASS | `POST /api/start` 在 DOUni 上跑固定的 DiT 单卡 2 步。`/api/log` 里有 loss 1.204 → 0.986 和 `EXIT:0`。checkpoint 写出后已删大权重。停止接口返回成功。页面本身没有在浏览器里点到：本机浏览器打不开 `127.0.0.1:8765` |

## 9. MolmoAct2 LoRA

2026-11-19。

| 任务 | 状态 | 说明 |
|---|---|---|
| molmo-lora-defer | PASS 79 s | 单卡。先按 fft 加载基座，再挂 LoRA。`trainable params 737356320`。loss 1.127 → 0.664 |
| molmo-lora-defer-ddp2 | PASS 100 s | `dp_replicate=2`，`dp_world_size=2`。loss 0.838 → 1.352 |

空批次的字段是 `atomic_task`。g05 CoT 那次给本地副本加字段时，`info.json` 是指向缓存 blob 的符号链接，写入穿透了原始数据集；parquet 里没有这一列，读出来就是 `None`。已经从 `info.json` 里去掉这个声明。

LoRA 本身要等基座 `model.safetensors` 严格加载完成后再调用 `_apply_lora_adapters`。发布包没有 `policy_postprocessor.json`，这次在当次目录里按配置生成了一份。`torchao 0.18.0` 装在当次容器里，HIP 仍在。大权重已删。

## 10. flux3 复查

2026-11-19 17:25（UTC+8）：三个门控仓库对 `alexhegit` 曾是 403。授权后可下载。

2026-11-19 18:19（UTC+8）单卡 2 步重跑结论：**BLOCKED**。根因见 10.1，用上游 PR 跑通见 10.2。

| 尝试 | 结果 | 说明 |
|---|---|---|
| 门控访问 | OK | `so101` / `base` / `droid` 的 `config.json` 均可下载 |
| `natten==0.21.7` + `torchao>0.16` | OK | 当次容器安装，HIP 与 `torchvision+rocm` 仍在 |
| 默认 `canvas_hw=[256,512]` | FAIL 111 s | PEFT 已挂上；Video VAE 的 `na3d` 走 `flex-fna`，创建_block_mask 试图分配 **882 GiB** → OOM。CUTLASS/Hopper/Blackwell 内核在 ROCm 上不可用 |
| 缩小 `canvas_hw=[64,128]` 并改 `n_obs_steps`/`history_snapshots` | FAIL 95 s | `History processor settings disagree with the policy`（与预训练 processor 不一致） |

`whl.natten.org` 只有 CUDA 匹配轮子，没有 ROCm/HIP。flash-attn 仍装不上，但 `attn_mode=torch` 已不依赖它；真正阻塞是 NATTEN Video VAE。

### 10.1 根因复查（2026-11-20 00:22–00:44，UTC+8）

脚本 `flux3-flexcompile.sh`：在运行时给 `video_vae` 打补丁，不改仓库源码。GPU 1，单卡 2 步。

| 尝试 | 结果 | 说明 |
|---|---|---|
| `attention_kwargs` 里加 `torch_compile=True` | FAIL 139 s | 仍是 882 GiB OOM，日志里没有编译版 `create_block_mask` 的告警，说明参数没生效 |
| 把 `backend` / `torch_compile` 提到 `na2d`/`na3d` 顶层 | FAIL 103 s | `Can't run NATTEN with Flex Attention (compiled)` |
| 再把 `natten_checks._FLEX_COMPILE_SUPPORTED` 设为 True | FAIL 285 s | 2D 路径编译版 flex 通过（自动调优约 3 分钟）。3D 路径编译版 `create_block_mask` 自动调优时 `CUDA driver error: 1` |
| 再加 `TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0` | FAIL 282 s | Triton 自己的 launcher 报 `Triton Error [HIP]: Code: 1, invalid argument` |

三层原因：

1. **LeRobot**：`video_vae.py` 把 `backend` 放进 `attention_kwargs`。NATTEN 0.21 的 `na2d`/`na3d` 只在顶层参数里读 `backend` 和 `torch_compile`，`attention_kwargs` 只用于自注意力捷径和额外 KV 的 FMHA。所以 `F3_NATTEN_BACKEND` 和代码里的后端探测对主路径都不生效，由 NATTEN 自己选。CUDA 上 NATTEN 会选中 Hopper/Blackwell 内核，问题被掩盖；没有 CUTLASS 内核的设备落到未编译的 flex，物化 dense mask。与 ROCm 无关，是平台无关的 bug。
2. **NATTEN**：`get_device_cc()` 只认 `torch.version.cuda`，ROCm 上返回 0，所以 `_FLEX_COMPILE_SUPPORTED` 为 False，编译版 flex 被整体禁用。
3. **NATTEN 3D mask + inductor on ROCm**：实际输入是 `(1, 42, 64, 128, 4, 64)`，窗口 `[5,5,5]`，约 34 万 token。只用 NATTEN 的复现（每项一个新进程）：T=8（6.5 万 token）编译版通过，42.9 s；T=42 非因果在 `create_block_mask` 自动调优时 `CUDA driver error: 1`；T=42 因果通过，但含编译用了 861.4 s。同一形状的因果调用放在 FLUX3 VAE 的训练步里两次都报错，所以失败不只取决于形状。纯 PyTorch 的 `torch.compile(create_block_mask)`（1-D 局部窗口）在 1,048,576 token 时也通过，所以问题集中在 NATTEN 的 3D mask，不是 PyTorch 的通用限制。脚本 `repro-natten-flex3d.py`、`repro-flex-blockmask.py`。

结论：当前 `main` 加 NATTEN 的路径仍为 BLOCKED。前两层可以在补丁里绕过，第三层要等 NATTEN 修。

### 10.2 上游 PR #5213（2026-11-20 03:16，UTC+8）

[huggingface/lerobot#5213](https://github.com/huggingface/lerobot/pull/5213)（作者 shoumikhin，未合并，head `141d22b8`）的第 1 个 commit 修了 10.1 里 LeRobot 那一层的 bug，第 2 个 commit 在没装 NATTEN 时用纯 PyTorch 计算邻域注意力。脚本 `flux3-pr5213.sh`，把 PR 源码挂到 `/pr` 并放在 `PYTHONPATH` 最前面，不装 NATTEN，只装 `torchao>0.16`。GPU 1。

| 项目 | 结果 | 说明 |
|---|---|---|
| flux3 单卡 2 步（LoRA r=8，bf16） | PASS 87 s | VAE 提示一次走纯 PyTorch 注意力。可训练参数 63M。loss 0.137 → 0.083（`video_mse` 0.119 → 0.072，`action_mse` 0.050 → 0.031）。峰值显存 27.6 GB。step 2 写出 `adapter_model.safetensors`，之后已删除 checkpoint |
| PR 的 `tests/policies/flux3`（GPU） | 155 passed，5 skipped | 3 个需要 Apple MPS，2 个需要 NATTEN 加 CUDA |

已在 #5213 下留言附上 ROCm 结果。我们本地的同类修复（`fix/flux3-natten-backend`）与它重复，不再提交。

## 11. 补充覆盖

2026-11-19 19:38–20:12（UTC+8）。单卡、`steps=2`、`push_to_hub=false`。日志：`run-coverage.log`。Long-WAM 的 Hub 缓存约 24 GB，训练步已经跑完，写 checkpoint 时磁盘满；为了继续后面的任务删了这份基座缓存。

| 模型 | 结果 | 说明 |
|---|---|---|
| VQ-BeT | PASS 33 s | aloha episode 0，从头训练 |
| TD-MPC | PASS 83 s | aloha 图像 480×640 被拒（只要方形）。改 `lerobot/pusht` 后通过 |
| DM05 | PASS 183 s | `lerobot/dm05_base`，LIBERO episode 0 |
| EVO1 | PASS 48 s | stage1，`InternVL3-1B`，SO100 |
| WALL-OSS (`wall_x`) | PASS 191 s | `x-square-robot/wall-oss-flow`，`recipe=null`，eager attention |
| VLA-JEPA | PASS 51 s | 冻结 Qwen，重初始化动作头。SO100 相机改名到 `exterior_1_left` / `exterior_2_left` |
| Long-WAM | PASS 277 s | 首次写 safetensors 时 `No space left on device`。2026-11-20 00:25（UTC+8）清理 GR00T checkpoint 后重跑 `cover-longwam2`：loss 5.662 → 3.986，峰值显存 68.2 GB，step 2 写出 `model.safetensors`（约 24.8 GB）和 `optimizer_state`（约 24.1 GB） |
| LingBot-VA | PASS 182 s | `attn_mode=flex`，LoRA。当次容器装 `torchao>0.16`，HIP 仍在。LIBERO episode 0 |
| SARM | PASS 44 s | `annotation_mode=single_stage`，SO100 的 `observation.images.top` |
| FineART-VLA | PASS 165 s | 本地 aloha 副本把同一路视频挂成三路相机，并写入 action/state 的 q01/q99。loss 12.050 → 13.802 |
| GR00T N1.7 | PASS 66 s | `nvidia/GR00T-N1.7-3B`，`new_embodiment`，bf16，SO100 episode 0。`nvidia/Cosmos-Reason2-2B` 授权后可下载。loss 1.015 → 0.999，step 2 存盘 `pretrained_model/model.safetensors`（约 12.6 GB），`optimizer_state` 约 13 GB。峰值显存约 36 GB |
| HIL-SERL | PASS 37 s | 不走 `lerobot-train`。GPU 上两步 SAC：critic loss 20.656 → 31.831，actor loss -3.731 → -4.334 |
| Robometer | 推理 PASS 100 s；RA-BC PASS 58 s | 零样本，`is_trainable=False`。SO100 一帧 `compute_reward` = 0.124。RA-BC 见 11.1 |
| TOPReward | 推理 PASS 22 s；RA-BC PASS 105 s | 零样本。4 帧视频 `compute_reward` = -19.875。RA-BC 见 11.1 |
| reward_classifier | PASS 26 s | 不走 `lerobot-train`：没有公开的 128×128 带奖励数据集，CNN 头写死 4×4 特征图。pusht episode 0 取 8 帧缩放到 128×128，`next.reward` 高于中位数记为 1。`lerobot/resnet10`，可训练参数 1,181,441。GPU 上两步 AdamW：loss 0.6633 → 0.6481。`compute_reward`、`save_pretrained`、`from_pretrained` 重载后结果一致。脚本 `cover-classifier.sh` |

2026-11-19 20:58（UTC+8）为这几项清了已经通过模型的 Hub 缓存，约 235 GB。没有动数据集缓存、`/opt/rocm-7.0.2` 和 Docker 镜像。GR00T 的门控授权已于 2026-11-19 23:19（UTC+8）前完成，`alexhegit` 可以访问 `nvidia/Cosmos-Reason2-2B` 和 `nvidia/GR00T-N1.7-3B`。GR00T 烟测已通过。

### 11.1 RA-BC（Robometer / TOPReward 打分 + 加权训练）

2026-11-20 01:45–01:49（UTC+8）。脚本 `cover-rabc.sh`，GPU 3。数据 `lerobot/svla_so100_pickplace` episode 0（454 帧）。进度文件只写本地 `/outputs/rocm-train/rabc/*.parquet`，没有 `--push-to-hub`。训练用 SmolVLA（`lerobot/smolvla_base`），`--sample_weighting.type=rabc`，`head_mode=sparse`，batch 4，2 步。

| 模型 | 打分 | 加权训练 | 说明 |
|---|---|---|---|
| Robometer | 454 帧逐帧，`progress_sparse` 0.2147 → 0.7966（max 0.8805），无 NaN | PASS 58 s。loss 0.893 → 2.697，`sample_weight_mean_weight` 0.25 | `--batch-size 16` |
| TOPReward | 15 个锚点，插值到 454 帧，归一化 0 → 0.9715，无 NaN | PASS 105 s。loss 0.406 → 2.229，`sample_weight_mean_weight` 0.50 → 0.75 | 需要运行时补丁，见下 |

- 第一次用 ACT 训练失败：`ACTPolicy.forward() got an unexpected keyword argument 'reduction'`。RA-BC 要求策略实现 `forward(batch, reduction="none")`，ACT 没有，改用 SmolVLA。
- TOPReward 的 `compute_rabc_weights` 前缀扫描从 1 帧开始，Qwen3-VL 报 `t:1 must be larger than temporal_factor:2`。镜像里 transformers 是 5.22.2，正在 LeRobot 锁定的 `>=5.22.0,<5.23.0` 内，所以这是平台无关的 LeRobot bug。脚本里在运行时把单帧前缀复制成 2 帧后通过。两处都已提交上游，见第 12 节。

### 11.2 补充 2 卡 DDP（2026-11-22 10:03–10:35，UTC+8）

脚本 `run-ddp2-coverage.sh`，日志 DOUni `run-ddp2.log`。每项由该模型最后一次通过的单卡脚本生成：`lerobot-train` 换成 `accelerate launch --num_processes=2 "$(which lerobot-train)"`，输出目录加 `-ddp2`，其他参数不变。GPU 0、1，串行。PASS 后删除 checkpoint。

| 任务 | 结果 | 用时 | step1 → step2 loss | 峰值显存 |
|---|---|---:|---|---:|
| VQ-BeT | PASS | 45 s | 0.425 → 0.437 | 0.5 GB |
| TD-MPC（pusht） | PASS | 98 s | 3.343 → 10.695 | 0.2 GB |
| DM05 | PASS | 211 s | 0.753 → 0.262 | 51.0 GB |
| EVO1 | PASS | 73 s | 0.724 → 0.885 | 3.5 GB |
| VLA-JEPA | PASS | 81 s | 1.358 → 1.486 | 7.6 GB |
| WALL-OSS flow（`wall_x`） | PASS | 243 s | 0.084 → 0.187 | 48.0 GB |
| GR00T N1.7 | PASS | 80 s | 0.942 → 1.049 | 42.1 GB |
| FineART-VLA | PASS | 184 s | 12.942 → 14.545 | 42.5 GB |
| LingBot-VA（LoRA，flex） | PASS | 516 s | 6.629 → 1.297 | 14.3 GB |
| Long-WAM | PASS | 250 s | 5.648 → 4.742 | 79.4 GB |
| FLUX3（PR #5213，LoRA） | PASS | 101 s | 0.135 → 0.181 | 27.8 GB |

每项日志里配置为 `dp_replicate: 2`，`effective batch size: 2`，并有 `Checkpoint policy after step 2`。两个 LoRA 任务由 PEFT 保存 adapter，日志里没有 `saved to` 这一行。LingBot-VA 第 1 步用了约 250 秒，是 flex attention 的编译。FLUX3 的日志确认 lerobot 从 `/pr/src` 加载、没有 NATTEN。

## 12. 上游跟进

PR 描述、留言和 issue 草稿在 `tmp/rocm-train/upstream/`。2026-11-20 以 GitHub 账号 `alexhegit` 提交。PR 描述都注明只在 AMD MI300X 上测试、没有 NVIDIA GPU。

| 对象 | 状态 | 内容 |
|---|---|---|
| flux3 NATTEN 后端 | 不提交，与 [#5213](https://github.com/huggingface/lerobot/pull/5213) 重复；已在 #5213 留言 ROCm 结果 | 本地分支 `fix/flux3-natten-backend`（commit `4df276b8`）保留作参考。见 10.2 |
| TOPReward 单帧视频 | 已提交 [#5432](https://github.com/huggingface/lerobot/pull/5432)（alexhegit）。2026-11-22：等待评审，主 CI 未运行 | `_prepare_video_batch` 加 `min_frames`，编码器按 `temporal_patch_size` 补帧。3 个新测试修复前失败、修复后通过；TOPReward 测试 34 passed / 1 skipped。修复后源码不打补丁跑 `compute_rabc_weights`：454 行，末帧 0.9715，与 11.1 一致 |
| RA-BC 策略检查 | 已提交 [#5433](https://github.com/huggingface/lerobot/pull/5433)（alexhegit）。2026-11-22：社区成员 AungKaung1928 在 CPU 上复现 24 passed / 1 skipped，并核对 `main` 上哪些策略的 `forward` 有 `reduction`，结论一致。等待维护者评审，主 CI 未运行 | 创建策略后检查 `forward` 是否有 `reduction`，没有就报清楚的 `ValueError`。`test_sample_weighting.py` 24 passed / 1 skipped。真实策略：`act`、`diffusion` 拒绝；`pi0`、`pi05`、`smolvla`、`fastwam` 通过 |
| NATTEN issue | 已提交 [SHI-Labs/NATTEN#354](https://github.com/SHI-Labs/NATTEN/issues/354)（alexhegit，2026-11-22）。提交约一分钟后更正了正文：T=42 因果的独立复现是通过的 | ROCm 上 `get_device_cc()` 返回 0，编译版 flex 被禁用；强制打开后 3D 大尺寸 mask 有时启动失败，成功时编译也很慢。纯 NATTEN 复现：T=8 OK 42.9 s；T=42 非因果 `CUDA driver error: 1`；T=42 因果 OK 861.4 s。FLUX3 VAE 内同形状因果两次失败。最新版 0.21.7 和 master 仍是同样的 `get_device_cc()` |
| LeRobot ROCm 文档 | 本地分支 `docs/rocm-known-issues`（worktree `~/Repo/lerobot-rocm-docs`），commit `77beb7e2`，基于 `origin/main`，未推送 | 在 `docs/source/torch_accelerators.mdx` 的 ROCm 小节下加「Known issues on ROCm」（render 组 gid、MIOpen 首步慢、MuJoCo 用 OSMesa、LoRA 要 `torchao>0.16`、没有 FlashAttention、FLUX3 需要 NATTEN）和「Policies trained on ROCm」。prettier 和 typos 通过 |
| PyTorch issue | 不提 | 纯 PyTorch 的 `torch.compile(create_block_mask)`，1-D 局部窗口 mask，65,536 / 262,144 / 1,048,576 token 都通过（13.4 / 4.0 / 7.1 s）。问题集中在 NATTEN 的 3D mask，先在 NATTEN issue 里说明 |
## 13. 分布式训练特性（2026-11-22 14:00–20:15，UTC+8）

脚本在 `tmp/rocm-train/dist/`：通用队列 `run-dist-queue.sh`（由单卡脚本生成多卡任务，额外参数接在 `--output_dir` 之后）、各队列文件 `queue-*.txt`、resume 与拓扑任务生成器、FSDP 包装 `fsdp_autowrap_train.py`、缓存清理 `prune-hf-models.py`。日志 DOUni `run-dist-{a,b,a2,c,d,e,f,g1,g2,h1,h2,i,j,k}.log`。每项 PASS 后删除 checkpoint。69 项（另有 3 项诊断），最终 59 PASS、10 FAIL。`results.txt` 保留每次尝试，以同名最后一条为准。

### 13.1 FSDP 切分单元

- 只有 ACT、WALL-OSS 0.5、FastWAM、LingBot-VLA v2、LaWAM、Cosmos3 声明了 `_fsdp_wrap_modules`。其他策略必须在命令行给切分规则。
- **`--accelerator.fsdp.min_num_params`（按参数量切分）不可用**：accelerate 的 FSDP2 size-based 策略会把整个 `nn.ModuleList` 当成单元，`fully_shard does not support containers that do not implement forward`。DiT、VQ-BeT、DiT 的梯度累积 / HSDP / fp16 / cpu_offload 等首轮都因此失败。与 ROCm 无关。
- 改用 `--accelerator.fsdp.wrap_modules`（按类名）。为了不逐个手查，`fsdp_autowrap_train.py` 在 `accelerator.prepare` 前把 `ModuleList` 里重复出现、带子模块的层类作为单元，并打印 `AUTO_WRAP [...]`，与手写 `wrap_modules` 等价。跳过叶子层（`Linear` 等），否则会拆开 tied 的 `lm_head`。
- 可选开关：`FSDP_CAST_DTYPE=float32|bfloat16` 在切分前把参数统一成一种 dtype；`FSDP_DTYPE_REPORT=1` 打印每个单元和根单元的 dtype 分布。

各模型最终使用的单元和精度：

| 模型 | FSDP 单元 | 精度处理 |
|---|---|---|
| DiT | `CLIPEncoderLayer`、`TransformerBlock` | — |
| VQ-BeT | `Block` | — |
| TD-MPC | `Sequential` | — |
| X-VLA | `BartEncoderLayer`、`Florence2VisionBlock`、`Florence2VisionConvEmbed`、`TransformerBlock` | — |
| EVO1 | `BasicTransformerBlock`、`InternVLVisionLayer`、`Qwen2DecoderLayer` | — |
| VLA-JEPA | `BasicTransformerBlock`、`Qwen3VLTextDecoderLayer`、`Qwen3VLVisionBlock`、`Qwen3VLVisionPatchMerger` | — |
| GR00T N1.7 | 自动 | — |
| LingBot-VA（LoRA） | 自动 | — |
| π0-FAST | `SiglipEncoderLayer`、`_PiGemmaDecoderLayerBase` | `--policy.dtype=float32 --accelerator.mixed_precision=bf16` |
| π0、π0.5 | 只 `SiglipEncoderLayer` | 同上 |
| SmolVLA | 只 `SmolVLMEncoderLayer` | `FSDP_CAST_DTYPE=float32` + bf16 |
| DM05 | 只 `SiglipEncoderLayer` | fp32 + bf16 |
| EO1 | `Qwen2_5_VLDecoderLayer`、`Qwen2_5_VLVisionBlock` | fp32 + bf16 |
| MolmoAct2 | 自动 | `FSDP_CAST_DTYPE=bfloat16` |
| FLUX3（PR #5213，LoRA） | 自动 | `--policy.gradient_checkpointing=false` |
| Diffusion | 自动 | — |

规律：

1. **混合 dtype**：多数 VLA 以 bf16 加载主干、fp32 保留动作头或 norm，FSDP2 报 `FSDP expects uniform original parameter dtype`。有 `dtype` 选项的用 fp32 权重加 bf16 autocast；没有的用 `FSDP_CAST_DTYPE`。
2. **绕过 `forward`**：π0、π0.5、SmolVLA、DM05、FineART 的联合注意力直接调用语言层的子模块（`q_proj` 等），FSDP 的 all-gather 钩子不触发，报 `got mixed torch.Tensor and DTensor`。π0-FAST 不走这条路径所以整层可切。解法是只把视觉编码层作为单元，语言层留在根单元。
3. **梯度检查点**：FLUX3 打开时，重算的张量 metadata 与前向不一致（`CheckpointError`）。关掉后通过，单卡也只用 27 GB。

### 13.2 特性测试（DiT / SmolVLA / ACT）

| 特性 | 任务 | 结果 |
|---|---|---|
| DDP 4 卡 | `dit-ddp4`、`smolvla-ddp4` | PASS |
| DDP 8 卡 | `smolvla-ddp8`、`cover-groot-ddp8`、`molmo-ddp8` | PASS |
| FSDP 自动推断分片数 `dp_shard=-1`（4 卡） | `dit-fsdp4-infer` | PASS |
| HSDP | `dit-hsdp4`（2×2）、`smolvla-hsdp4`（2×2）、`dit-hsdp8`（4×2）、`pi0-hsdp8`（2×4） | PASS |
| FSDP 8 卡 | `pi0-fsdp8`、`fastwam-fsdp8`、`wall-fsdp8`、`cosmos3edge-fsdp8` | PASS |
| 梯度累积 2 | `dit-gradacc-ddp2`、`dit-gradacc-fsdp2`、`smolvla-gradacc-ddp2`；`sync_each_batch=true`：`dit-gradacc-sync-ddp2` | PASS |
| fp16 + GradScaler | `dit-fp16-ddp2`、`dit-fp16-fsdp2` | PASS |
| bf16 | `dit-bf16-fsdp2`、`smolvla-bf16-ddp2` | PASS |
| FSDP 混合精度策略（param bf16、reduce fp32） | `dit-mpp-fsdp2` | PASS |
| FSDP `reshard_after_forward=false` | `dit-noreshard-fsdp2` | PASS |
| FSDP `cpu_offload=true` | `dit-cpuoffload-fsdp2` | PASS，需要 `--accelerator.init_process_group.backend=cpu:gloo,cuda:nccl`。默认只有 nccl 时报 `No backend type associated with device type cpu` |
| DDP 选项（`find_unused_parameters=false`、`gradient_as_bucket_view`、`static_graph`） | `dit-ddpopts-ddp2` | PASS |
| 用户指定 `wrap_modules` | `act-wrapoverride-fsdp2`（`["ACTEncoderLayer"]`） | PASS |
| `num_workers=2` | `dit-workers-ddp2` | PASS |
| 训练中仿真评估（OSMesa） | `dit-eval-ddp2` | PASS |
| RA-BC | `smolvla-rabc-fsdp2` PASS；`smolvla-rabc-ddp2` FAIL，见 13.3 | — |
| resume，2 → 4 步 | `dit-resume-ddp2`、`act-resume-ddp2`、`act-resume-fsdp2` PASS；`dit-resume-fsdp2` FAIL，见 13.3 | — |
| 换分片 resume：FSDP 2 → FSDP 4 | `act-reshard-resume` | PASS |
| 分片 → 非分片 resume 被拒绝 | `act-fsdp-to-ddp-reject` | PASS，报 `Cannot resume: the checkpoint was written with a sharded topology` |
| `--checkpoint_format=dcp` / `safetensors_dcp` + resume | `act-fsdp2-dcp`、`act-fsdp2-stdcp` | PASS |

### 13.3 失败与发现

| 任务 | 原因 | 性质 |
|---|---|---|
| `smolvla-rabc-ddp2` | `'DistributedDataParallel' object has no attribute 'config'`。`lerobot_train.py` 在 `accelerator.prepare` 之后用包装过的 `policy` 创建 sample weighter，`_make_rabc_weighter` 读 `policy.config`。FSDP2 不改模块类所以不受影响 | **LeRobot bug**，最新 `main`（`e5be0c16`）仍在。本地修复分支 `fix/rabc-ddp-unwrap`（worktree `~/Repo/lerobot-rabc-ddp`）：传 `accelerator.unwrap_model(policy)`。修复后 `smolvla-rabc-ddp2-fix` PASS。未提交 |
| `dit-resume-fsdp2` | 恢复时 `Missing key in checkpoint state_dict: optimizer.state...post_layernorm.weight.step`。DiT 的 CLIP 视觉塔 `post_layernorm` 可训练但前向没用到，保存时没有 optimizer 状态；加载时 torch `_init_optim_state` 为所有可训练参数造状态，键对不上。ACT（参数全参与）的 FSDP resume 通过 | LeRobot / accelerate 问题：带「可训练但未使用」参数的模型无法 FSDP resume。与 ROCm 无关。未提交 |
| `smolvla-resume-fsdp2` | 用的是 SmolVLA 整层自动切分的早期配方（Tensor/DTensor 混用）。可用配方（只切 `SmolVLMEncoderLayer`）找到后没有重跑 | 未验证 |
| `g05-fsdp2`、`cover-wallx-fsdp2`、`cover-fineart-fsdp2` | `FSDP_DTYPE_REPORT` 显示 G05、WALL-OSS flow 的单个 Transformer 层内同时有 bf16 和 fp32 参数，FineART 的根单元混有 bf16 PaliGemma 和 fp32 动作头。统一成 fp32 或 bf16 后，模型代码里写死的 dtype 又对不上（`mat1 and mat2 must have the same dtype`、`Index put requires the source and destination dtypes match`） | 需要模型侧声明 `_fsdp_wrap_modules` / 处理 dtype。DDP 正常 |
| `cover-longwam-fsdp2`、`cover-longwam-fsdp8` | 自动切分把 Wan VAE 残差块也切了，特征缓存尺寸不对；只切 `WanTransformerBlock`、`UMT5Block` 后又报 Tensor/DTensor 混用 | 同上 |
| `act-ddp2-to-1-resume` | `Parallelism degrees do not multiply to the world size: dp_replicate=2 ... != WORLD_SIZE=1`。checkpoint 记录了 `dp_replicate=2`，单卡恢复需要显式 `--parallelism.dp_replicate=1` | 用法要求；加参数后没有重跑 |
| `dit-fsdp2-deploy` | FSDP 训练部分通过。单卡加载推理那一步同时设了 `CUDA_VISIBLE_DEVICES` 和 `HIP_VISIBLE_DEVICES`，ROCm 运行时 abort（`Conflicting visibility of agent`） | 测试脚本错误；部署未验证 |

其他：

- 队列 B 的执行脚本在运行中被覆盖，跑完最后一项后解析出错、没写 `QUEUE END`，手动补写。以后不在运行中改脚本。
- Hugging Face 缓存的大文件在共享的 `hub/blobs`，删模型目录不释放空间。`prune-hf-models.py` 只删不被其他模型引用的 blob，清了 Qwen3-VL-8B、Robometer-4B、X-VLA、Qwen3-VL-2B、InternVL3-1B、VLA-JEPA 共 42.3 GB。
- 这一轮原定 5 小时，实际约 6 小时 15 分。

## 14. 夜间验证（2026-11-22 20:20 – 11-23 01:25，UTC+8）

脚本在 `tmp/rocm-train/dist/`，日志 DOUni `run-dist-{l,m,n,t,v,y1,s2,x,u,u2,u3,w,z,p,r,e1,e2,e3,e4,q,q2,s3,k2,s4}.log`。原计划 06:30 停发新任务，实际最后一批在 01:25 前跑完；之后本机休眠到 08:15，收尾晚了。

### 14.1 结论先行

1. **ROCm 多卡扩展差的根因是 CPU 亲和性**：HIP 运行时初始化时把进程的 CPU 亲和性从 160 个核缩到 8 个（`torch.cuda.device_count()` 前后 160 → 8）。`accelerate launch` 为了数 GPU 会初始化 HIP，所有 rank 和 DataLoader worker 继承这 8 个核。**设置 `AMD_CPU_AFFINITY=0` 后解决**（`HSA_OVERRIDE_CPU_AFFINITY_DEBUG` 无效）：

   | 模型（每卡 batch） | 卡数 | 默认（smp/s） | `AMD_CPU_AFFINITY=0`（smp/s） |
   |---|---:|---:|---:|
   | ACT（32） | 1 / 2 / 4 / 8 | 584 / 1056 / 968 / 706 | 579 / 1066 / 2112 / **4122** |
   | SmolVLA（8） | 1 / 4 / 8 | 80 / 173 / 137 | — / 269 / **506** |
   | π0（4） | 1 / 4 / 8 | 9 / 26 / 21 | — / 28 / **53** |

   排查过程：RCCL all-reduce 正常（8 卡 256 MB busbw 278 GB/s，xGMI 全互联）；8 个独立单卡任务并发不变慢；profiler 显示 DDP 时 GPU 时间不变（494 → 519 ms/10 步）而 CPU 时间涨 3 倍（509 → 1557 ms）；`num_workers=0` 时 DDP 不变慢；去掉 `torchao`（worker 里会经 `transformers → torchao → has_triton()` 建 HIP 上下文）不改善；`taskset` 看到 rank 进程都是 `0-7`；最后定位到 HIP 初始化和 `AMD_CPU_AFFINITY`。
2. **MIOpen 冷启动**：卷积骨干（ACT 的 ResNet18）第一步编译 kernel 464 s，之后 0.053 s/步。容器 `--rm` 会丢缓存。挂持久目录并设 `MIOPEN_USER_DB_PATH`、`MIOPEN_CUSTOM_CACHE_DIR` 后，下一次第一步 27.5 s，整次 502 s → 66 s。`MIOPEN_FIND_MODE=1` 稳态不更快（0.060 vs 0.052 s）。
3. **在 ROCm 上从头训出的策略和官方相当**：ACT 按官方配方（1 卡、batch 8、10 万步，74 分钟）在 AlohaTransferCube 50 局成功 66%（33/50，CI 52.2–77.6）；官方 `lerobot/act_aloha_sim_transfer_cube_human` 在同一 ROCm 评估下 76%（38/50，CI 62.6–85.7），区间重叠。2 卡 batch 64、1.25 万步的版本只有 48%（24/50），更新次数是官方的 1/8。
4. **官方 VLA 检查点的 LIBERO 成功率低于文档**：EVO1（文档参数原样，revision 515921f4）LIBERO-Spatial 84%（42/50），文档 97%（485/500）；SmolVLA `lerobot/smolvla_libero` 60%（30/50），文档无参考值。MI300X 没有图形上下文，只能用 OSMesa 软件渲染，参考结果是 EGL。**GPU/CPU 数值对比（14.4）排除了 ROCm 数值误差**，差距来自渲染、仿真环境或评估本身的方差。

### 14.2 其他结果

| 项目 | 结果 | 说明 |
|---|---|---|
| 上一轮 3 项补跑 | 全 PASS | DDP2 → 单卡 resume（需 `--parallelism.dp_replicate=1`）；FSDP2 训练后单卡 `from_pretrained` + `select_action`（动作 `(1, 14)`）；SmolVLA FSDP resume（只切 `SmolVLMEncoderLayer`） |
| `nat_vla`（上游新策略，最新 `main` 经 `PYTHONPATH`） | 单卡 / DDP2 / FSDP2 PASS | LIBERO episode 0 |
| 最新 `main`（`e5be0c16`）回归 | 18/18 PASS | ACT（1 卡 / DDP2 / FSDP2）、DiT、SmolVLA、π0.5、GR00T、X-VLA（1 卡 / DDP2）、WALL-OSS 0.5（1 卡 / FSDP2） |
| 按参数归属切分 FSDP（`FSDP_WRAP_STRATEGY=params`，每个直接持有参数的模块一个单元） | WALL-OSS flow PASS；G05、FineART、Long-WAM FAIL | 原始混合 dtype 下能过 FSDP 的 dtype 检查。G05 把 `conv1d` 权重直接传给 `causal_conv1d` kernel；FineART 的融合交叉熵直接取 `lm_head` 权重；Long-WAM 的 MoT 块直接用参数 |
| SARM | DDP2 PASS；FSDP 不支持 | LeRobot 明确报错：奖励模型没有 FSDP 单元和分片保存路径 |
| resume 组合（ACT） | 全 PASS | EMA + DDP、EMA + FSDP、fp16 + DDP（GradScaler）、梯度累积 + FSDP |
| `praxis`（上游新策略，27.2B LoRA） | FAIL | 按文档的 `rename_map`（Cartesian 列替换 `observation.state`）报键冲突；把原键挪走后，`lerobot-train` 在 processor 里第二次应用 map 又冲突。文档与代码不一致，未做本地 DROID 副本 |
| `lerobot/diffusion_pusht` 评估 | FAIL | 旧格式，`migrate_policy_normalization` 保存配置时 `Couldn't encode 84`；最新 `main` 同样失败 |
| GR00T LIBERO 检查点评估 | FAIL | 检查点图像特征 `(256, 256, 3)`（HWC），环境给 `(3, 360, 360)`，特征检查拒绝 |

### 14.3 上游提交（2026-11-23）

| 链接 | 内容 |
|---|---|
| PR [#5471](https://github.com/huggingface/lerobot/pull/5471) | RA-BC 在 DDP 下崩溃：`make_sample_weighter` 解包 `DistributedDataParallel`，带单进程 gloo 单测 |
| PR [#5472](https://github.com/huggingface/lerobot/pull/5472) | `docker/Dockerfile.rocm` 加 `ENV AMD_CPU_AFFINITY=0` |
| PR [#5473](https://github.com/huggingface/lerobot/pull/5473) | `torch_accelerators.mdx`：ROCm 已知问题（CPU 亲和性写成部分 ROCm 版本的回归、MIOpen 缓存等）和 ROCm 上训过的策略 |
| issue [#5474](https://github.com/huggingface/lerobot/issues/5474) | FSDP resume 在有「可训练但无梯度」参数时失败；纯 torch CPU gloo 复现（`dist/fsdp_unused_param_repro.py`），与设备无关 |
| issue [#5475](https://github.com/huggingface/lerobot/issues/5475) | `--accelerator.fsdp.min_num_params` 在 FSDP2 下失败（`fully_shard` 拒绝 ModuleList）；cpu_offload 需 `cpu:gloo,cuda:nccl` |
| issue [#5476](https://github.com/huggingface/lerobot/issues/5476) | Praxis DROID 文档的 `rename_map` 自 #5316 起失效，且 map 被应用两次 |
| [ROCm/rocm-systems#6712 留言](https://github.com/ROCm/rocm-systems/pull/6712#issuecomment-7088991627) | `AMD_CPU_AFFINITY` 默认开启对 PyTorch DDP 的影响数据（ACT 8 卡 706 → 4122 smp/s） |

旧文档分支 `docs/rocm-known-issues` 已被 #5473 取代。

### 14.4 LIBERO 数值对比：ROCm fp32 推理准确

`dist/numerics_check.py`：同一检查点在 CPU fp32 和 GPU 上各跑一遍，16 帧 LIBERO 数据，每帧传入同一份固定噪声，比较整个动作块。

| 模型 | 对比 | max_abs | 相对 mean_abs |
|---|---|---:|---:|
| ACT | GPU fp32 vs CPU fp32 | 2.7e-6 | — |
| SmolVLA（`lerobot/smolvla_libero`） | GPU fp32 重复两次 | 0 | 0 |
| SmolVLA | GPU fp32 vs CPU fp32 | 9.1e-6 | 6.9e-7 |
| SmolVLA | GPU fp32（math SDPA）vs CPU fp32 | 6.9e-6 | 6.4e-7 |
| SmolVLA | GPU 默认 dtype（bf16）vs CPU fp32 | 2.6e-2 | 4.3e-3 |

中间一度看到 GPU/CPU 相差 7.5%，原因是脚本 bug：用 `__code__.co_varnames` 判断 `predict_action_chunk` 是否接受 `noise`，而它被装饰器包装，只看得到 `*args, **kwargs`，噪声没传进去，模型在各自设备上采样。改用 `inspect.signature` 后差距消失。结论：EVO1 / SmolVLA 的 LIBERO 成功率偏低不是 ROCm 数值问题。
