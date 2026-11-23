# LeRobot v0.7.x 在 AMD Instinct MI300X（ROCm）上的训练实践报告

2026-11-23 · 作者 Alex He（GitHub [alexhegit](https://github.com/alexhegit)） · 过程记录、脚本和日志：[alexhegit/lerobot-rocm](https://github.com/alexhegit/lerobot-rocm)

## 摘要

- **覆盖**：LeRobot 0.7 的 27 个策略（含上游新加的 `nat_vla`）在 MI300X 上单卡训练和 2 卡 DDP 训练全部通过；FLUX3 在当前 `main` 上因 NATTEN 没有 ROCm 路径而受阻，用上游 PR #5213 可以训练；Praxis 因文档与代码不一致没有跑通。4 个奖励模型和 HIL-SERL 也通过。
- **分布式**：DDP 最多 8 卡；FSDP2 有 24 个策略通过（另有 FLUX3 用 #5213 通过），其中 ACT、π0、FastWAM、WALL-OSS 0.5、Cosmos3 Edge 到 8 卡；HSDP 2×2 / 2×4 / 4×2；梯度累积、fp16 / bf16、FSDP 混合精度策略、CPU offload、DDP 选项、断点续训（含换分片数）、DCP checkpoint 格式、训练中多卡仿真评估都通过。只有 G05、FineART-VLA、Long-WAM 的 FSDP 需要模型侧改动。
- **最重要的性能发现**：多卡训练必须设 `AMD_CPU_AFFINITY=0`。不设时 HIP 运行时把所有 rank 和 DataLoader worker 限制在 8 个 CPU 核上，ACT 8 卡只有 706 samples/s；设了之后是 4,122 samples/s（单卡的 7.1 倍）。
- **效果与数值**：按官方配方在 ROCm 上从头训练的 ACT，仿真成功率 66%；官方检查点在同一 ROCm 评估下是 76%，95% 置信区间重叠。ACT 和 SmolVLA 的 GPU fp32 推理与 CPU 的最大差小于 1e-5。
- **上游贡献**：向 LeRobot 提交 5 个 PR、3 个 issue，向 NATTEN 提交 1 个 issue，在 ROCm/rocm-systems 的 PR 下补充了 PyTorch 多卡数据，并在 LeRobot 上游 PR 下留言了 ROCm 验证结果。
- **未做**：多机训练（没有 InfiniBand 互联的机器）。

## 1. 环境

| 项 | 值 |
|---|---|
| 加速器 | 8× AMD Instinct MI300X VF（`gfx942`），每卡 192 GB HBM3，xGMI 全互联 |
| CPU | 160 核 |
| 镜像 | 用仓库自带 `docker/Dockerfile.rocm` 构建，`LEROBOT_EXTRAS=all`，基座 `rocm/primus:v26.4`，未修改 |
| 软件 | Python 3.12，PyTorch `2.12.0+rocm7.14.0a20260608`（HIP 7.14），torchvision `0.27.0+rocm7.14` |
| LeRobot | `v0.7.0` 之后一个提交（版本号 0.7.1）；回归测试另用最新 `main`（`e5be0c16`），新策略（`nat_vla`、`praxis`）用最新 `main` |

原则：不修改 LeRobot 源码，不重建镜像。环境兼容问题用启动参数、环境变量、本地配置或挂载进容器的一次性运行时补丁绕过。`push_to_hub=false`。

## 2. 训练设置

### 2.1 容器

```bash
docker run --rm --ipc=host --shm-size 16g \
  --device=/dev/kfd --device=/dev/dri \
  --group-add <video-gid> --group-add <render-gid> \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e AMD_CPU_AFFINITY=0 \
  -e MIOPEN_FIND_MODE=2 \
  -e MIOPEN_USER_DB_PATH=/cache/miopen -e MIOPEN_CUSTOM_CACHE_DIR=/cache/miopen \
  -v <outputs>:/outputs -v <miopen-cache>:/cache/miopen \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lerobot-rocm bash /outputs/jobs/<job>.sh
```

- 设备仍写 `--policy.device=cuda`（PyTorch 的 ROCm 版沿用 `cuda` 设备名），用 `HIP_VISIBLE_DEVICES` 选卡。不要同时设 `CUDA_VISIBLE_DEVICES`，两者冲突时 ROCm 运行时会 abort（`Conflicting visibility of agent`）。
- `--group-add` 要用数字 gid，部分主机上 `render` 组名不能解析。
- `MIOPEN_FIND_MODE` 要写数字 `2`，写字符串 `FAST` 不生效。
- 无头仿真评估：MI300X 没有图形上下文，EGL 不可用，需要在容器里装 `libosmesa6 libglu1-mesa` 并设 `MUJOCO_GL=osmesa`。

### 2.2 单卡 smoke 测试

```bash
lerobot-train --policy.path=<hub-or-local> --policy.device=cuda \
  --dataset.repo_id=<dataset> --dataset.episodes='[0]' \
  --steps=2 --save_freq=2 --log_freq=1 --batch_size=1 --num_workers=0 \
  --env_eval_freq=0 --policy.push_to_hub=false --output_dir=...
```

通过标准：退出码 0；日志里有每步 loss；第 2 步的 checkpoint（`model.safetensors`，或 FSDP 的 `*.distcp`）和 `training_step.json` 写完。只验证"能训练"，不评估策略质量；质量单独在第 6 节验证。

数据集：`lerobot/aloha_sim_transfer_cube_human`（视觉策略）、`lerobot/svla_so100_pickplace`（多数 VLA）、`lerobot/libero`（FastWAM、DM05、LingBot-VA、`nat_vla`）、`lerobot/pusht`（TD-MPC、reward_classifier）。

### 2.3 多卡

由同一个单卡脚本生成，只换启动器、加并行参数：

| 模式 | 命令差异 |
|---|---|
| DDP | `accelerate launch --num_processes=N $(which lerobot-train) ...`（N = 2 / 4 / 8） |
| FSDP2 | 再加 `--parallelism.dp_shard=N`；`dp_shard=-1` 自动推断 |
| HSDP | `--parallelism.dp_replicate=R --parallelism.dp_shard=S`，R×S = 卡数 |
| FSDP 切分单元 | 策略未声明 `_fsdp_wrap_modules` 时加 `--accelerator.fsdp.wrap_modules='["LayerClass", ...]'` |
| 混合 dtype 的 VLA 上 FSDP | `--policy.dtype=float32 --accelerator.mixed_precision=bf16` |
| FSDP CPU offload | `--accelerator.fsdp.cpu_offload=true --accelerator.init_process_group.backend=cpu:gloo,cuda:nccl` |
| DDP checkpoint 恢复到单卡 | `--resume=true --parallelism.dp_replicate=1` |

多卡 smoke 的通过标准再加一条：`training_step.json` 里的 `dp_world_size` 等于进程数。

## 3. 模型覆盖

"FSDP2" 指 2 卡 `dp_shard=2`，括号里是需要的额外设置。

### 3.1 策略

| 策略 | 单卡 | DDP | FSDP2 | 更大规模 |
|---|---|---|---|---|
| ACT | PASS | 2 | PASS（自带单元） | FSDP 8 |
| Diffusion | PASS | 2 | PASS | DDP 8 |
| DiT（`multi_task_dit`） | PASS | 2 / 4 | PASS（`CLIPEncoderLayer`、`TransformerBlock`） | FSDP 4，HSDP 2×2 / 4×2 |
| VQ-BeT | PASS | 2 | PASS（`Block`） | — |
| TD-MPC | PASS | 2 | PASS（`Sequential`） | — |
| SmolVLA | PASS | 2 / 4 / 8 | PASS（只切 `SmolVLMEncoderLayer`，参数统一 fp32 + bf16 autocast） | HSDP 2×2 |
| π0 | PASS（含 `compile_model`） | 2 | PASS（只切 `SiglipEncoderLayer`，fp32 + bf16） | FSDP 8，HSDP 2×4 |
| π0.5 | PASS | 2 | PASS（同 π0） | — |
| π0-FAST | PASS | 2 | PASS（`SiglipEncoderLayer`、Gemma 解码层，fp32 + bf16） | — |
| X-VLA | PASS | 2 | PASS | — |
| EVO1 | PASS | 2 | PASS | — |
| EO1 | PASS | 2 | PASS（fp32 + bf16） | — |
| VLA-JEPA | PASS | 2 | PASS | — |
| GR00T N1.7 | PASS | 2 / 8 | PASS | — |
| MolmoAct2（全参 / LoRA） | PASS | 2 / 8 | PASS（参数统一 bf16） | — |
| DM05 | PASS | 2 | PASS（只切 `SiglipEncoderLayer`，fp32 + bf16） | — |
| LingBot-VA（LoRA） | PASS | 2 | PASS | — |
| LingBot-VLA v2 | PASS | 2 | PASS（自带单元） | — |
| WALL-OSS 0.5 | PASS | 2 | PASS（自带单元，fp32 + bf16） | FSDP 8 |
| FastWAM | PASS | 2 | PASS（自带单元） | FSDP 8 |
| Cosmos3 Edge / Nano | PASS | 2（Nano 要用 bf16，fp32 下 DDP 显存不够） | PASS（自带单元） | Edge FSDP 8 |
| LaWAM | PASS | 2 | PASS（加载后参数统一 fp32） | — |
| WALL-OSS flow（`wall_x`） | PASS | 2 | PASS（按参数归属切分） | — |
| `nat_vla`（最新 `main`） | PASS | 2 | PASS | — |
| FLUX3（LoRA） | 当前 `main` 受阻；PR [#5213](https://github.com/huggingface/lerobot/pull/5213) PASS | 2（#5213） | PASS（#5213，关掉梯度检查点） | — |
| G05（含 CoT） | PASS | 2 | FAIL | — |
| FineART-VLA | PASS | 2 | FAIL | — |
| Long-WAM | PASS | 2 | FAIL | — |
| Praxis（最新 `main`，27.2B LoRA） | FAIL | — | — | — |

FSDP 失败的三个模型都是层内混合 bf16 / fp32 参数，或在 `forward` 之外直接使用子模块的参数，需要模型代码配合，DDP 正常（见 5.3）。Praxis 失败是文档里的 `rename_map` 与代码不一致，与 ROCm 无关（issue [#5476](https://github.com/huggingface/lerobot/issues/5476)）。

最新 `main`（`e5be0c16`）回归：ACT（1 卡 / DDP2 / FSDP2）、DiT、SmolVLA、π0.5、GR00T、X-VLA（1 卡 / DDP2）、WALL-OSS 0.5（1 卡 / FSDP2）共 18 项全部通过。

### 3.2 奖励模型与强化学习

| 模型 | 结果 |
|---|---|
| SARM | 单卡训练 PASS，DDP2 PASS；FSDP 不支持（LeRobot 明确报错） |
| reward_classifier | 单卡训练 PASS；保存后重新加载，推理结果一致 |
| Robometer | 零样本推理 PASS；RA-BC 打分 + SmolVLA 加权训练 PASS |
| TOPReward | 零样本推理 PASS；RA-BC 加权训练 PASS（需要修复，见 PR [#5432](https://github.com/huggingface/lerobot/pull/5432)） |
| HIL-SERL（Gaussian actor + SAC） | GPU 上训练步 PASS |

RA-BC 加权训练：FSDP2 PASS；DDP2 原本崩溃，修复见 PR [#5471](https://github.com/huggingface/lerobot/pull/5471)。

## 4. 训练特性覆盖

| 类别 | 已验证 |
|---|---|
| 加载 | Hub 预训练权重、本地目录、本地修改过的 `config.json` |
| 精度 | fp32、bf16 混合精度、fp16 + GradScaler（DDP 和 FSDP）、fp32 权重 + bf16 autocast、FSDP 混合精度策略（参数 bf16、梯度规约 fp32） |
| 并行 | DDP 2 / 4 / 8，FSDP2 2 / 4 / 8，`dp_shard=-1` 自动推断，HSDP 2×2 / 2×4 / 4×2 |
| FSDP 选项 | 策略自带 `_fsdp_wrap_modules`、命令行 `wrap_modules`、`reshard_after_forward=false`、`cpu_offload` |
| DDP 选项 | `find_unused_parameters=false`、`gradient_as_bucket_view`、`static_graph` |
| 梯度累积 | 2 步累积，DDP 和 FSDP；`sync_each_batch=true` |
| 断点续训 | 单卡、DDP、FSDP；EMA + DDP / FSDP；fp16 + DDP；梯度累积 + FSDP；FSDP 2 卡 → 4 卡换分片；DDP 2 卡 → 单卡；分片 → 非分片会被正确拒绝 |
| checkpoint 格式 | 默认 safetensors、`--checkpoint_format=dcp`、`safetensors_dcp`，均可恢复 |
| 部署 | FSDP2 训练后单卡 `from_pretrained` + `select_action` |
| 数据 | 视频帧解码、`num_workers=2`（`spawn`）、`rename_map`、相机槽位、QUANTILES 归一化 |
| 训练中评估 | MuJoCo 仿真评估，单卡和 DDP2（OSMesa 渲染） |
| 编译 | π0 `compile_model=true`；LingBot-VA flex attention 编译 |
| 微调 | PEFT LoRA（MolmoAct2、LingBot-VA、FLUX3） |
| 样本加权 | RA-BC（Robometer / TOPReward 打分） |

## 5. 发现的问题

### 5.1 ROCm 平台相关

| 问题 | 影响 | 处理 |
|---|---|---|
| **HIP 运行时初始化时把进程 CPU 亲和性缩到 8 核**（ROCm/rocm-systems [#6134](https://github.com/ROCm/rocm-systems/pull/6134) 让 `AMD_CPU_AFFINITY` 默认开启，[#6712](https://github.com/ROCm/rocm-systems/pull/6712) 已回退，但已发布版本仍受影响） | `accelerate launch` 为了数 GPU 会初始化 HIP，所有 rank 和 DataLoader worker 继承这 8 个核；多卡训练 CPU 成为瓶颈，ACT 8 卡比 2 卡还慢 | `AMD_CPU_AFFINITY=0`（`HSA_OVERRIDE_CPU_AFFINITY_DEBUG` 无效）。已提 PR [#5472](https://github.com/huggingface/lerobot/pull/5472) 写进 `Dockerfile.rocm`，并在 [#6712](https://github.com/ROCm/rocm-systems/pull/6712#issuecomment-7088991627) 留言补充数据 |
| MIOpen 冷启动编译卷积 kernel | ACT（ResNet18）第一步 464 s；容器 `--rm` 后缓存丢失 | 挂持久目录并设 `MIOPEN_USER_DB_PATH`、`MIOPEN_CUSTOM_CACHE_DIR`，第一步降到 27.5 s |
| 没有 EGL 图形上下文 | MuJoCo 仿真评估报 `can't create a graphics context on a compute chip` | 用 OSMesa 软件渲染 |
| 镜像里的 aiter 没有 `flash_attn_func`，`flash-attn` 装不上 | 默认要 FlashAttention 的模型要改注意力后端 | 改用 SDPA / `attn_mode=torch` |
| NATTEN 没有 ROCm 轮子；源码构建只有 flex 后端，`get_device_cc()` 在 ROCm 上返回 0，编译版 flex 被禁用；强行打开后 3D 大 mask 的 inductor kernel 启动失败 | FLUX3 的 Video VAE 物化 dense mask，试图分配 882 GiB | 用 PR [#5213](https://github.com/huggingface/lerobot/pull/5213) 的纯 PyTorch 邻域注意力；已提 [SHI-Labs/NATTEN#354](https://github.com/SHI-Labs/NATTEN/issues/354) |
| `CUDA_VISIBLE_DEVICES` 与 `HIP_VISIBLE_DEVICES` 同时设置 | ROCm 运行时 abort | 只用 `HIP_VISIBLE_DEVICES` |
| 镜像里 `torchao 0.15`，PEFT 要求 `> 0.16` | 所有 LoRA 训练失败 | 在容器里 `uv pip install 'torchao>0.16'`，装完检查 `torch.version.hip` 仍在 |

### 5.2 LeRobot 框架问题（与平台无关）

| 问题 | 状态 |
|---|---|
| RA-BC 在 DDP 下崩溃：sample weighter 在 `accelerator.prepare` 之后用被包装的模型创建，读不到 `policy.config` | 修复 PR [#5471](https://github.com/huggingface/lerobot/pull/5471)，附单进程 gloo 单测 |
| RA-BC 用于不支持 `forward(reduction="none")` 的策略（ACT、Diffusion）时，报错晦涩 | 修复 PR [#5433](https://github.com/huggingface/lerobot/pull/5433)，提前给出清楚的报错 |
| TOPReward 的 RA-BC 打分从 1 帧前缀开始，Qwen3-VL 要求至少 2 帧 | 修复 PR [#5432](https://github.com/huggingface/lerobot/pull/5432) |
| FSDP 断点续训：策略有可训练但没有梯度的参数时失败（`Missing key ... optimizer.state`），例如 DiT 的 CLIP `post_layernorm` | issue [#5474](https://github.com/huggingface/lerobot/issues/5474)，附纯 PyTorch CPU 复现，证明与设备无关 |
| `--accelerator.fsdp.min_num_params` 在 FSDP2 下不可用：按参数量会选中 `nn.ModuleList`，`fully_shard` 拒绝；CPU offload 需要额外设 CPU 进程组后端 | issue [#5475](https://github.com/huggingface/lerobot/issues/5475) |
| Praxis DROID 文档里的 `rename_map` 自 #5316 起失效，且 map 被应用两次，任何写法都会冲突 | issue [#5476](https://github.com/huggingface/lerobot/issues/5476) |
| FLUX3 把 NATTEN 的 `backend` 放进 `attention_kwargs`，NATTEN 只读顶层参数，所以后端选择不生效 | 已由他人的 PR [#5213](https://github.com/huggingface/lerobot/pull/5213) 修复；本地同类修复不再重复提交，已[留言](https://github.com/huggingface/lerobot/pull/5213#issuecomment-7029511117) ROCm 结果 |
| 旧格式检查点 `lerobot/diffusion_pusht` 迁移时报 `Couldn't encode 84`（最新 `main` 也一样） | 已记录，未提交 |
| GR00T LIBERO 检查点的图像特征是 HWC `(256, 256, 3)`，环境给 CHW `(3, 360, 360)`，评估被特征检查拒绝 | 已记录，未提交 |
| 命令行 `--policy.align_params='{}'` 不能清空检查点里的字典（draccus 合并字典） | 改本地 `config.json` 绕过 |

### 5.3 FSDP 的使用限制

- 只有 ACT、WALL-OSS 0.5、FastWAM、LingBot-VLA v2、LaWAM、Cosmos3 声明了 `_fsdp_wrap_modules`，其他策略要在命令行给 `wrap_modules`。
- **混合 dtype**：多数 VLA 主干是 bf16、动作头或 norm 是 fp32，FSDP2 要求同一单元内 dtype 一致。用 `--policy.dtype=float32 --accelerator.mixed_precision=bf16`，或在切分前把参数统一成一种 dtype。
- **绕过 `forward`**：π0、π0.5、SmolVLA、DM05 的联合注意力直接调用语言层的子模块，FSDP 的 all-gather 钩子不触发（`mixed torch.Tensor and DTensor`）。只把视觉编码层作为单元，语言层留在根单元。
- **梯度检查点**：FLUX3 打开时 FSDP 报 `CheckpointError`，关掉后通过。
- **需要模型侧改动**：G05、FineART-VLA、Long-WAM 在层内混合 dtype，或直接把子模块权重传给 kernel / 融合算子，不能靠启动参数解决。WALL-OSS flow 按"每个直接持有参数的模块一个单元"切分后通过。

### 5.4 Hub 检查点与 0.7.1 代码不匹配（配置层面）

- EO1、MolmoAct2 的 Hub `config.json` 有 0.7.1 不认识的字段，需要本地删掉。
- π0.5、G05、WALL-OSS 的 QUANTILES 归一化需要 q01/q99，SO100 数据集没有，需要本地计算后写进 `meta/stats.json`。
- G05 默认 language recipe 需要 `atomic_task` 字段；FastWAM 要求 7 维动作、8 维状态，SO100 不满足，改用 LIBERO。

## 6. 性能数据

### 6.1 多卡扩展（`AMD_CPU_AFFINITY` 的影响）

吞吐 = 每秒训练样本数（samples/s），取稳态步。括号里是每卡 batch。

| 模型（每卡 batch） | 卡数 | 默认 | `AMD_CPU_AFFINITY=0` |
|---|---:|---:|---:|
| ACT（32） | 1 / 2 / 4 / 8 | 584 / 1,056 / 968 / 706 | 579 / 1,066 / 2,112 / **4,122** |
| SmolVLA（8） | 1 / 4 / 8 | 80 / 173 / 137 | — / 269 / **506** |
| π0（4） | 1 / 4 / 8 | 9 / 26 / 21 | — / 28 / **53** |

设置 `AMD_CPU_AFFINITY=0` 后，ACT 8 卡是单卡的 7.1 倍（扩展效率 89%），SmolVLA 8 卡是 4 卡的 1.9 倍。

排查中排除的因素：

- RCCL all-reduce 正常：8 卡、256 MB 消息，总线带宽 278 GB/s。
- 8 个独立的单卡任务同时运行不变慢，说明不是 GPU 或显存带宽争用。
- profiler：ACT DDP 时 GPU 时间基本不变（10 步 494 → 519 ms），CPU 时间涨到 3 倍（509 → 1,557 ms）。
- `num_workers=0` 时 DDP 不变慢；`taskset` 显示每个 rank 进程都被限制在 CPU `0-7`。

### 6.2 MIOpen 缓存

| 场景（ACT，单卡） | 第一步 | 整次任务 |
|---|---:|---:|
| 无持久缓存 | 464 s | 502 s |
| 持久缓存命中 | 27.5 s | 66 s |
| 稳态 | 0.053 s/步 | — |

`MIOPEN_FIND_MODE=1`（完整查找）的稳态没有更快（0.060 vs 0.052 s/步），保持 `2` 即可。

### 6.3 大模型 smoke 的显存峰值（2 卡 DDP，batch 1，每卡）

| 模型 | 峰值显存 |
|---|---:|
| Long-WAM | 79.4 GB |
| DM05 | 51.0 GB |
| WALL-OSS flow | 48.0 GB |
| FineART-VLA | 42.5 GB |
| GR00T N1.7 | 42.1 GB |
| FLUX3（LoRA，#5213） | 27.8 GB |
| LingBot-VA（LoRA） | 14.3 GB |

Cosmos3 Nano 在 fp32 下 2 卡 DDP 每卡用到约 184 GB 后 OOM，改 bf16 后通过；单卡和 FSDP2 在 fp32 下就能通过。

## 7. 训练效果与数值正确性

### 7.1 从头训练

| 训练 | 设置 | AlohaTransferCube 成功率（50 局，95% CI） |
|---|---|---|
| ACT，官方配方，ROCm 上训练 | 1 卡、batch 8、100k 步，74 分钟 | 66%（33/50，52.2–77.6） |
| 官方检查点 `lerobot/act_aloha_sim_transfer_cube_human` | 同一 ROCm 评估 | 76%（38/50，62.6–85.7） |
| ACT，2 卡 batch 64、12.5k 步 | 更新次数只有官方的 1/8 | 48%（24/50） |

两者置信区间重叠，ROCm 上训练出的策略和官方检查点相当。

### 7.2 官方 VLA 检查点的 LIBERO 评估

| 检查点 | LIBERO-Spatial（50 局） | 文档参考 |
|---|---:|---:|
| EVO1（文档参数原样） | 84%（42/50） | 97%（485/500） |
| SmolVLA `lerobot/smolvla_libero` | 60%（30/50） | 无 |

### 7.3 GPU / CPU 数值对比

同一检查点在 CPU fp32 和 MI300X 上各推理 16 帧 LIBERO 数据，每帧传入同一份固定噪声，比较整个动作块：

| 模型 | 对比 | 最大绝对差 | 相对平均差 |
|---|---|---:|---:|
| ACT | GPU fp32 vs CPU fp32 | 2.7e-6 | — |
| SmolVLA | GPU fp32 两次运行 | 0 | 0 |
| SmolVLA | GPU fp32 vs CPU fp32 | 9.1e-6 | 6.9e-7 |
| SmolVLA | GPU fp32（math SDPA）vs CPU fp32 | 6.9e-6 | 6.4e-7 |
| SmolVLA | GPU bf16 vs CPU fp32 | 2.6e-2 | 4.3e-3 |

ROCm 的 fp32 推理准确，可以排除数值误差。7.2 的成功率差距更可能来自 OSMesa 和 EGL 的渲染差异、仿真环境，或 50 局评估本身的方差。

## 8. 上游贡献

| 链接 | 类型 | 内容 | 状态（2026-11-23） |
|---|---|---|---|
| [huggingface/lerobot#5432](https://github.com/huggingface/lerobot/pull/5432) | PR | TOPReward 单帧视频补帧 | 等待评审 |
| [huggingface/lerobot#5433](https://github.com/huggingface/lerobot/pull/5433) | PR | RA-BC 不支持的策略提前报错 | 等待评审，社区已复现 |
| [huggingface/lerobot#5471](https://github.com/huggingface/lerobot/pull/5471) | PR | RA-BC 在 DDP 下解包模型 | 等待评审 |
| [huggingface/lerobot#5472](https://github.com/huggingface/lerobot/pull/5472) | PR | `Dockerfile.rocm` 设 `AMD_CPU_AFFINITY=0` | 等待评审 |
| [huggingface/lerobot#5473](https://github.com/huggingface/lerobot/pull/5473) | PR | 文档：ROCm 已知问题、ROCm 上训练过的策略 | 等待评审 |
| [huggingface/lerobot#5474](https://github.com/huggingface/lerobot/issues/5474) | issue | FSDP 断点续训与无梯度参数 | 已提交 |
| [huggingface/lerobot#5475](https://github.com/huggingface/lerobot/issues/5475) | issue | FSDP2 下 `min_num_params` 不可用 | 已提交 |
| [huggingface/lerobot#5476](https://github.com/huggingface/lerobot/issues/5476) | issue | Praxis `rename_map` | 已提交 |
| [huggingface/lerobot#5213](https://github.com/huggingface/lerobot/pull/5213#issuecomment-7029511117) | 留言 | FLUX3 不依赖 NATTEN 的 PR 在 ROCm 上的验证结果 | 已留言 |
| [SHI-Labs/NATTEN#354](https://github.com/SHI-Labs/NATTEN/issues/354) | issue | ROCm 上编译版 flex 被禁用，3D mask kernel 失败 | 已提交 |
| [ROCm/rocm-systems#6712](https://github.com/ROCm/rocm-systems/pull/6712#issuecomment-7088991627) | 留言 | `AMD_CPU_AFFINITY` 默认开启对 PyTorch 多卡训练的影响数据 | 已留言 |

所有 PR 都注明只在 AMD MI300X 上测试，没有 NVIDIA GPU。

## 9. 建议

1. 在 ROCm 上做多卡训练时设 `AMD_CPU_AFFINITY=0`（PR #5472 合并后镜像默认就有）。
2. 卷积类策略挂持久 MIOpen 缓存目录；`MIOPEN_FIND_MODE=2`。
3. 用 LoRA 时把 `torchao` 升到 0.16 以上。
4. 仿真评估用 `MUJOCO_GL=osmesa`。
5. 大 VLA 做 FSDP：用 fp32 权重 + bf16 autocast；没有自带切分单元的策略按 3.1 表给 `wrap_modules`；不要用 `min_num_params`。

## 10. 未完成

- 多机训练：现有机器之间没有 InfiniBand / RoCE 互联，留待后续。
- FLUX3 在当前 `main` 上（等 PR #5213 合并或 NATTEN 支持 ROCm）。
- G05、FineART-VLA、Long-WAM 的 FSDP（需要模型侧改动）。
- Praxis 训练（等 #5476）。
