# LeRobot on AMD ROCm

在 AMD Instinct MI300X（ROCm）上验证 [LeRobot](https://github.com/huggingface/lerobot) 0.7 训练的过程记录、脚本和结论。

## Summary (English)

- **Coverage:** on 8× MI300X (ROCm 7.14, PyTorch 2.12), every policy in LeRobot 0.7 trains on a single GPU and on 2-GPU DDP, except FLUX3 on `main`. Many large models also pass FSDP, and Diffusion and ACT pass on 8 GPUs. Resume, sim eval, `torch.compile` and LoRA work.
- **Reward models:** SARM and the reward classifier train, HIL-SERL runs, and the RA-BC flow works end to end with Robometer and TOPReward.
- **FLUX3:** blocked on `main` because NATTEN has no usable ROCm path. With upstream PR [huggingface/lerobot#5213](https://github.com/huggingface/lerobot/pull/5213) it trains on ROCm without NATTEN (1 GPU and DDP 2).
- **Upstream:** PRs [#5432](https://github.com/huggingface/lerobot/pull/5432) and [#5433](https://github.com/huggingface/lerobot/pull/5433), ROCm results on [#5213](https://github.com/huggingface/lerobot/pull/5213#issuecomment-7029511117), issue [SHI-Labs/NATTEN#354](https://github.com/SHI-Labs/NATTEN/issues/354).
- **Distributed (2026-11-22):** 59 of 69 multi-GPU jobs pass. DDP up to 8 GPUs; FSDP2 on 2 GPUs for 21 policies and on 8 GPUs for π0, FastWAM, WALL-OSS 0.5 and Cosmos3; HSDP 2×2 / 2×4 / 4×2; gradient accumulation, fp16 / bf16 / FSDP mixed-precision policy, CPU offload, DDP options, resume (DDP, FSDP, FSDP 2→4 reshard), DCP checkpoint formats, multi-GPU eval. FSDP fails for G05, FineART, WALL-OSS flow and Long-WAM (mixed dtypes inside layers). Found two LeRobot issues: RA-BC crashes under DDP, and FSDP resume fails for models with unused trainable parameters.
- **Multi-GPU performance (2026-11-23):** set `AMD_CPU_AFFINITY=0`. Otherwise the HIP runtime pins every process that initialises it to 8 of 160 CPUs, and `accelerate launch` passes that to all ranks and DataLoader workers: ACT on 8 GPUs goes from 706 to 4,122 samples/s (7.1× one GPU), SmolVLA from 137 to 506, π0 from 21 to 53. Mount a persistent MIOpen cache for conv backbones (first step 464 s → 28 s).
- **Training quality:** ACT trained on ROCm with the official recipe scores 66% (33/50) on AlohaTransferCube; the official checkpoint scores 76% (38/50) under the same ROCm evaluation (overlapping 95% intervals).
- **Numerics:** fp32 inference on ROCm matches CPU (max abs diff 2.7e-6 for ACT, 9.1e-6 for SmolVLA on LIBERO frames with fixed noise), so the lower LIBERO scores of official checkpoints (EVO1 84% vs 97% documented, SmolVLA 60%) are not a ROCm numerics issue.
- **Upstream (2026-11-23):** PRs [#5471](https://github.com/huggingface/lerobot/pull/5471), [#5472](https://github.com/huggingface/lerobot/pull/5472), [#5473](https://github.com/huggingface/lerobot/pull/5473); issues [#5474](https://github.com/huggingface/lerobot/issues/5474), [#5475](https://github.com/huggingface/lerobot/issues/5475), [#5476](https://github.com/huggingface/lerobot/issues/5476); comment on [ROCm/rocm-systems#6712](https://github.com/ROCm/rocm-systems/pull/6712#issuecomment-7088991627).
- **Open:** multi-node (no InfiniBand).

## 验证范围

通过标准：`steps=2`，有 loss，第 2 步 checkpoint 写完。只验证"能训练"，不评估策略质量。没有改 LeRobot 源码，没有重建镜像；兼容问题用启动参数、本地配置或一次性运行时补丁绕过。

| 类别 | 模型 | 单卡 | 多卡 |
|---|---|---|---|
| 策略 | Diffusion | PASS | DDP 2 / DDP 8 |
| 策略 | ACT | PASS | FSDP 2 / FSDP 8 |
| 策略 | Multi-task DiT | PASS | DDP 2 |
| 策略 | SmolVLA、X-VLA、π0、π0.5、π0-FAST、EO1、G05（含 CoT）、MolmoAct2（fft 与 LoRA） | PASS | DDP 2 |
| 策略 | WALL-OSS 0.5、FastWAM、Cosmos3 Edge / Nano、LaWAM、LingBot-VLA v2 | PASS | DDP 2 / FSDP 2 |
| 策略 | VQ-BeT、TD-MPC、DM05、EVO1、VLA-JEPA、Long-WAM、LingBot-VA（LoRA）、FineART-VLA、GR00T N1.7、WALL-OSS flow | PASS | DDP 2 |
| 策略 | FLUX3（LoRA） | `main` BLOCKED；PR #5213 PASS | PR #5213 DDP 2 / FSDP 2 |
| 奖励模型 | SARM、reward_classifier | PASS（训练） | — |
| 奖励模型 | Robometer、TOPReward | 零样本推理 + RA-BC 加权训练 PASS | — |
| RL | HIL-SERL（Gaussian actor + SAC） | PASS | — |

训练特性：从 checkpoint 恢复、训练中仿真评估（MuJoCo + OSMesa）、`num_workers=2`、`torch.compile`（π0）、bf16 混合精度、PEFT LoRA。

完整结果、每个模型的 loss、失败原因和绕过方法见 [reports/smoke-report.md](reports/smoke-report.md)。

## 分布式训练

完整结果见报告第 13 节。要点：

- **FSDP 切分单元**：只有 ACT、WALL-OSS 0.5、FastWAM、LingBot-VLA v2、LaWAM、Cosmos3 声明了 `_fsdp_wrap_modules`。其他策略用 `--accelerator.fsdp.wrap_modules='["类名", ...]'`。`--accelerator.fsdp.min_num_params` 在 FSDP2 下会把 `ModuleList` 当单元而报错，不可用。各模型可用的类名在报告 13.1。`scripts/dist/fsdp_autowrap_train.py` 能自动挑出重复的层类。
- **混合 dtype**：bf16 主干 + fp32 动作头的策略，FSDP2 要求每个单元一种 dtype。有 `dtype` 选项的用 `--policy.dtype=float32 --accelerator.mixed_precision=bf16`。
- **π0 / π0.5 / SmolVLA / DM05**：语言层通过子模块调用，只能把视觉编码层作为 FSDP 单元。
- **CPU offload**：加 `--accelerator.init_process_group.backend=cpu:gloo,cuda:nccl`。
- **resume**：DDP、FSDP、FSDP 换分片数都可以；分片与非分片之间会被拒绝；DDP 换卡数要显式改 `--parallelism.dp_replicate`。

发现的 LeRobot 问题（与 ROCm 无关）：

| 问题 | 状态 |
|---|---|
| RA-BC（`--sample_weighting`）在 DDP 下报 `'DistributedDataParallel' object has no attribute 'config'` | 本地修复已验证，未提交 |
| 有「可训练但未使用」参数的模型（如 DiT）FSDP resume 报 `Missing key in checkpoint state_dict` | 已定位，未提交 |
| `--accelerator.fsdp.min_num_params` 在 FSDP2 下不可用 | 已记录 |
| FSDP `cpu_offload` 需要手动加 gloo 后端 | 已记录 |

## 环境

| 项 | 值 |
|---|---|
| GPU | 8× AMD Instinct MI300X VF（gfx942） |
| 镜像 | 按 LeRobot `docker/Dockerfile.rocm` 构建，基座 `rocm/primus:v26.4`，`LEROBOT_EXTRAS=all` |
| PyTorch | `2.12.0+rocm7.14.0a20260608`，HIP 7.14 |
| LeRobot | 0.7.1（`v0.7.0` 之后一个提交） |

## 快速上手

```bash
# 在 LeRobot 仓库根目录构建
docker build -f docker/Dockerfile.rocm -t lerobot-rocm .

# 单卡 2 步 smoke
docker run --rm --ipc=host --shm-size 16g \
  --device=/dev/kfd --device=/dev/dri \
  --group-add $(getent group video | cut -d: -f3) --group-add $(getent group render | cut -d: -f3) \
  -e HIP_VISIBLE_DEVICES=0 -e MIOPEN_FIND_MODE=2 \
  -v /data/lerobot/outputs:/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lerobot-rocm \
  lerobot-train --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --dataset.episodes='[0]' \
    --batch_size=1 --steps=2 --save_freq=2 --log_freq=1 --env_eval_freq=0 \
    --output_dir=/outputs/act-smoke

# 2 卡 DDP：HIP_VISIBLE_DEVICES=0,1，把 lerobot-train 换成
#   accelerate launch --num_processes=2 "$(which lerobot-train)"
# FSDP 再加 --parallelism.dp_shard=2
```

## 常见问题

| 现象 | 处理 |
|---|---|
| 多卡 DDP 加卡反而变慢、CPU 侧操作整体变慢 | HIP 运行时把进程限制在少数 CPU 上；启动前设 `AMD_CPU_AFFINITY=0` |
| 卷积策略（ACT、Diffusion）每次第一步要好几分钟 | MIOpen 在编译 kernel；挂持久目录并设 `MIOPEN_USER_DB_PATH`、`MIOPEN_CUSTOM_CACHE_DIR` |
| `--group-add render` 失败 | 用宿主机的数字 gid（见上面的命令） |
| 第一步要几十秒 | MIOpen 在查找卷积内核；设 `MIOPEN_FIND_MODE=2` |
| PyPI 的 CUDA torch 把 ROCm torch 换掉 | 镜像用 `UV_OVERRIDE` 锁住 torch / torchvision / numpy；装完检查 `torch.version.hip` 和 torchvision 的 `+rocm` |
| PEFT / LoRA 报 torchao 版本 | 镜像里是 0.15；当次容器里 `uv pip install 'torchao>0.16'` |
| MuJoCo 评估 EGL / GLFW 失败 | MI300X 没有图形上下文；装 `libosmesa6 libglu1-mesa`，`MUJOCO_GL=osmesa` |
| 需要 FlashAttention 的策略 | 换成策略自带的 PyTorch / SDPA 实现，例如 FLUX3 `--policy.attn_mode=torch` |
| FLUX3 的 Video VAE 需要 NATTEN | NATTEN 没有 ROCm 轮子，源码构建只有 flex 后端，会 OOM；用 PR #5213 的纯 PyTorch 路径 |
| QUANTILES 归一化缺 q01/q99 | 在本地算好写进数据集 `meta/stats.json`；官方脚本会推 Hub |
| 门控模型 403 | 在 Hugging Face 页面接受授权；token 放在挂载的缓存目录，不写进脚本 |

## 目录

```
reports/          smoke-report.md（完整报告）、计划文件、results.txt（每次运行的结果，含失败）
scripts/runners/  批量队列脚本（生成单任务脚本并串行执行）
scripts/jobs/     单任务脚本；remote/ 是 GPU 节点上实际执行过的版本
scripts/patches/  一次性运行时补丁（LaWAM FSDP、MolmoAct2 LoRA）
scripts/repro/    NATTEN / flex attention 复现脚本
scripts/dist/     分布式队列：run-dist-queue.sh、queue-*.txt、resume/拓扑任务生成器、fsdp_autowrap_train.py
upstream/         提交给上游的 PR 描述、留言、issue 原文
logs/             各批次的训练日志
webui/            本地 WebUI：通过 ssh 在 GPU 节点上启动一次固定的 DiT 2 步训练
```

脚本里的输出目录 `/data/lerobot/outputs` 和主机名是占位，按你的环境替换。`DOUni` 是 8 卡节点的 ssh 别名。

## WebUI

```bash
LEROBOT_ROCM_HOST=<ssh-alias> LEROBOT_OUTPUTS=<outputs-dir-on-node> python3 webui/server.py
# 打开 http://127.0.0.1:8765/
```

## 上游跟进

| 对象 | 状态 |
|---|---|
| [huggingface/lerobot#5432](https://github.com/huggingface/lerobot/pull/5432) TOPReward 单帧视频补帧 | 等待评审 |
| [huggingface/lerobot#5433](https://github.com/huggingface/lerobot/pull/5433) RA-BC 不支持的策略提前报错 | 等待评审，已有社区复现 |
| [huggingface/lerobot#5213](https://github.com/huggingface/lerobot/pull/5213) FLUX3 不依赖 NATTEN（他人 PR） | 已留言 ROCm 结果 |
| [SHI-Labs/NATTEN#354](https://github.com/SHI-Labs/NATTEN/issues/354) ROCm 上编译版 flex 不可用 | 已提交 |
| [huggingface/lerobot#5471](https://github.com/huggingface/lerobot/pull/5471) RA-BC 在 DDP 下解包 `DistributedDataParallel` | 等待评审 |
| [huggingface/lerobot#5472](https://github.com/huggingface/lerobot/pull/5472) `Dockerfile.rocm` 设 `AMD_CPU_AFFINITY=0` | 等待评审 |
| [huggingface/lerobot#5473](https://github.com/huggingface/lerobot/pull/5473) `torch_accelerators.mdx` ROCm 已知问题 | 等待评审 |
| [huggingface/lerobot#5474](https://github.com/huggingface/lerobot/issues/5474) FSDP resume：可训练但无梯度的参数（CPU gloo 复现） | 已提交 |
| [huggingface/lerobot#5475](https://github.com/huggingface/lerobot/issues/5475) FSDP2 下 `min_num_params` 失败 | 已提交 |
| [huggingface/lerobot#5476](https://github.com/huggingface/lerobot/issues/5476) Praxis DROID `rename_map` | 已提交 |
| [ROCm/rocm-systems#6712](https://github.com/ROCm/rocm-systems/pull/6712#issuecomment-7088991627) `AMD_CPU_AFFINITY` 对 PyTorch DDP 的影响 | 已留言 |
