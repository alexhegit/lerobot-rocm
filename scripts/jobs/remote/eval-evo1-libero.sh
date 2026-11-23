#!/bin/bash
# EVO1 LIBERO checkpoint on ROCm with the documented settings (docs/source/evo1.mdx, revision
# 515921f4, reference LIBERO Spatial 97.0% over 500 episodes): libero_spatial, 10 tasks x 5 episodes,
# one process per task, OSMesa rendering instead of EGL.
set -uo pipefail
apt-get update -qq && apt-get install -y -qq libosmesa6 libglu1-mesa >/dev/null
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
OUT=/outputs/rocm-train/eval-evo1-libero
rm -rf "$OUT"
suite=libero_spatial
horizon=220
for task_id in {0..9}; do
  yes N | lerobot-eval \
    --policy.path=zuoxingdong/evo1_libero \
    --policy.pretrained_revision=515921f4a2c1d3f3ad523721eafa26fdf2af315b \
    --policy.vlm_model_name=OpenGVLab/InternVL3-1B-hf \
    --policy.device=cuda \
    --policy.use_amp=true \
    --policy.dtype=bfloat16 \
    --policy.use_flash_attn=false \
    --policy.enable_gradient_checkpointing=false \
    --policy.vlm_num_layers=14 \
    --policy.image_resolution='[448,448]' \
    --policy.max_text_length=1024 \
    --policy.chunk_size=50 \
    --policy.n_action_steps=14 \
    --policy.max_state_dim=24 \
    --policy.max_action_dim=24 \
    --policy.num_inference_timesteps=32 \
    --policy.postprocess_action_dim=7 \
    --policy.binarize_gripper=true \
    --policy.gripper_threshold=0.0 \
    --policy.gripper_below_threshold_value=-1.0 \
    --policy.gripper_above_threshold_value=1.0 \
    --env.type=libero \
    --env.task="${suite}" \
    --env.task_ids="[${task_id}]" \
    --env.camera_name=agentview_image,robot0_eye_in_hand_image \
    --env.camera_name_mapping="{agentview_image: image, robot0_eye_in_hand_image: image2}" \
    --env.control_mode=relative \
    --env.obs_type=pixels_agent_pos \
    --env.observation_width=448 \
    --env.observation_height=448 \
    --env.init_states=true \
    --env.episode_length="${horizon}" \
    --env.render_mode=rgb_array \
    --env.max_parallel_tasks=1 \
    --eval.n_episodes=5 \
    --eval.batch_size=1 \
    --eval.use_async_envs=false \
    --eval.recording=false \
    --seed=1000 \
    --output_dir="$OUT/task-${task_id}" >"$OUT-task-${task_id}.log" 2>&1
  echo "task ${task_id} done"
done
python3 - "$OUT" <<'PY'
import json, pathlib, sys
n = s = 0
for p in sorted(pathlib.Path(sys.argv[1]).glob("task-*/eval_info.json")):
    o = json.load(open(p))["overall"]
    n += o["n_episodes"]; s += o["n_success"]
    print(p.parent.name, o["n_success"], "/", o["n_episodes"])
print(f"EVAL libero_spatial {s}/{n} = {100 * s / max(n, 1):.1f}%")
PY
