#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig
from lerobot.utils.constants import ACTION, OBS_STATE

state_dim, action_dim = 10, 6
cfg = GaussianActorConfig(
    device="cuda",
    input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))},
    output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
    dataset_stats={
        OBS_STATE: {"min": [0.0] * state_dim, "max": [1.0] * state_dim},
        ACTION: {"min": [0.0] * action_dim, "max": [1.0] * action_dim},
    },
)
cfg.validate_features()
policy = GaussianActorPolicy(cfg)
algo_cfg = SACAlgorithmConfig.from_policy_config(cfg)
algo_cfg.utd_ratio = 1
algo_cfg.policy_update_freq = 1
algorithm = SACAlgorithm(policy=policy, config=algo_cfg)
algorithm.make_optimizers_and_scheduler()
assert torch.cuda.is_available() and torch.version.hip

def batch():
    b, s, a = 4, state_dim, action_dim
    obs = {OBS_STATE: torch.randn(b, s, device="cuda")}
    nxt = {OBS_STATE: torch.randn(b, s, device="cuda")}
    return {
        ACTION: torch.randn(b, a, device="cuda"),
        "reward": torch.randn(b, device="cuda"),
        "state": obs,
        "next_state": nxt,
        "done": torch.zeros(b, device="cuda"),
        "complementary_info": {},
    }

losses = []
for step in range(2):
    stats = algorithm.update(iter([batch(), batch()]))
    losses.append(stats.losses)
    print(f"step {step+1} {stats.losses}")
print("HIL_PASS", losses)
PY
