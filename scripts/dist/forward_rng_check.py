"""Does SmolVLA's predict_action_chunk use the global RNG beyond the noise it is given?

Loads once, then predicts the same frames with the same explicit noise after torch.manual_seed(1) and
torch.manual_seed(2). Different outputs mean the forward draws random numbers itself.
"""

import torch
from huggingface_hub import snapshot_download

from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

path = snapshot_download("lerobot/smolvla_libero", revision="a24f1f380461b665158d52acd140152c79e8b36d")
rename = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}
policy = SmolVLAPolicy.from_pretrained(path).to("cuda").eval()
pre, _ = make_pre_post_processors(policy.config, pretrained_path=path,
                                  preprocessor_overrides={"device_processor": {"device": "cuda"}})
ds = LeRobotDataset("lerobot/libero", episodes=[0])
batches = [pre({rename.get(k, k): v for k, v in ds[i].items() if k.startswith("observation") or k == "task"})
           for i in (0, 50)]
noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim),
                    generator=torch.Generator().manual_seed(0)).cuda()


def predict(seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        return torch.cat([policy.predict_action_chunk(b, noise=noise).float().cpu() for b in batches])


a, a2, b = predict(1), predict(1), predict(2)
print(f"FORWARD_RNG same_seed_maxdiff={(a - a2).abs().max().item():.3e} "
      f"other_seed_maxdiff={(a - b).abs().max().item():.3e}", flush=True)
for name in ("_queues",):
    print("HAS", name, hasattr(policy, name))
