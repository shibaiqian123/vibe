"""PPO runner config — the perloco VISION agent.

One row: AdaptSonic + the cross-attention extractor, plain PPO. The actor and
the runner spine are `vibe.core.rl` (shared with repose — a frozen SONIC base is
not a task's property); the privileged and from-scratch rows are orcs's.

  adapt_sonic_ext_agent_cfg()  frozen SONIC + LoRA + extractor, no predictor

No aux objectives here. `-Sfd`/`-Lfd` need a prediction TARGET, and a target set
is task-specific by construction — when perloco earns one, it declares it here
the way repose does.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from vibe.core.rl import adapt_sonic_agent_cfg, attach_extractor
from vibe.tasks.perloco.config.g1 import observation_cfgs


def adapt_sonic_ext_agent_cfg(
    experiment_name: str,
    *,
    rank: int = 16,
    alpha: float = 1.0,
    query_groups: tuple[str, ...] = observation_cfgs.DEFAULT_QUERY_GROUPS,
    extractor: str = "cross_attention",
) -> RslRlOnPolicyRunnerCfg:
    """Frozen SONIC base + LoRA adapter, conditioned on z instead of a scan.

    Extractor-only (`-Ext`): the extractor is trained by PPO gradients alone, so
    this row answers "does a task gradient reach a vision encoder at all" before
    any auxiliary objective is spent on the question. The env's token + query
    groups feed the extractor by NAME (`observation_cfgs`).
    """
    cfg = adapt_sonic_agent_cfg(experiment_name, rank=rank, alpha=alpha)
    attach_extractor(cfg, query_groups, extractor=extractor)
    return cfg


def stairs_smpl_agent_cfg():
    from orcs.core.sonic.runtime import agent_cfg
    cfg = adapt_sonic_agent_cfg("vibe_perloco_grail_stairs_smpl")
    cfg.actor.update(agent_cfg("release", "smpl", cfg.experiment_name).actor)
    attach_extractor(cfg, observation_cfgs.DEFAULT_QUERY_GROUPS)
    cfg.actor["class_name"] = "vibe.core.versioned_sonic.VersionedExtractorSonicAdapterModel"
    cfg.actor["allow_legacy_resume"] = False
    cfg.actor["visual_contract_id"] = stairs_visual_contract()
    cfg.amp_dtype = None
    cfg.save_interval = 100
    cfg.logger = "tensorboard"
    return cfg


def stairs_visual_contract():
    """Content identity, independent of the machine's dataset directory."""
    import hashlib
    import json
    import os
    from dataclasses import asdict
    from pathlib import Path
    from orcs.tasks.perloco.env_cfg import staged_root
    from vibe.core.sensors import head_cam_cfg
    from vibe.core.observation_cfgs import IMG_ENCODER
    root = staged_root("grail_stairs")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(root)
    identity = {"dataset": [(str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
                            for p in files], "camera": asdict(head_cam_cfg()),
                "backbone": IMG_ENCODER, "precision": "float32"}
    hand = Path(os.environ["ORCS_G1_HAND_URDF"])
    import xml.etree.ElementTree as ET
    meshes = sorted({e.attrib["filename"] for e in ET.parse(hand).iter("mesh")})
    identity["hand"] = hashlib.sha256(hand.read_bytes()).hexdigest()
    identity["meshes"] = [(name, hashlib.sha256((hand.parent / name).read_bytes()).hexdigest())
                          for name in meshes]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
