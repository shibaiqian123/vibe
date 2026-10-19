"""Play-only RGB intervention. No environment variable or training-side activation."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import random
from typing import Literal

import torch

RgbMode = Literal["normal", "gray", "shuffle"]
_REQUEST = ContextVar("vibe_rgb_play_request", default=None)


@contextmanager
def rgb_play_request(mode: RgbMode, seed: int, checkpoint: str):
    token = _REQUEST.set((mode, seed, checkpoint))
    try:
        yield
    finally:
        _REQUEST.reset(token)


def attach_play_control(env, sensor_name):
    request = _REQUEST.get()
    if request is None:
        return
    if not hasattr(env, "_rgb_compare"):
        env._rgb_compare = RgbCompare(*request)
    env._rgb_compare.sensors.add(sensor_name)


def cross_clip_permutation(clips: tuple[int, ...], seed: int) -> tuple[int, ...]:
    """Rotate grouped indices by largest group: a bijection across clip groups."""
    groups = {}
    for i, clip in enumerate(clips):
        groups.setdefault(clip, []).append(i)
    largest = max(map(len, groups.values()), default=0)
    if not clips or 2 * largest > len(clips):
        raise ValueError("shuffle 不可用：需要不同片段，且任一片段的环境数不能超过总数一半")
    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    indices = []
    for key in keys:
        rng.shuffle(groups[key])
        indices.extend(groups[key])
    result = [0] * len(clips)
    for i, dst in enumerate(indices):
        result[dst] = indices[(i + largest) % len(indices)]
    assert all(i != src and clips[i] != clips[src] for i, src in enumerate(result))
    return tuple(result)


@dataclass
class InputFrame:
    raw: torch.Tensor
    used: torch.Tensor
    step: int
    version: int
    clips: tuple[int, ...]
    frames: tuple[int, ...]
    sources: tuple[int, ...]
    sensor: str


class RgbCompare:
    def __init__(self, requested_mode="normal", seed=0, checkpoint=""):
        self.requested_mode = requested_mode
        self.seed = seed
        self.checkpoint = checkpoint
        # Construct/reset observations before enabling shuffle, when clips are ready.
        self.mode = "normal"
        self.version = 0
        self.sensors: set[str] = set()
        self.inputs: dict[str, InputFrame] = {}
        self.policy_frame = None
        self.policy_attention = None
        self.mapping = None
        self.mapping_clips = None

    @staticmethod
    def motion_ids(env):
        try:
            cmd = env.command_manager.get_term("motion")
            clips = tuple(cmd._clip_ids.detach().cpu().tolist())
            offsets = getattr(getattr(cmd, "motion", None), "clip_offsets", None)
            times = cmd.time_steps if offsets is None else cmd.time_steps - offsets[cmd._clip_ids]
            frames = tuple(times.detach().cpu().tolist())
            return clips, frames
        except (AttributeError, KeyError, ValueError):
            return (), ()

    def configure(self, mode, env):
        if mode not in ("normal", "gray", "shuffle"):
            raise ValueError(f"Unknown RGB mode: {mode}")
        if mode == "shuffle":
            clips, _ = self.motion_ids(env)
            mapping = cross_clip_permutation(clips, self.seed)
        else:
            clips, mapping = None, None
        self.mode = mode
        self.mapping_clips, self.mapping = clips, mapping
        self.invalidate()

    def invalidate(self):
        self.version += 1
        self.inputs.clear()
        self.policy_frame = None
        self.policy_attention = None

    def cache_token(self, env):
        if self.mode == "shuffle":
            clips, _ = self.motion_ids(env)
            if clips != self.mapping_clips:
                self.mapping = cross_clip_permutation(clips, self.seed)
                self.mapping_clips = clips
                self.invalidate()
        return self.version

    def apply(self, rgb, env, sensor):
        if rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError("RGB comparison expects uint8 (B,H,W,3)")
        sources = tuple(range(rgb.shape[0]))
        if self.mode == "gray":
            used = torch.full_like(rgb, 127)
        elif self.mode == "shuffle":
            self.cache_token(env)
            sources = self.mapping
            if len(sources) != rgb.shape[0]:
                raise ValueError("clip count does not match RGB batch")
            used = rgb[list(sources)]
        else:
            used = rgb
        clips, frames = self.motion_ids(env)
        self.inputs[sensor] = InputFrame(
            rgb.detach().clone(), used.detach().clone(), int(env.common_step_counter),
            self.version, clips, frames, sources, sensor,
        )
        return used

    def capture_policy(self, frame, attention):
        if frame is not None and frame.version == self.version and attention is not None:
            self.policy_frame = frame
            self.policy_attention = attention.detach().clone()

    def matched(self):
        f = self.policy_frame
        return f is not None and f.version == self.version and self.policy_attention is not None
