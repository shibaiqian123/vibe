"""CPU contracts: no model download, simulator, or checkpoint required."""
from types import SimpleNamespace

import pytest
import torch

from vibe.core.rgb_compare import RgbCompare, cross_clip_permutation, attach_play_control, rgb_play_request
from vibe.core.mdp.observations import image_feature


def env_fixture(clips=(0, 1, 2, 3, 4, 5)):
    cmd = SimpleNamespace(_clip_ids=torch.tensor(clips), time_steps=torch.arange(len(clips)))
    rgb = torch.arange(len(clips)*2*3*3, dtype=torch.uint8).reshape(len(clips), 2, 3, 3)
    env = SimpleNamespace(common_step_counter=1, device="cpu",
                          command_manager=SimpleNamespace(get_term=lambda _: cmd),
                          scene={"head": SimpleNamespace(data=SimpleNamespace(rgb=rgb))})
    return env, rgb


def fake_term(env):
    t = object.__new__(image_feature)
    t._env = env
    t._sensor_name = "head"
    t._ckey = (id(env), "head", "fake", torch.float32)
    t._dev, t._dtype = "cpu", torch.float32
    t._mean, t._std = torch.zeros(1,3,1,1), torch.ones(1,3,1,1)
    t.calls = 0
    def encode(x):
        t.calls += 1
        return x.clone(), x.mean((2,3))
    t._encode = encode
    return t


def test_activation_is_play_only_and_context_is_restored():
    env, _ = env_fixture()
    attach_play_control(env, "head")
    assert not hasattr(env, "_rgb_compare")
    with rgb_play_request("gray", 7, "a.pt"):
        attach_play_control(env, "head")
        attach_play_control(env, "head")
        assert env._rgb_compare.requested_mode == "gray"
        assert env._rgb_compare.sensors == {"head"}
    fresh, _ = env_fixture()
    attach_play_control(fresh, "head")
    assert not hasattr(fresh, "_rgb_compare")


@pytest.mark.parametrize("clips", [(0,1), (0,0,1,1), (0,0,1,2,2,3), tuple(range(6))])
def test_shuffle_bijection_cross_clip_reproducible(clips):
    p = cross_clip_permutation(clips, 123)
    assert p == cross_clip_permutation(clips, 123)
    assert sorted(p) == list(range(len(clips)))
    assert all(src != dst and clips[src] != clips[dst] for dst,src in enumerate(p))


@pytest.mark.parametrize("clips", [(), (0,), (0,0), (0,0,0,1)])
def test_shuffle_refuses_impossible_mapping(clips):
    with pytest.raises(ValueError, match="shuffle"):
        cross_clip_permutation(clips, 0)


def test_normal_gray_shuffle_preserve_sensor_and_snapshot():
    env, rgb = env_fixture()
    original = rgb.clone()
    c = RgbCompare()
    assert c.apply(rgb, env, "head") is rgb
    c.configure("gray", env)
    assert torch.all(c.apply(rgb, env, "head") == 127)
    c.configure("shuffle", env)
    used = c.apply(rgb, env, "head")
    assert torch.equal(used, rgb[list(c.mapping)])
    assert torch.equal(rgb, original)
    snapshot = c.inputs["head"].used.clone()
    rgb.zero_()
    assert torch.equal(c.inputs["head"].used, snapshot)


def test_same_step_mode_cache_and_shared_cls():
    env, rgb = env_fixture()
    t = fake_term(env)
    baseline = t._forward(env)
    c = env._rgb_compare = RgbCompare()
    normal = t._forward(env)
    assert torch.equal(baseline[0], normal[0])
    c.configure("gray", env)
    gray = t._forward(env)
    assert t.calls == 3
    sibling = fake_term(env)
    sibling_out = sibling._forward(env)
    assert sibling.calls == 0
    assert sibling_out is gray
    assert torch.all(gray[0] == 127/255)
    c.configure("normal", env)
    assert torch.equal(t._forward(env)[0], baseline[0])
    t.reset()
    t._forward(env)
    assert t.calls == 5


def test_policy_frame_stays_with_attention_not_next_render():
    env, rgb = env_fixture()
    c = RgbCompare()
    c.apply(rgb, env, "head")
    consumed = c.inputs["head"]
    attn = torch.rand(6,3,21)
    c.capture_policy(consumed, attn)
    rgb.zero_()
    env.common_step_counter += 1
    c.apply(rgb, env, "head")
    assert c.policy_frame is consumed
    assert c.policy_frame.step == 1
    assert c.matched()
    attn.zero_()
    assert c.policy_attention.count_nonzero() > 0
    c.configure("gray", env)
    assert not c.matched()
    c.capture_policy(consumed, attn)
    assert not c.matched()


def test_failed_mode_change_keeps_previous_mode():
    env, _ = env_fixture((0,0))
    c = RgbCompare()
    with pytest.raises(ValueError):
        c.configure("shuffle", env)
    assert c.mode == "normal"


def test_cli_has_rgb_flags_and_rejects_nonvisual_task(monkeypatch):
    import mjlab.scripts.play as play
    cfg = play.PlayConfig(rgb_mode="gray", rgb_seed=17, viewer="viser")
    assert cfg.rgb_mode == "gray" and cfg.rgb_seed == 17
    monkeypatch.setattr(play, "load_env_cfg", lambda *a, **k: SimpleNamespace(observations={}))
    with pytest.raises(ValueError, match="image_feature"):
        play.run_play("nonvisual", cfg)


def test_display_frame_is_clip_local():
    env, _ = env_fixture((0,1))
    cmd = env.command_manager.get_term("motion")
    cmd.motion = SimpleNamespace(clip_offsets=torch.tensor([0,499]))
    cmd.time_steps = torch.tensor([12,511])
    assert RgbCompare.motion_ids(env) == ((0,1),(12,12))
