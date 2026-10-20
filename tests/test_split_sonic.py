"""Exercise the real upstream extractor, FSQ and merged adapter export on CPU."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from vibe.export.agent.split_sonic import (
    SplitOnnxAgent,
    export_split,
    split_graph,
    validate_observation,
)


@pytest.fixture
def sample(tmp_path):
    from rsl_rl.models.extractor_model import ExtractorSonicAdapterModel
    torch.manual_seed(17)
    obs = TensorDict({'tokenizer': torch.randn(2, 840), 'policy': torch.randn(2, 930),
                      'augmentation': torch.randn(2, 6), 'q_task': torch.randn(2, 6),
                      'kv': {'img': torch.randn(2, 21, 192)}}, batch_size=[2])
    model = ExtractorSonicAdapterModel(
        obs, {'actor': ['policy']}, 'actor', 29,
        encoder_hidden_dims=(32,), decoder_hidden_dims=(32, 32), rank=4,
        adapter_obs_group=['augmentation', 'kv'],
        extractor_cfg={'kv': {'class_name': 'rsl_rl.modules.CrossAttentionExtractor',
                              'token_terms': ['img'], 'query_groups': ['q_task'],
                              'latent_dim': 128, 'attn_dim': 16}}).eval()
    # Nonzero adapters are essential: zero-init could conceal a missing visual path.
    with torch.no_grad():
        for adapter in model.decoder.adapters:
            adapter.up.weight.normal_(0, .05)
    graph = model.as_onnx(with_attn=True).cpu().eval()
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save(model.state_dict(), checkpoint)
    inputs = []
    for p in graph.layout:
        inputs.append({**p, 'terms': [{'name': p['term'] or p['name'], 'offset': 0,
                                      'dim': int(np.prod(p['shape'])), 'shape': p['shape']}]})
    manifest = {'checkpoint': str(checkpoint), 'inputs': inputs,
                'outputs': graph.output_names,
                'action': {'joint_names': [str(i) for i in range(29)],
                           'scale': [1.] * 29, 'default_joint_pos': [0.] * 29}}
    return model, graph, obs, manifest


def test_real_upstream_split_and_onnx(sample, tmp_path):
    model, graph, obs, manifest = sample
    original = {k: v.clone() for k, v in model.state_dict().items()}
    encoder, decoder = export_split(graph, manifest, tmp_path)
    agent = SplitOnnxAgent(tmp_path)
    assert 'latent' in decoder.input_names
    assert 'tokenizer' not in decoder.input_names
    assert validate_observation(model, graph, encoder, decoder, agent, obs, 1e-5) < 1e-5
    altered = obs.clone()
    altered['kv', 'img'] = torch.zeros_like(obs['kv', 'img'])
    validate_observation(model, graph, encoder, decoder, agent, altered, 1e-5)
    assert not np.allclose(agent.act(obs, 0), agent.act(altered, 0), atol=1e-6)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in original.items())
    assert np.array_equal(agent.encode(obs, 0), agent.encode(altered, 0))


def test_alias_and_attention_preserved(sample):
    _, graph, obs, _ = sample
    graph.alias_ports({'q_task': 'augmentation'})
    encoder, decoder, _ = split_graph(graph)
    inputs = (obs['tokenizer'][:1], obs['policy'][:1], obs['augmentation'][:1], obs['kv', 'img'][:1])
    with torch.no_grad():
        expected = graph(*inputs)
        actual = decoder(encoder(inputs[0]), *inputs[1:])
    for a, b in zip(expected, actual, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_bundle_rejects_changed_model(sample, tmp_path):
    _, graph, _, manifest = sample
    export_split(graph, manifest, tmp_path)
    with (tmp_path / 'visual_decoder.onnx').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        SplitOnnxAgent(tmp_path)


def test_wrong_input_and_missing_condition(sample, tmp_path):
    _, graph, obs, manifest = sample
    export_split(graph, manifest, tmp_path)
    agent = SplitOnnxAgent(tmp_path)
    with pytest.raises(KeyError):
        agent.act(obs.exclude('q_task'), 0)
    feed = agent.decoder.assemble(obs, 0)
    feed['latent'] = np.zeros((1, 63), dtype=np.float32)
    with pytest.raises(Exception, match='dimension|dimensions'):
        agent.decoder.session.run(None, feed)


def test_unsupported_graph():
    with pytest.raises(TypeError):
        split_graph(torch.nn.Linear(64, 29))


def test_focus_family_keeps_grid_and_command_consistent():
    from types import SimpleNamespace as NS

    from vibe.export.agent.case import pin_terrain_family
    terrain = NS(sub_terrains={'a': object(), 'b': object()}, num_cols=2)
    cmd = NS(tile_keys=('a/level_0.00', 'b/level_0.00'), n_rows=1,
             clips={'a/level_0.00': ('a',), 'b/level_0.00': ('b',)})
    env = NS(scene=NS(terrain=NS(terrain_generator=terrain)), commands={'motion': cmd})
    pin_terrain_family(env, 'b')
    assert list(terrain.sub_terrains) == ['b']
    assert cmd.tile_keys == ('b/level_0.00',)
    assert list(cmd.clips) == ['b/level_0.00']
    with pytest.raises(ValueError, match='Unknown'):
        pin_terrain_family(env, 'absent')


def test_theia_identity_matches_training_loader():
    from vibe.export.agent.run_split import backbone_id
    assert backbone_id("theia-tiny-patch16-224-cddsv") == "theaiinstitute/theia-tiny-patch16-224-cddsv"
    assert backbone_id("other/model") == "other/model"
    assert backbone_id("theia-small") != backbone_id("theia-tiny")
