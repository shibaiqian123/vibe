"""Synthetic vision features test routing and gradients, not camera/Theia accuracy."""
import pytest
import torch
from orcs.core.sonic.port import fsq, make_mlp
from tensordict import TensorDict

from vibe.core.versioned_sonic import VersionedExtractorSonicAdapterModel


def test_visual_gradients_and_resume(tmp_path):
    torch.manual_seed(21)
    e, d = make_mlp((840, 16, 64)), make_mlp((994, 16, 29))
    path = tmp_path / 'base.pt'
    torch.save({'model_state_dict': {**{'encoder.'+k:v for k,v in e.state_dict().items()},
                                    **{'decoder.'+k:v for k,v in d.state_dict().items()}},
                'meta': {'action_std': torch.full((29,), .1)}}, path)
    obs = TensorDict({'policy': torch.randn(4,930), 'tokenizer':torch.randn(4,840),
                      'augmentation':torch.randn(4,6), 'query':torch.randn(4,8),
                      'kv_tokens':TensorDict({'img_tokens':torch.randn(4,21,192)},batch_size=[4])},batch_size=[4])
    kw = dict(sonic_contract_id='ab'*32, tokenizer_width=840,
              encoder_hidden_dims=[16], decoder_hidden_dims=[16], base_checkpoint=str(path),
              freeze_base=True, rank=2, alpha=2., adapter_obs_group=['augmentation','kv_tokens'],
              extractor_cfg={'kv_tokens':dict(class_name='rsl_rl.modules.CrossAttentionExtractor',
                  token_terms=['img_tokens'],query_groups=['query'],latent_dim=16,attn_dim=8,
                  num_heads=1,num_learned_queries=0,layer_norm=True)})
    model = VersionedExtractorSonicAdapterModel(obs,{'actor':['policy']},'actor',29,**kw)
    expected = d(torch.cat([fsq(e(obs['tokenizer'])),obs['policy']],-1))
    torch.testing.assert_close(model(obs),expected,rtol=0,atol=0)
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):
        opt.zero_grad()
        model(obs).square().mean().backward()
        opt.step()
    changed = []
    for n,p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(before[n],p),n
        elif not torch.equal(before[n],p):
            changed.append(n)
    assert any('extractor' in n for n in changed),changed
    assert any('adapters' in n for n in changed),changed
    alternate = obs.clone()
    alternate['kv_tokens','img_tokens'] = torch.randn(4,21,192)*3
    assert not torch.allclose(model(obs),model(alternate),atol=1e-6,rtol=1e-6)
    state = {k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    state['_orcs_sonic_contract'].zero_()
    with pytest.raises(ValueError,match='resume'):
        model.load_state_dict(state)
