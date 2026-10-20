"""Thin split of rsl_rl's official SONIC ONNX wrapper; no network/FSQ rewrite.

Private upstream fields are checked deliberately: incompatible dependency versions fail
rather than silently exporting a different policy. The upstream decoder merge retains
conditioning as additional first-layer columns, not as a constant visual correction.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class MotionEncoder(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.encoder = copy.deepcopy(graph.encoder)
        self.fsq = copy.deepcopy(graph.fsq)
        self.num_tokens, self.token_dim = graph.num_tokens, graph.token_dim

    def forward(self, tokenizer):
        z = self.encoder(tokenizer)
        return self.fsq(z.reshape(z.shape[0], self.num_tokens, self.token_dim)).reshape(z.shape[0], -1)


def split_graph(graph):
    """Reuse the complete official forward, bypassing only encoder + quantization."""
    from rsl_rl.models.sonic_base_model import _OnnxSonicBaseModel

    if not isinstance(graph, _OnnxSonicBaseModel):
        raise TypeError('split-sonic requires the upstream SONIC ONNX wrapper')
    if graph.num_tokens * graph.token_dim != 64:
        raise ValueError('Expected 64-dimensional SONIC latent')
    first = graph._ports[0]
    if graph._slot_to_port[0] != 0 or graph._slot_to_port.count(0) != 1:
        raise ValueError('Tokenizer must not be aliased with decoder conditions')
    if any(p.name == 'latent' for p in graph._ports):
        raise ValueError('latent is reserved for the split boundary')
    encoder = MotionEncoder(graph).cpu().eval()
    decoder = copy.deepcopy(graph).cpu().eval()
    # Identity preserves the upstream forward, including the deterministic action head,
    # extractor normalizers, query ports and optional attention outputs.
    decoder.encoder = nn.Identity()
    decoder.fsq = nn.Identity()
    from rsl_rl.models._onnx_export import Port
    port = Port('latent', (64,), ('latent',))
    decoder._ports[0] = port
    decoder._slots[0] = port
    return encoder, decoder, first.name


def export_split(graph, manifest, directory):
    """Write to a fresh directory owned by the caller; publish only after validation."""
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx

    encoder, decoder, tokenizer = split_graph(graph)
    directory = Path(directory)
    em = copy.deepcopy(manifest)
    em.update(schema='vibe.motion_encoder.v1', inputs=[manifest['inputs'][0]], outputs=['latent'])
    em.pop('action', None)
    boundary = {'name': 'latent', 'shape': [1, 64], 'dtype': 'float32',
                'semantics': 'post-FSQ; flattened token order; decoder does not quantize',
                'num_tokens': graph.num_tokens, 'token_dim': graph.token_dim,
                'fsq_half_width': graph.fsq.half_width.tolist()}
    em['latent_contract'] = boundary
    dm = copy.deepcopy(manifest)
    dm['schema'] = 'vibe.visual_decoder.v1'
    dm['inputs'][0] = {'name': 'latent', 'shape': [64], 'groups': ['latent'],
                       'terms': [{'name': 'latent', 'dim': 64, 'offset': 0, 'shape': [64]}]}
    dm['latent_contract'] = boundary
    dm['requires'] = 'All listed ports, including task queries; latent alone is insufficient.'
    for model, name, meta, dummy, names in (
        (encoder, 'motion_encoder', em, (graph.get_dummy_inputs()[0],), [tokenizer]),
        (decoder, 'visual_decoder', dm, decoder.get_dummy_inputs(), decoder.input_names),
    ):
        path = directory / f'{name}.onnx'
        torch.onnx.export(model, dummy, str(path), input_names=names,
                          output_names=meta['outputs'], opset_version=18,
                          dynamic_axes={}, dynamo=False)
        attach_metadata_to_onnx(str(path), {'manifest': json.dumps(meta)})
        (directory / f'{name}.manifest.json').write_text(json.dumps(meta, indent=2))
    bundle = {'schema': 'vibe.split_sonic.v1', 'status': 'unvalidated',
              'checkpoint_sha256': sha256(manifest['checkpoint']),
              'files': {name: sha256(directory / name) for name in (
                  'motion_encoder.onnx', 'visual_decoder.onnx',
                  'motion_encoder.manifest.json', 'visual_decoder.manifest.json')},
              'tokenizer_port': tokenizer, 'latent_contract': boundary,
              'vision': 'External Theia encoder; use export-encoder and matching camera/preprocessing.',
              'source_manifest': manifest}
    (directory / 'bundle.json').write_text(json.dumps(bundle, indent=2))
    return encoder, decoder


class SplitOnnxAgent:
    """Reuse OnnxAgent's term assembly and PD metadata; two ORT sessions in series."""
    def __init__(self, directory, provider='cpu'):
        import onnxruntime as ort

        from vibe.export.agent.onnx_agent import OnnxAgent

        self.directory = Path(directory)
        self.bundle = json.loads((self.directory / 'bundle.json').read_text())
        for name, digest in self.bundle['files'].items():
            if sha256(self.directory / name) != digest:
                raise ValueError(f'Split bundle hash mismatch: {name}')
        self.decoder = OnnxAgent(self.directory / 'visual_decoder.onnx', provider=provider)
        self.providers = self.decoder.providers
        self.encoder = ort.InferenceSession(str(self.directory / 'motion_encoder.onnx'), providers=self.providers)
        em = json.loads(self.encoder.get_modelmeta().custom_metadata_map['manifest'])
        if em['latent_contract'] != self.decoder.manifest['latent_contract']:
            raise ValueError('Encoder/decoder latent contracts differ')
        self.encoder_port = em['inputs'][0]
        self.decoder.ports = self.decoder.ports[1:]  # latent is supplied by encoder, not observations
        self.manifest = self.decoder.manifest
        self.joint_names = self.decoder.joint_names

    def encode(self, obs, world):
        import numpy as np
        p = self.encoder_port
        x = obs[p['groups'][0]][world:world + 1].detach().cpu().numpy().astype(np.float32)
        return self.encoder.run(['latent'], {p['name']: x})[0]

    def act(self, obs, world):
        inputs = self.decoder.assemble(obs, world)
        inputs['latent'] = self.encode(obs, world)
        return self.decoder.session.run(None, inputs)[self.decoder.action_index]

    def joint_targets(self, actions):
        return self.decoder.joint_targets(actions)


def validate_observation(model, graph, encoder, decoder, agent, obs, tolerance):
    """Compare raw actor, upstream merged graph, split torch and split ORT on real obs."""
    import numpy as np
    # Use upstream term assembly, including aliased query groups.
    inputs = agent.decoder.assemble(obs, 0)
    p = agent.encoder_port
    inputs[p['name']] = obs[p['groups'][0]][0:1].detach().cpu().numpy()
    tensors = tuple(torch.from_numpy(inputs[n]) for n in graph.input_names)
    with torch.no_grad():
        latent = encoder(tensors[0])
        expected_latent = model.encode_tokens(obs[0:1])
        if not torch.equal(latent, expected_latent):
            raise RuntimeError('FSQ mismatch: original actor versus export encoder')
        ort_latent = agent.encode(obs, 0)
        if not np.array_equal(latent.numpy(), ort_latent):
            raise RuntimeError('FSQ mismatch: torch versus ONNX encoder')
        joined = graph(*tensors)
        split = decoder(latent, *tensors[1:])
        joined = joined if isinstance(joined, tuple) else (joined,)
        split = split if isinstance(split, tuple) else (split,)
        if not all(torch.isfinite(x).all() for x in (*joined, *split)):
            raise RuntimeError("Non-finite exported output")
        error = max(float((a - b).abs().max()) for a, b in zip(joined, split, strict=True))
        error = max(error, float(np.abs(model(obs)[0:1].numpy() - agent.act(obs, 0)).max()))
        if not np.isfinite(error) or error > tolerance:
            raise RuntimeError(f'Split action parity failed: {error} > {tolerance}')
    return error


def run_split_export(model, graph, manifest, out_dir, env, cfg, steps, min_steps):
    """Reuse ViBe's DualPolicy/WorldStats and publish a checked bundle atomically."""
    import tempfile

    from vibe.export.agent.export_onnx import _launch_viewer, _two_world_rollout
    from vibe.export.agent.onnx_agent import DualPolicy

    destination = Path(out_dir) / 'split_sonic'
    if destination.exists():
        raise FileExistsError(f'Refusing to replace {destination}; choose another --output-dir')
    staging = Path(tempfile.mkdtemp(prefix='.split-pending-', dir=out_dir))
    report = {'status': 'unvalidated', 'tolerance': cfg.tolerance,
              'limits': ['two-world mjlab check, not independent MuJoCo deployment',
                         'RGB/gray end-to-end comparison and six-clip coverage require separate runs']}
    try:
        reference = copy.deepcopy(model).cpu().float().eval()
        if getattr(reference, 'adapt_encoder', False):
            raise ValueError('Conditioned/adapted encoders are not supported by this split')
        encoder, decoder = export_split(graph, manifest, staging)
        agent = SplitOnnxAgent(staging, provider=cfg.provider)
        gate = SplitOnnxAgent(staging, provider='cpu')
        policy = DualPolicy(model, agent, gate_agent=gate, reference_model=reference)
        if cfg.check:
            obs_cpu = env.get_observations().to('cpu')
            report['initial_parity_max'] = validate_observation(
                reference, graph, encoder, decoder, gate, obs_cpu, cfg.tolerance)
            # Validate latent boundaries every step, not just the initial pose.
            class CheckedPolicy:
                def __call__(self, obs):
                    validate_observation(reference, graph, encoder, decoder, gate,
                                         obs.to('cpu'), cfg.tolerance)
                    return policy(obs)

                @property
                def open_loop_max(self):
                    return policy.open_loop_max

            stats = _two_world_rollout(env, CheckedPolicy(), steps, cfg.tolerance)
            survived = stats.first_reset[1] or steps
            report.update(open_loop_max=policy.open_loop_max, steps=steps,
                          reference=stats.line(0, steps), onnx=stats.line(1, steps))
            if survived < min_steps:
                raise RuntimeError(f'ONNX reset after {survived} steps (< {min_steps})')
            report['status'] = 'passed'
        bundle = json.loads((staging / 'bundle.json').read_text())
        bundle['status'] = report['status']
        (staging / 'bundle.json').write_text(json.dumps(bundle, indent=2))
        (staging / 'validation.json').write_text(json.dumps(report, indent=2))
        staging.rename(destination)
        print(f'[export] split bundle {destination}: {report["status"]}')
        if cfg.viewer != 'none':
            _launch_viewer(env, policy, cfg.viewer)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        (staging / 'validation.json').write_text(json.dumps(report, indent=2))
        bundle_path = staging / 'bundle.json'
        if bundle_path.exists():
            bundle = json.loads(bundle_path.read_text())
            bundle['status'] = 'failed'
            bundle_path.write_text(json.dumps(bundle, indent=2))
        print(f'[export] FAILED; diagnostics retained at {staging}')
        raise
    finally:
        env.close()
