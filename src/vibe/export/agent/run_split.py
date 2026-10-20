"""Offline deployment example: saved named inputs + optional BGR image -> actions.

Usage: python -m vibe.export.agent.run_split --bundle DIR --inputs inputs.npz
    [--vision-encoder theia-tiny.onnx --image-bgr image.npy]
The NPZ uses ONNX port names, float32 arrays with batch dimension 1. Supply either
motion tokenizer or post-FSQ latent; all decoder query/conditioning ports remain required.
"""
import argparse
import json

import numpy as np
import onnxruntime as ort

from vibe.export.agent.split_sonic import SplitOnnxAgent


def backbone_id(name):
    """Match image_feature._load: bare Theia names use the official organization."""
    return f"theaiinstitute/{name}" if name.startswith("theia-") else name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--vision-encoder')
    parser.add_argument('--image-bgr')
    args = parser.parse_args()
    agent = SplitOnnxAgent(args.bundle)
    if agent.bundle['status'] != 'passed':
        raise ValueError('Bundle has not passed validation')
    with np.load(args.inputs, allow_pickle=False) as data:
        feeds = {k: data[k] for k in data.files}
    if bool(args.vision_encoder) != bool(args.image_bgr):
        raise ValueError('Provide both --vision-encoder and --image-bgr')
    if args.vision_encoder:
        session = ort.InferenceSession(args.vision_encoder, providers=['CPUExecutionProvider'])
        meta = json.loads(session.get_modelmeta().custom_metadata_map['manifest'])
        if meta['schema'] != 'vision.onnx.v1' or meta['input']['layout'] != 'bgr_hwc_uint8':
            raise ValueError('Expected official ViBe BGR vision encoder contract')
        image = np.load(args.image_bgr, allow_pickle=False)
        expected = (1, meta['input']['height'], meta['input']['width'], 3)
        if image.dtype != np.uint8 or image.shape != expected:
            raise ValueError(f'Expected uint8 {expected}, got {image.dtype} {image.shape}')
        results = session.run(None, {meta['input']['name']: image})
        outputs = dict(zip((o.name for o in session.get_outputs()), results, strict=True))
        for port in agent.manifest['inputs']:
            for term in port['terms']:
                source = term.get('source')
                if source and source.get("kind") == "vision_encoder":
                    if backbone_id(source["model_name"]) != meta["hf_id"]:
                        raise ValueError("Vision backbone identity differs from trained policy")
                    key = 'patch_tokens' if source['slice'] == 'tokens' else 'cls_token'
                    feeds[port['name']] = outputs[key]
    if 'latent' not in feeds:
        name = agent.encoder_port['name']
        feeds['latent'] = agent.encoder.run(['latent'], {name: feeds[name]})[0]
    inputs = {p.name: feeds[p.name] for p in agent.decoder.session.get_inputs()}
    if any(x.dtype != np.float32 or not np.isfinite(x).all() for x in inputs.values()):
        raise ValueError('Policy inputs must be finite float32 arrays')
    actions = agent.decoder.session.run(['actions'], inputs)[0]
    print(json.dumps({'actions': actions.tolist(), 'joint_targets': agent.joint_targets(actions).tolist()}))


if __name__ == '__main__':
    main()
