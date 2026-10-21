"""Read-only checkpoint versus native observation/three-ONNX gate on prepared cases."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--bundle',required=True)
    p.add_argument('--vision-encoder',required=True)
    p.add_argument('--cases-root',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    out=Path(a.out)
    if out.exists(): raise FileExistsError(out)
    import torch
    import onnxruntime as ort
    from tensordict import TensorDict
    import vibe
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
    from vibe.export.agent.case import pin_terrain_family
    from vibe.export.agent.split_sonic import SplitOnnxAgent
    try:
        from .runtime import NativeCase, bgr_input, sha
    except ImportError:
        from runtime import NativeCase, bgr_input, sha
    bundle=json.loads((Path(a.bundle)/'bundle.json').read_text())
    task=bundle['source_manifest']['task_id']
    cfg=load_env_cfg(task,play=True); cfg.scene.num_envs=1
    pin_terrain_family(cfg,'stair_p2_6768415cac75882b')
    ManagerBasedRlEnv.seed(0)
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    result={}
    try:
        agent_cfg=load_rl_cfg(task)
        wrapper=RslRlVecEnvWrapper(env,clip_actions=agent_cfg.clip_actions)
        runner_cls=load_runner_cls(task) or MjlabOnPolicyRunner
        runner=runner_cls(wrapper,asdict(agent_cfg),device='cuda:0')
        checkpoint=bundle['source_manifest']['checkpoint']
        if sha(checkpoint)!=bundle['checkpoint_sha256']: raise ValueError('Checkpoint digest mismatch')
        runner.load(checkpoint,load_cfg={'actor':True},strict=True,map_location='cuda:0')
        model=runner.alg.get_policy().cpu().eval()
        torch.set_num_threads(1)
        agent=SplitOnnxAgent(a.bundle,provider='cpu')
        options=ort.SessionOptions(); options.intra_op_num_threads=1
        vision=ort.InferenceSession(a.vision_encoder,sess_options=options,providers=['CPUExecutionProvider'])
        vi=vision.get_inputs()[0].name
        for file in sorted(Path(a.cases_root).glob('*/case.json')):
            case=NativeCase(file.parent,agent.manifest)
            data=case.initial
            groups={}
            for key,value in data.items():
                if not key.startswith('oracle__'): continue
                path=key[len('oracle__'):].split('__')
                tensor=torch.from_numpy(value)
                if len(path)==2: groups.setdefault(path[0],{})[path[1]]=tensor
                else: groups[path[0]]=tensor
            for port in bundle['source_manifest']['inputs']:
                canonical=port['groups'][0]
                for alias in port['groups'][1:]: groups[alias]=groups[canonical]
            obs=TensorDict(groups,batch_size=[1])
            with torch.inference_mode():
                expected=model(obs).numpy()
                latent=model.encode_tokens(obs).numpy()
            native=case.observations()
            outputs=dict(zip([o.name for o in vision.get_outputs()],vision.run(None,{vi:bgr_input(data['rgb'][0])}),strict=True))
            native['kv_tokens__img_tokens']=outputs['patch_tokens'];native['q_cls']=outputs['cls_token']
            native['latent']=agent.encoder.run(['latent'],{'tokenizer':native['tokenizer']})[0]
            actual=agent.decoder.session.run(['actions'],{x.name:native[x.name] for x in agent.decoder.session.get_inputs()})[0]
            error=float(np.max(np.abs(expected-actual)))
            latent_equal=bool(np.array_equal(latent,native['latent']))
            result[file.parent.name]={'max_action_error':error,'latent_exact':latent_equal,
                'passed':bool(error<=1e-5 and latent_equal),'initial_frame':case.meta['initial_frame']}
            print(file.parent.name,result[file.parent.name],flush=True)
        out.write_text(json.dumps({'checkpoint_sha256':sha(checkpoint),'tolerance':1e-5,'cases':result,'passed':bool(result) and all(x['passed'] for x in result.values())},indent=2))
        if not result or not all(x['passed'] for x in result.values()): raise ValueError('Checkpoint comparison failed; see report')
    finally: env.close()

if __name__=='__main__': main()
