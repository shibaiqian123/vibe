"""Bounded 64-env visual PPO validation; no long-training entry."""
import os,json,hashlib,traceback
from pathlib import Path
from dataclasses import asdict
import torch,numpy as np
import mjlab,orcs,vibe
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from vibe.core.mdp.observations import image_feature
from PIL import Image

def digest(items):
 h=hashlib.sha256()
 for n,p in items: h.update(n.encode());h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def run():
 r=Path(os.environ['ORCS_DATA_ROOT']);out=r/'ppo_validation';out.mkdir(exist_ok=False)
 report={}
 def save(): (out/'report.json').write_text(json.dumps(report,indent=2))
 try:
  task='Vibe-PerLoco-GrailStairs-ImgFeat-Ext-Smpl';cfg=load_env_cfg(task);cfg.scene.num_envs=64
  agent=load_rl_cfg(task);agent.save_interval=10
  raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode=None);env=RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
  runner=MjlabOnPolicyRunner(env,asdict(agent),log_dir=str(out/'train'),device='cuda:0');env.reset();obs=env.get_observations();actor=runner.alg.actor;actor.eval()
  command=raw.command_manager.get_term('motion');assert len(command.motion.motion_files)==6
  seen=set(command._clip_ids.cpu().tolist());assert seen==set(range(6)),seen
  report['reset_clip_ids']=sorted(seen)
  t=command.time_steps
  # Seed loader indices and command indices share the same concatenated clock.
  report['rsi_joint_max_error']=float((command.robot.data.joint_pos-command.motion.joint_pos[t]).abs().max())
  limits=command.robot.data.soft_joint_pos_limits
  expected=command.motion.joint_pos[t].clamp(limits[...,0],limits[...,1])
  report['rsi_after_soft_limit_error']=float((command.robot.data.joint_pos-expected).abs().max())
  lo,hi=command.cfg.joint_position_range
  report['configured_reset_joint_noise']=[lo,hi]
  lower=(command.motion.joint_pos[t]+lo).clamp(limits[...,0],limits[...,1])
  upper=(command.motion.joint_pos[t]+hi).clamp(limits[...,0],limits[...,1])
  assert torch.all(command.robot.data.joint_pos>=lower-1e-5)
  assert torch.all(command.robot.data.joint_pos<=upper+1e-5)
  with torch.no_grad():
   y=actor(obs);base=actor._head(actor._head_input(actor.decoder.base(torch.cat([actor.encode_tokens(obs),actor._get_proprio(obs)],-1))),False)
  report['zero_lora_error']=float((y-base).abs().max());assert torch.allclose(y,base,atol=1e-5,rtol=1e-5)
  frozen=[(n,p) for n,p in actor.named_parameters() if not p.requires_grad];frozen_before=digest(frozen)
  before={n:p.detach().cpu().clone() for n,p in actor.named_parameters() if p.requires_grad}
  critic_before=digest(runner.alg.critic.named_parameters())
  models=list(image_feature._MODELS.values());theia_before=[digest(m.state_dict().items()) for m in models]
  assert all(not p.requires_grad for m in models for p in m.parameters())
  rgb=raw.scene['head_cam'].data.rgb
  report['rgb_shape']=list(rgb.shape);report['rgb_range']=[int(rgb.min()),int(rgb.max())]
  assert int(rgb.max())>int(rgb.min())
  for i in range(6): Image.fromarray(rgb[i].cpu().numpy()).save(out/f'rgb_{i}.png')
  report['camera']=asdict(next(s for s in cfg.scene.sensors if s.name=='head_cam'));report['visual_contract']=agent.actor['visual_contract_id'];save()
  runner.learn(num_learning_iterations=20)
  changed=[n for n,p in actor.named_parameters() if n in before and not torch.equal(before[n],p.detach().cpu())]
  assert any(n.startswith('extractors.') for n in changed)
  assert any(n.startswith('decoder.adapters.') for n in changed)
  assert digest(frozen)==frozen_before
  assert [digest(m.state_dict().items()) for m in models]==theia_before
  assert digest(runner.alg.critic.named_parameters())!=critic_before
  report['changed_parameters']=changed;report['base_unchanged']=True;report['theia_unchanged']=True;report['critic_changed']=True
  actor.eval();obs=env.get_observations();altered=obs.clone()
  # Re-encode a black RGB frame through the same frozen backbone; keep body/SMPL fixed.
  m=models[0];rgb=raw.scene['head_cam'].data.rgb
  x=torch.zeros_like(rgb,dtype=torch.float32).permute(0,3,1,2)/255
  mean=torch.tensor([.485,.456,.406],device=x.device).view(1,3,1,1);std=torch.tensor([.229,.224,.225],device=x.device).view(1,3,1,1)
  with torch.no_grad():
   h=m.backbone.model(pixel_values=(x-mean)/std,interpolate_pos_encoding=True).last_hidden_state
  # Replace only visual channels with the black-image encoding.
  altered['kv_tokens']['img_tokens']=h[:,1:];altered['q_cls']=h[:,0]
  with torch.no_grad(): delta=(actor(obs)-actor(altered)).abs().max().item()
  report['black_rgb_action_delta']=delta;assert delta>0
  ckpt=out/'train'/'verified_20.pt';runner.save(str(ckpt));report['checkpoint']=str(ckpt);save()
  bad={k:v.clone() for k,v in actor.state_dict().items()};bad['_vibe_visual_contract'][0]^=1
  try: actor.load_state_dict(bad)
  except ValueError: report['wrong_contract_rejected']=True
  else: raise AssertionError('wrong contract accepted')
  resumed=MjlabOnPolicyRunner(env,asdict(agent),log_dir=str(out/'resume'),device='cuda:0');resumed.load(str(ckpt))
  assert digest(resumed.alg.actor.state_dict().items())==digest(actor.state_dict().items())
  report['resume_iteration']=resumed.current_learning_iteration
  resumed.learn(num_learning_iterations=2);report['resume_final_iteration']=resumed.current_learning_iteration
  report['status']='passed';save();env.close()
 except Exception as e:
  report['status']='failed';report['error']=repr(e);save();raise
if __name__=='__main__':run()
