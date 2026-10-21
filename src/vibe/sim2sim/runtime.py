"""Independent MuJoCo/ONNX loop; no training env.step, DDS, or external forces."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class History:
    """Term-major, chronological history, matching mjlab CircularBuffer."""
    def __init__(self, values, length=10):
        self.values = [np.repeat(np.asarray(x, np.float32)[None], length, axis=0) for x in values]
    def append(self, values):
        for history, value in zip(self.values, values, strict=True):
            history[:-1] = history[1:]
            history[-1] = value
    def flat(self):
        return np.concatenate([x.ravel() for x in self.values])[None].astype(np.float32)


class Deadline:
    """SONIC diagnostic fix: cumulative deadlines, not sleep(dt) each iteration."""
    def __init__(self, dt):
        self.dt, self.next = dt, time.perf_counter()
    def wait(self):
        self.next += self.dt
        delay = self.next - time.perf_counter()
        if delay > 0: time.sleep(delay)
    def resume(self):
        self.next = time.perf_counter()


def bgr_input(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Expected HWC uint8 RGB')
    return np.ascontiguousarray(rgb[None, :, :, ::-1])


def resolve(model, kind, name):
    import mujoco
    # The compiled training scene namespaces robot names.
    found = [mujoco.mj_name2id(model, kind, n) for n in (name, 'robot/'+name)]
    found = set(x for x in found if x >= 0)
    if len(found) != 1: raise ValueError(f'Ambiguous/missing model name {name}: {found}')
    return found.pop()


class RGBPanel:
    """Standard-library GUI, keeping deployment environment dependencies unchanged."""
    def __init__(self):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.title('Actual Theia RGB - native MuJoCo')
        self.text = tk.Label(self.root, text='Waiting for policy input')
        self.text.pack()
        self.label = tk.Label(self.root)
        self.label.pack()
        self.closed = False
        self.root.protocol('WM_DELETE_WINDOW', self.close)
    def pump(self):
        if not self.closed:
            self.root.update_idletasks()
            self.root.update()
    def show(self, rgb, frame, mode):
        if self.closed: return
        from PIL import Image, ImageTk
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((448,252)))
        self.label.configure(image=self.photo)
        self.text.configure(text=f'Frame {frame} | {mode} | actual RGB before exported BGR conversion')
        self.pump()
    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()


class NativeCase:
    def __init__(self, directory, manifest):
        import mujoco
        import torch
        from mocke.sonic.mdp.observations import sonic_smpl_tokenizer
        self.mujoco, self.torch, self.tokenizer_fn = mujoco, torch, sonic_smpl_tokenizer
        directory = Path(directory)
        self.meta = json.loads((directory/'case.json').read_text())
        for filename, value in self.meta['files'].items():
            if sha(directory/filename) != value: raise ValueError(f'Changed case file: {filename}')
        self.initial = dict(np.load(directory/'initial.npz', allow_pickle=False))
        self.reference = dict(np.load(directory/'reference.npz', allow_pickle=False))
        self.model = mujoco.MjModel.from_binary_path(str(directory/'scene.mjb'))
        self.data = mujoco.MjData(self.model)
        names = manifest['action']['joint_names']
        if list(self.meta['joint_names']) != names: raise ValueError('Robot joint order differs from exported actor')
        self.jids = np.array([resolve(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names])
        self.qids = self.model.jnt_qposadr[self.jids]
        self.vids = self.model.jnt_dofadr[self.jids]
        self.root = resolve(self.model,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
        self.torso = resolve(self.model,mujoco.mjtObj.mjOBJ_BODY,'torso_link')
        self.cam = resolve(self.model,mujoco.mjtObj.mjOBJ_CAMERA,'head_cam')
        self.points = [resolve(self.model,mujoco.mjtObj.mjOBJ_BODY,n) for n in self.meta['point_body_names']]
        self.aids = []
        for j in self.jids:
            ids = np.flatnonzero((self.model.actuator_trnid[:,0] == j) & (self.model.actuator_trntype == 0))
            if len(ids) != 1: raise ValueError('Only one native position actuator per joint supported')
            self.aids.append(int(ids[0]))
        if len(self.aids) != self.model.nu: raise ValueError('Unmapped actuators')
        action = manifest['action']
        self.default = np.asarray(action['default_joint_pos'],np.float32)
        self.scale = np.asarray(action['scale'],np.float32)
        if not np.allclose(self.initial['default_joint_pos'],self.default,atol=1e-7):
            raise ValueError('Default joint pose mismatch')
        for i, aid in enumerate(self.aids):
            if not np.allclose(self.model.actuator_gainprm[aid,0],action['stiffness'][i],rtol=1e-6):
                raise ValueError('PD stiffness mismatch')
            if not np.allclose(self.model.actuator_biasprm[aid,1:3],[-action['stiffness'][i],-action['damping'][i]],rtol=1e-6):
                raise ValueError('Native actuator does not implement expected PD')
        motion = SimpleNamespace(**{k:torch.tensor(self.reference[k]) for k in ('smpl_joints','smpl_root_quat','joint_pos')})
        self.command = SimpleNamespace(motion=motion,future_frames=self.future_frames)
        self.robot = SimpleNamespace(data=SimpleNamespace())
        self.shim = SimpleNamespace(num_envs=1,scene={'robot':self.robot},command_manager=SimpleNamespace(get_term=lambda _:self.command))
        self.reset()
    def future_frames(self, steps, skip):
        return self.torch.tensor(np.minimum(self.frame+np.arange(steps)*skip,self.meta['frames']-1)[None])
    def reset(self):
        self.mujoco.mj_resetData(self.model,self.data)
        for name in ('qpos','qvel','ctrl'): getattr(self.data,name)[:] = self.initial[name]
        self.frame = self.meta['initial_frame']
        self.last = self.initial['last_action'].astype(np.float32).copy()
        self.mujoco.mj_forward(self.model,self.data)
        self.history = History(self.terms())
    def terms(self):
        # mj_objectVelocity(..., local=1) uses inertial axes for BODY.
        # Training explicitly rotates WORLD angular velocity into LINK axes.
        rot = self.data.xmat[self.root].reshape(3,3)
        return [rot.T @ self.data.cvel[self.root,:3],self.data.qpos[self.qids]-self.default,self.data.qvel[self.vids],self.last,rot.T @ np.array([0.,0.,-1.])]
    def observations(self):
        self.robot.data.root_link_quat_w = self.torch.tensor(self.data.xquat[self.root][None],dtype=self.torch.float32)
        with self.torch.no_grad(): tokenizer = self.tokenizer_fn(self.shim).numpy()
        av,jp,jv,_,gravity = self.terms()
        return {'tokenizer':tokenizer,'policy':self.history.flat(),
            'augmentation':self.reference['augmentation'][self.frame][None].astype(np.float32),
            'q_proprio':np.concatenate([gravity,av,jp,jv])[None].astype(np.float32)}
    def step(self,action):
        action = np.asarray(action,np.float32).reshape(29)
        if not np.isfinite(action).all(): raise FloatingPointError('Non-finite action')
        clip = self.meta['clip_actions']
        if clip is not None: action = np.clip(action,-clip,clip)
        target = self.default+self.scale*action
        self.data.ctrl[self.aids] = target
        for _ in range(4): self.mujoco.mj_step(self.model,self.data)
        self.mujoco.mj_forward(self.model,self.data)
        self.last = action.copy()
        self.frame += 1
        self.history.append(self.terms())
        return target


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--case',required=True)
    p.add_argument('--bundle',required=True)
    p.add_argument('--vision-encoder',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--provider',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--steps',type=int,default=128)
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--ort-threads',type=int,default=1)
    p.add_argument('--viewer-hz',type=float,default=15.)
    p.add_argument('--hold-final',action='store_true')
    p.add_argument('--realtime',action='store_true')
    p.add_argument('--rgb-mode',choices=['normal','gray'],default='normal')
    p.add_argument('--check-only',action='store_true')
    a = p.parse_args()
    if a.viewer_hz <= 0: raise ValueError('--viewer-hz must be positive')
    if a.steps < 1: raise ValueError('--steps must be positive')
    if a.hold_final and not a.viewer: raise ValueError('--hold-final requires --viewer')
    out = Path(a.out); out.mkdir(parents=True,exist_ok=False)
    import mujoco
    import onnxruntime as ort
    from vibe.export.agent.split_sonic import SplitOnnxAgent
    from vibe.export.agent.run_split import backbone_id
    if a.provider == 'cuda' and 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('CUDA provider requested but not installed; no silent CPU fallback')
    if a.ort_threads < 1: raise ValueError('--ort-threads must be positive')
    agent = SplitOnnxAgent(a.bundle,provider=a.provider)
    options = ort.SessionOptions(); options.intra_op_num_threads = a.ort_threads; options.inter_op_num_threads = 1
    # Reuse official loading/hash/contract checks, replace sessions only to bound thread pools.
    agent.encoder = ort.InferenceSession(str(Path(a.bundle)/'motion_encoder.onnx'),sess_options=options,providers=agent.providers)
    agent.decoder.session = ort.InferenceSession(str(Path(a.bundle)/'visual_decoder.onnx'),sess_options=options,providers=agent.providers)
    if agent.bundle['status'] != 'passed': raise ValueError('Unvalidated ONNX bundle')
    case = NativeCase(a.case,agent.manifest)
    if case.meta['bundle_sha256'] != sha(Path(a.bundle)/'bundle.json'): raise ValueError('Wrong case/model pairing')
    session = ort.InferenceSession(a.vision_encoder,sess_options=options,providers=agent.providers)
    vm = json.loads(session.get_modelmeta().custom_metadata_map['manifest'])
    if vm['schema'] != 'vision.onnx.v1' or vm['input']['layout'] != 'bgr_hwc_uint8': raise ValueError('Unsupported vision preprocessing')
    camera = case.meta['camera']; h,w = camera['height'],camera['width']
    if (h,w) != (vm['input']['height'],vm['input']['width']): raise ValueError('Camera resolution mismatch')
    renderer = mujoco.Renderer(case.model,height=h,width=w)
    option = mujoco.MjvOption(); option.geomgroup[:] = 0
    option.geomgroup[camera['geom_groups']] = 1
    def vision(rgb):
        raw = bgr_input(rgb) # MuJoCo RGB -> exported BGR input (normalization is inside ONNX).
        values = session.run(None,{vm['input']['name']:raw})
        outputs = dict(zip([x.name for x in session.get_outputs()],values,strict=True))
        feeds = {}
        for port in agent.manifest['inputs']:
            for term in port['terms']:
                source = term.get('source',{})
                if source.get('kind') == 'vision_encoder':
                    if backbone_id(source['model_name']) != vm['hf_id']: raise ValueError('Vision backbone mismatch')
                    feeds[port['name']] = outputs['patch_tokens' if source['slice']=='tokens' else 'cls_token']
        return feeds
    def action(obs,rgb):
        feed = {**obs,**vision(rgb)}
        feed['latent'] = agent.encoder.run(['latent'],{agent.encoder_port['name']:feed['tokenizer']})[0]
        inputs = {x.name:feed[x.name] for x in agent.decoder.session.get_inputs()}
        if not all(np.isfinite(v).all() for v in inputs.values()): raise FloatingPointError('Invalid model input')
        return agent.decoder.session.run(['actions'],inputs)[0]
    obs = case.observations()
    errors = {k:float(np.max(np.abs(v-case.initial['oracle__'+k]))) for k,v in obs.items()}
    oracle_obs = {k:case.initial['oracle__'+k] for k in obs}
    rgb_oracle = case.initial['rgb'][0]
    ours = action(obs,rgb_oracle); expected = action(oracle_obs,rgb_oracle)
    errors['action_same_rgb'] = float(np.max(np.abs(ours-expected)))
    # Check actual vision features, not just action sensitivity to those features.
    for name,value in vision(rgb_oracle).items():
        key = 'oracle__'+name
        errors[name] = float(np.max(np.abs(value-case.initial[key])))
    oracle_feed = {**oracle_obs, **{k:case.initial['oracle__'+k] for k in vision(rgb_oracle)}}
    oracle_feed['latent'] = agent.encoder.run(['latent'],{agent.encoder_port['name']:oracle_feed['tokenizer']})[0]
    actual_latent = agent.encoder.run(['latent'],{agent.encoder_port['name']:obs['tokenizer']})[0]
    errors['latent'] = float(np.max(np.abs(actual_latent-oracle_feed['latent'])))
    oracle_action = agent.decoder.session.run(['actions'],{x.name:oracle_feed[x.name] for x in agent.decoder.session.get_inputs()})[0]
    errors['action_training_features'] = float(np.max(np.abs(ours-oracle_action)))
    gated = ('tokenizer','policy','augmentation','q_proprio','latent','action_same_rgb','action_training_features')
    report = {'gated_fields':gated,'action_tolerance':1e-5,'feature_errors_diagnostic_only':True,'same_state_errors':errors,'status':'blocked','rgb_mode':a.rgb_mode,'steps':0,'providers':session.get_providers(),'ort_threads':a.ort_threads,'runtime_sha256':sha(__file__),'case_sha256':sha(Path(a.case)/'case.json')}
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    if any(errors[k]>1e-5 for k in gated):
        renderer.close()
        raise ValueError(f'Same-state gate failed: {errors}')
    if a.check_only:
        report['status']='same_state_passed'
        (out/'validation.json').write_text(json.dumps(report,indent=2)); renderer.close(); return
    viewer = None; requests=[]; paused=False
    if a.viewer:
        from orcs.core.data.smpl import SMPL_PARENTS
        import mujoco.viewer
        panel = RGBPanel()
        viewer = mujoco.viewer.launch_passive(case.model,case.data,key_callback=lambda key:requests.append(key))
        viewer.cam.lookat[:] = case.data.xpos[case.root]
        viewer.cam.distance = 3.0
    last_display=0.
    rows=[]; deadline=Deadline(.02); start=time.perf_counter(); start_sim=case.data.time
    try:
        episode_steps=0
        while episode_steps < a.steps and case.frame < case.meta['frames']-1:
            while requests or paused:
                while requests:
                    key=requests.pop(0)
                    if key == 32: paused=not paused; deadline.resume()
                    if key == 82:
                        case.reset(); deadline.resume(); episode_steps=0
                        rows.append({'event':'reset','wall':time.perf_counter()})
                if viewer:
                    if not viewer.is_running(): break
                    viewer.sync(); panel.pump()
                if not paused: break
                time.sleep(.02)
            if viewer and not viewer.is_running(): break
            t=time.perf_counter()
            renderer.update_scene(case.data,camera=case.cam,scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = camera['shadows']
            if not camera['textures']:
                for geom in renderer.scene.geoms[:renderer.scene.ngeom]: geom.texid = -1
            rgb=renderer.render().copy()
            actual=rgb if a.rgb_mode=='normal' else np.full_like(rgb,127)
            input_frame=case.frame
            rendered=time.perf_counter()
            obs=case.observations(); act=action(obs,actual)
            inferred=time.perf_counter()
            target=case.step(act)
            stepped=time.perf_counter()
            error=np.linalg.norm(case.data.xpos[case.points]-case.reference['point_targets'][case.frame],axis=-1)
            row={'input_frame':input_frame,'frame':case.frame,'sim_time':float(case.data.time),'compute_s':time.perf_counter()-t,
                'render_s':rendered-t,'observation_inference_s':inferred-rendered,'physics_s':stepped-inferred,
                'root_height':float(case.data.xpos[case.root,2]),
                'torque_saturation_fraction':float(np.mean(np.abs(case.data.actuator_force[case.aids]) >= np.max(np.abs(case.model.actuator_forcerange[case.aids]),axis=1)*.99)),'tracking_mean':float(error.mean()),'tracking_max':float(error.max()),
                'min_contact_dist':float(min([c.dist for c in case.data.contact[:case.data.ncon]],default=0)),
                'root_up':float(case.data.xmat[case.root].reshape(3,3)[2,2]),'action':act[0].tolist(),
                'target':target.tolist(),'qpos':case.data.qpos.tolist()}
            rows.append(row)
            episode_steps += 1
            if len(rows)==1 or input_frame in (228,286,436,case.meta['frames']-2):
                np.save(out/'actual_theia_rgb.npy',actual)
                import PIL.Image
                PIL.Image.fromarray(actual).save(out/'actual_theia_rgb.png')
                PIL.Image.fromarray(actual).save(out/f'rgb_{input_frame:04d}.png')
            if not np.isfinite(case.data.qpos).all() or row['root_up']<.25:
                report['status']='stopped_unstable'; break
            if viewer and time.perf_counter()-last_display >= 1. / a.viewer_hz:
                display_start=time.perf_counter()
                # Visualization only: user_scn is not part of collision geometry or camera RGB.
                with viewer.lock():
                    viewer.cam.lookat[:] = case.data.xpos[case.root]
                    viewer.user_scn.ngeom = 0
                    joints=case.reference['smpl_joints_viz'][case.frame]+case.reference['origin']
                    for j,parent in enumerate(SMPL_PARENTS):
                        if parent < 0: continue
                        geom=viewer.user_scn.geoms[viewer.user_scn.ngeom]
                        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.1,.8,.9,.7]))
                        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.008,joints[parent],joints[j])
                        viewer.user_scn.ngeom += 1
                viewer.sync(); panel.show(actual,input_frame,a.rgb_mode)
                last_display=time.perf_counter()
                row['display_s']=last_display-display_start
            if a.realtime: deadline.wait()
        else: report['status']='rollout_completed_not_quality_certified'
    finally:
        elapsed=time.perf_counter()-start
        report.update(steps=len(rows),wall_seconds=elapsed,sim_seconds=float(case.data.time-start_sim),
            realtime_factor=float(case.data.time-start_sim)/elapsed,model_hashes=agent.bundle['files'],vision_sha256=sha(a.vision_encoder))
        (out/'validation.json').write_text(json.dumps(report,indent=2))
        (out/'trajectory.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        renderer.close()
        if viewer:
            while a.hold_final and viewer.is_running():
                viewer.sync(); panel.pump(); time.sleep(.02)
            viewer.close(); panel.close()
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
