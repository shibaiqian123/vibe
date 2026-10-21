"""Prepare an immutable native-MuJoCo case from the registered training task.

Training framework is used ONLY here to resolve assets/reference and record an oracle.
The playback process never calls the training environment's step().
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--family', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--data-root', required=True)
    p.add_argument('--frame', type=int, default=0)
    a = p.parse_args()
    if Path(os.environ.get('ORCS_DATA_ROOT', '')).resolve() != Path(a.data_root).resolve():
        raise ValueError('--data-root must match ORCS_DATA_ROOT set before task imports')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import mujoco
    import vibe
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from vibe.export.agent.case import pin_terrain_family
    from orcs.core.mdp.observations import robot_root_lin_vel_cmd, robot_root_ang_vel_cmd

    bundle = json.loads((Path(a.bundle)/'bundle.json').read_text())
    if bundle['status'] != 'passed':
        raise ValueError('Unvalidated split bundle')
    manifest = bundle['source_manifest']
    cfg = load_env_cfg(manifest['task_id'], play=True)
    cfg.scene.num_envs = 1
    # Single-world Warp preparation needs enough startup contact slots before RSI.
    # Memory capacity only; contact geometry, solvers and tolerances are unchanged.
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 512)
    cfg.sim.njmax = max(cfg.sim.njmax or 0, 4096)
    pin_terrain_family(cfg, a.family)
    # Deterministic deployment comparison; all noise and randomization are disabled.
    cfg.events = {}
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values():
            term.noise = None
    ManagerBasedRlEnv.seed(0)
    env = ManagerBasedRlEnv(cfg, device=a.device)
    def arr(x):
        return x.detach().cpu().numpy().copy()
    try:
        cmd = env.command_manager.get_term('motion')
        if not 0 <= a.frame < cmd.motion.time_step_total:
            raise ValueError('Initial frame outside reference')
        cmd._sample_init_frame = lambda ids: (torch.zeros_like(ids), torch.full_like(ids, a.frame))
        obs, _ = env.reset()
        robot = env.scene['robot']
        print('INITIAL_FRAME', arr(cmd.time_steps), flush=True)
        model = env.sim.mj_model
        if abs(model.opt.timestep - .005) > 1e-9 or cfg.decimation != 4:
            raise ValueError('Expected 5ms physics and decimation 4')
        if any(type(x).__name__ != 'BuiltinPositionActuatorCfg' for x in cfg.scene.entities['robot'].articulation.actuators):
            raise ValueError('Only verified native position actuators supported')
        mujoco.mj_saveModel(model, str(out/'scene.mjb'))
        # MJB embeds mesh data and preserves compiler-generated actuator parameters.
        fields = {n: arr(getattr(env.sim.data, n))[0] for n in ('qpos','qvel','ctrl')}
        fields['last_action'] = arr(env.action_manager.action)[0]
        fields['default_joint_pos'] = arr(robot.data.default_joint_pos)[0]
        for name in ('tokenizer', 'policy', 'augmentation', 'q_proprio', 'q_cls', 'kv_tokens'):
            x = obs[name]
            if isinstance(x, dict):
                for key, v in x.items(): fields['oracle__'+name+'__'+key] = arr(v)
            else: fields['oracle__'+name] = arr(x)
        fields['rgb'] = arr(env.scene['head_cam'].data.rgb)
        np.savez(out/'initial.npz', **fields)
        motion = cmd.motion
        reference = {n: arr(getattr(motion,n)) for n in ('smpl_joints','smpl_root_quat','smpl_joints_viz','joint_pos','joint_vel','last_action','body_pos_w','body_quat_w')}
        start = int(cmd.time_steps[0])
        if motion.n_clips != 1: raise ValueError('Expected exactly one clip')
        queries, targets = [], []
        with torch.no_grad():
            for frame in range(motion.time_step_total):
                cmd.time_steps[:] = frame
                queries.append(arr(torch.cat((robot_root_lin_vel_cmd(env,'motion'), robot_root_ang_vel_cmd(env,'motion')), -1))[0])
                targets.append(arr(cmd.point_target_pos_w)[0])
        cmd.time_steps[:] = start
        reference['augmentation'] = np.asarray(queries)
        reference['point_targets'] = np.asarray(targets)
        reference['origin'] = arr(env.scene.env_origins)[0]
        np.savez(out/'reference.npz', **reference)
        camera = next(x for x in cfg.scene.sensors if x.name == 'head_cam')
        meta = {'schema':'vibe.native_case.v1','family':a.family,'task':manifest['task_id'],
                'bundle_sha256':digest(Path(a.bundle)/'bundle.json'), 'initial_frame':start,
                'frames':motion.time_step_total,'joint_names':list(robot.joint_names),
                'point_body_names':list(cmd.point_body_names), 'motion_files':motion.motion_files,
                'seed_files':motion.seed_files,'mujoco':mujoco.__version__,
                'data_root':str(Path(a.data_root).resolve()),
                'camera':{'name':'head_cam','width':camera.width,'height':camera.height,
                    'pos':list(camera.pos),'quat':list(camera.quat),'fovy':camera.fovy,
                    'geom_groups':list(camera.enabled_geom_groups),'shadows':camera.use_shadows,'textures':camera.use_textures},
                'clip_actions':load_rl_cfg(manifest['task_id']).clip_actions,
                'preparation_nconmax':cfg.sim.nconmax,'preparation_njmax':cfg.sim.njmax,'randomization':False,'physics_dt':.005,'control_dt':.02,
                'sources': {str(f):digest(f) for f in motion.seed_files+motion.motion_files + [str(Path(f).parent / n) for f in motion.motion_files for n in ('smpl_motion.npz','metadata.json','seed_source.json')]},
                'code':{'vibe':str(Path(vibe.__file__).resolve()),
                        'prepare_sha256':digest(__file__),
                        'hand_urdf':os.environ.get('ORCS_G1_HAND_URDF'),
                        'hand_sha256':digest(os.environ['ORCS_G1_HAND_URDF'])},
                'files':{f:digest(out/f) for f in ('scene.mjb','initial.npz','reference.npz')}}
        (out/'case.json').write_text(json.dumps(meta,indent=2))
        print('PREPARED',out,flush=True)
    finally: env.close()

if __name__ == '__main__': main()
