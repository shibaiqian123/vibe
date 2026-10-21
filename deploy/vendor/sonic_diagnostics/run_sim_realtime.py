"""Simulation-only deadline pacing; production repository remains untouched."""
import inspect
from pathlib import Path
import os
import runpy
import sys
import textwrap

ROOT = Path('/home/cqu023/GR00T-WholeBodyControl')
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from gear_sonic.utils.mujoco_sim import base_sim

source = textwrap.dedent(inspect.getsource(base_sim.BaseSimulator.start))
old = ('elapsed = time.monotonic() - step_start\n'
       '            sleep_time = self.sim_dt - elapsed')
if source.count(old) != 1 or source.count('sim_cnt = 0') != 1:
    raise RuntimeError('Original simulation loop changed; review this wrapper before use')
source = source.replace('sim_cnt = 0', 'sim_cnt = 0\n    next_deadline = time.monotonic()')
source = source.replace(old,
    'next_deadline += self.sim_dt\n'
    '            sleep_time = next_deadline - time.monotonic()\n'
    '            if sleep_time < -1.0:\n'
    '                raise RuntimeError("Cannot sustain real time: >1s behind; no physics steps skipped")')
namespace = {}
exec(compile(source, __file__, 'exec'), base_sim.__dict__, namespace)
base_sim.BaseSimulator.start = namespace['start']
print('SIM_REALTIME: absolute deadlines; unchanged physics dt, PD, DDS and rendering', flush=True)
sys.argv[0] = str(ROOT / 'gear_sonic/scripts/run_sim_loop.py')
runpy.run_path(sys.argv[0], run_name='__main__')
