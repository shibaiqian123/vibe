"""Read-only dependency and readiness report. Does not start any robot processes."""
import argparse,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(__doc__);p.add_argument('--config',default=str(ROOT/'configs/sim_only.json'));a=p.parse_args()
c=json.loads(Path(a.config).read_text())
if not c.get('simulation_only') or c.get('network_interface')!='lo':raise SystemExit('Only loopback simulation is allowed')
checks={n:importlib.util.find_spec(n) is not None for n in ('numpy','mujoco','zmq','unitree_sdk2py','onnxruntime')}
print(json.dumps({'python':sys.executable,'dependencies':checks,'config':c,'motor_commands_enabled':False},indent=2))
raise SystemExit(0 if all(checks.values()) and c['status']=='validated' else 2)
