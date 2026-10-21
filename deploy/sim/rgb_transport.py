"""RGB side channel for the SONIC DDS loop; never emits motor commands."""
import json,time
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Frame:
    episode: int
    step: int
    sim_time: float
    rgb: np.ndarray

def pack(frame):
    rgb=np.asarray(frame.rgb)
    if rgb.shape!=(63,112,3) or rgb.dtype!=np.uint8:raise ValueError('Expected 63x112x3 RGB uint8')
    if frame.episode<0 or frame.step<0 or not np.isfinite(frame.sim_time) or frame.sim_time<0:raise ValueError('Invalid frame clock')
    header={'schema':'vibe.sim.rgb.v1','episode':frame.episode,'step':frame.step,'sim_time':frame.sim_time,'shape':[63,112,3],'color':'RGB','dtype':'uint8'}
    return [json.dumps(header,separators=(',',':')).encode(),np.ascontiguousarray(rgb).tobytes()]

def unpack(parts):
    if len(parts)!=2 or len(parts[0])>2048:raise ValueError('Malformed RGB message')
    h=json.loads(parts[0])
    if h.get('schema')!='vibe.sim.rgb.v1' or h.get('shape')!=[63,112,3] or h.get('color')!='RGB' or h.get('dtype')!='uint8':raise ValueError('RGB contract mismatch')
    if len(parts[1])!=63*112*3:raise ValueError('RGB size mismatch')
    if type(h['episode'])!=int or type(h['step'])!=int:raise ValueError('Invalid clock types')
    f=Frame(h['episode'],h['step'],float(h['sim_time']),np.frombuffer(parts[1],np.uint8).reshape(63,112,3).copy())
    pack(f)
    return f

def match(frame,episode,step,sim_time):
    if (frame.episode,frame.step)!=(episode,step) or abs(frame.sim_time-sim_time)>1e-7:
        raise ValueError('Stale/future RGB or reset mismatch')
    return frame.rgb

class Publisher:
    def __init__(self,port=15558):
        import zmq
        self.zmq=zmq;self.context=zmq.Context();self.socket=self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM,2);self.socket.setsockopt(zmq.LINGER,0)
        self.socket.bind(f'tcp://127.0.0.1:{port}')
    def send(self,frame):self.socket.send_multipart(pack(frame))
    def close(self):self.socket.close();self.context.term()
