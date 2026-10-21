"""Bounded read-only decoder for the extracted SONIC manager wire format."""
import importlib.util,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'vendor/sonic/gear_sonic/utils/teleop/zmq/zmq_planner_sender.py'
spec=importlib.util.spec_from_file_location('sonic_packer',SOURCE)
packer=importlib.util.module_from_spec(spec);spec.loader.exec_module(packer)

def decode(packet):
    topic=next((t for t in (b'command',b'planner',b'pose') if packet.startswith(t)),None)
    if topic is None:raise ValueError('Unknown SONIC topic')
    start=len(topic);end=start+packer.HEADER_SIZE
    if len(packet)<end or len(packet)>16*1024*1024:raise ValueError('Invalid packet size')
    header=json.loads(packet[start:end].rstrip(b'\0'))
    if header.get('endian')!='le':raise ValueError('Expected little endian')
    types={'u8':np.dtype('u1'),'bool':np.dtype('?'),'f32':np.dtype('<f4'),'f64':np.dtype('<f8'),'i32':np.dtype('<i4'),'i64':np.dtype('<i8')}
    values={};offset=end
    for field in header['fields']:
        name=field['name'];shape=field['shape']
        if name in values or not isinstance(shape,list) or any(type(x)!=int or x<0 for x in shape):raise ValueError('Invalid field')
        dtype=types[field['dtype']];count=math.prod(shape);nbytes=count*dtype.itemsize
        if offset+nbytes>len(packet):raise ValueError('Truncated payload')
        values[name]=np.frombuffer(packet,dtype,count,offset).reshape(shape).copy();offset+=nbytes
        if not np.isfinite(values[name]).all():raise ValueError('Non-finite '+name)
    if offset!=len(packet):raise ValueError('Unexpected trailing bytes')
    return topic.decode(),values

def describe(topic,fields):
    return {'topic':topic,'fields':{n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in fields.items()},
      'deployment_ready':False,'reason':'Wire decoding is not tokenizer/planner/VR compatibility validation'}
