"""Observe SONIC manager data without starting or sending any robot commands."""
import argparse,json,time
from pathlib import Path
from wire import decode,describe

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--port',type=int,default=15556);p.add_argument('--seconds',type=float,default=10);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    import zmq
    ctx=zmq.Context();sock=ctx.socket(zmq.SUB);sock.setsockopt(zmq.LINGER,0)
    sock.connect(f'tcp://127.0.0.1:{a.port}');sock.setsockopt(zmq.SUBSCRIBE,b'')
    result={'status':'no_messages','topics':{},'errors':[]};start=time.monotonic()
    try:
        while time.monotonic()-start<a.seconds:
            if not sock.poll(100):continue
            try:
                topic,fields=decode(sock.recv());result['topics'][topic]=describe(topic,fields);result['status']='received_not_validated'
            except Exception as e:result['errors'].append(str(e))
    finally:sock.close();ctx.term()
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
