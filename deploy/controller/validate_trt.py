"""No actuation: compare SONIC TRT wrapper to the validated split ONNX chain."""
import argparse,json,hashlib,traceback
from pathlib import Path
import numpy as np
from trt_session import Session

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('bundle','vision','case','library','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    report={'status':'blocked','action_tolerance':1e-5,'precision':'FP32; TF32 disabled','cases':[]}
    sessions=[]
    try:
        import onnxruntime as ort
        from vibe.export.agent.split_sonic import SplitOnnxAgent
        from vibe.export.agent.run_split import backbone_id
        agent=SplitOnnxAgent(a.bundle,provider='cpu')
        if agent.bundle['status']!='passed':raise ValueError('Unvalidated bundle')
        report['model_sha256']={n:sha(Path(a.bundle)/(n+'.onnx')) for n in ('motion_encoder','visual_decoder')}
        report['model_sha256']['theia']=sha(a.vision)
        report['library_sha256']=sha(a.library)
        meta=json.loads((Path(a.case)/'case.json').read_text())
        if meta['bundle_sha256']!=sha(Path(a.bundle)/'bundle.json'):raise ValueError('Case bundle mismatch')
        for f,h in meta['files'].items():
            if sha(Path(a.case)/f)!=h:raise ValueError('Case modified: '+f)
        initial=dict(np.load(Path(a.case)/'initial.npz',allow_pickle=False))
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        paths=[Path(a.bundle)/'motion_encoder.onnx',Path(a.bundle)/'visual_decoder.onnx',Path(a.vision)]
        references=[ort.InferenceSession(str(x),sess_options=opts,providers=['CPUExecutionProvider']) for x in paths]
        from prepare_vision_trt import prepare
        adapted=out/'vision_float.onnx'
        report['vision_adaptation']=prepare(a.vision,adapted)
        for path in [*paths[:2],adapted]:sessions.append(Session(a.library,path,out/'engine_cache'))
        vm=json.loads(references[2].get_modelmeta().custom_metadata_map['manifest'])
        if vm['input']['layout']!='bgr_hwc_uint8':raise ValueError('Unexpected image interface')
        for mode in ('normal','gray'):
            rgb=initial['rgb'][0] if mode=='normal' else np.full_like(initial['rgb'][0],127)
            image=np.ascontiguousarray(rgb[None,:,:,::-1])
            def chain(trt):
                def run(i,feeds):
                    if trt:
                        if i==2:feeds={k:v.astype(np.float32) for k,v in feeds.items()}
                        return sessions[i].run(feeds)
                    return dict(zip([x.name for x in references[i].get_outputs()],references[i].run(None,feeds)))
                latent=run(0,{agent.encoder_port['name']:initial['oracle__tokenizer']})['latent']
                features=run(2,{vm['input']['name']:image})
                feed={'latent':latent}
                for port in agent.manifest['inputs']:
                    if 'oracle__'+port['name'] in initial:feed[port['name']]=initial['oracle__'+port['name']]
                    for term in port['terms']:
                        src=term.get('source',{})
                        if src.get('kind')=='vision_encoder':
                            if backbone_id(src['model_name'])!=vm['hf_id']:raise ValueError('Backbone mismatch')
                            feed[port['name']]=features['patch_tokens' if src['slice']=='tokens' else 'cls_token']
                names=[x.name for x in references[1].get_inputs()]
                return latent,run(1,{n:feed[n] for n in names})['actions']
            l0,y0=chain(False);l1,y1=chain(True)
            err=float(np.max(np.abs(y0-y1)));same=bool(np.array_equal(l0,l1))
            report['cases'].append({'mode':mode,'latent_exact':same,'latent_max_abs':float(np.max(np.abs(l0-l1))),'action_max_abs':err,'passed':same and err<=1e-5})
        report['status']='passed' if all(x['passed'] for x in report['cases']) else 'blocked_numerical_mismatch'
    except Exception as e:
        report['error']=str(e);report['traceback']=traceback.format_exc()
    finally:
        for s in sessions:s.close()
        (out/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));return 0 if report['status']=='passed' else 2
if __name__=='__main__':raise SystemExit(main())
