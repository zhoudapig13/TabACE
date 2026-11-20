"""Resume public official checkpoint downloads using bounded HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

RUN=Path(__file__).resolve().parents[1]


def download(repo,name,size):
    out=RUN/'checkpoints'/name
    if out.exists() and out.stat().st_size==size:return
    partial=out.with_suffix('.partial')
    url=f'https://huggingface.co/{repo}/resolve/main/{name}'
    offset=partial.stat().st_size if partial.exists() else 0
    while offset<size:
        stop=min(size-1,offset+8*1024*1024-1)
        for attempt in range(8):
            try:
                r=requests.get(url,headers={'Range':f'bytes={offset}-{stop}'},timeout=(20,90))
                r.raise_for_status()
                if r.status_code==200 and len(r.content)==size:
                    partial.write_bytes(r.content);offset=size;break
                if r.status_code!=206 or len(r.content)!=stop-offset+1:
                    raise IOError(f'bad range response: status={r.status_code}, bytes={len(r.content)}')
                with partial.open('ab') as f:f.write(r.content)
                offset=stop+1
                print('PROGRESS',name,offset,size,flush=True);break
            except Exception as e:
                print('RETRY',name,offset,attempt,type(e).__name__,flush=True)
                if attempt==7:raise
                time.sleep(min(2**attempt,20))
    digest=hashlib.sha256()
    with partial.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):digest.update(chunk)
    partial.replace(out)
    out.with_suffix('.source.json').write_text(json.dumps({'repo':repo,'filename':name,
        'bytes':size,'sha256':digest.hexdigest(),'method':'resumed official HTTP range download'},indent=2))
    print('READY',out,flush=True)


if __name__=='__main__':
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(download,*x) for x in [
            ('Prior-Labs/tabpfn_3','tabpfn-v3-regressor-v3_default.ckpt',233289807),
            ('Prior-Labs/tabpfn_2_5','tabpfn-v2.5-regressor-v2.5_default.ckpt',40831995)]]
        for f in futures:f.result()
