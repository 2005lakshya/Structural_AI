"""Fetch one horizontal band of one stage of one krkCMd TIFF via HTTP range requests,
plus the gridline ROI lines. Usage: fetch_krk.py SERIES IMAGE_NO STAGE OUTDIR"""
import json, struct, sys, time, zipfile, io, os, requests, numpy as np
from concurrent.futures import ThreadPoolExecutor
from remotezip import RemoteZip
url="https://zenodo.org/api/records/14568863/files/krkCMd_images.zip/content"
series, img_no, stage, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(outdir, exist_ok=True)
def rng(a,b):
    for _ in range(6):
        try:
            r=requests.get(url,headers={"Range":f"bytes={a}-{b}"},timeout=120)
            if r.status_code==206: return r.content
        except Exception: time.sleep(2)
    raise RuntimeError("range failed")
tif=f"{series}/{series}_Image{img_no}.tif"; roi=f"{series}/ROI_{series.replace('CMd_','')}_Image{img_no}.zip"
with RemoteZip(url) as z:
    i=z.getinfo(tif); hdr, size = i.header_offset, i.file_size
    roizip=z.read(roi)
lh=rng(hdr,hdr+29); fnl,exl=struct.unpack("<HH",lh[26:30]); data0=hdr+30+fnl+exl
def ifd(buf,off):
    n=struct.unpack(">H",buf[off:off+2])[0]; e={}
    for k in range(n):
        tag,typ,cnt,val=struct.unpack(">HHII",buf[off+2+12*k:off+14+12*k]); e[tag]=val
    return e, struct.unpack(">I",buf[off+2+12*n:off+6+12*n])[0]
head=rng(data0,data0+4095); e,nxt=ifd(head,8); W,H=e[256],e[257]
tail=rng(data0+nxt,data0+size-1); offs=[e[273]]; o=0
while True:
    ee,nx=ifd(tail,o); offs.append(ee[273])
    if nx==0: break
    o=nx-nxt
zf=zipfile.ZipFile(io.BytesIO(roizip)); lines=[]
for n in zf.namelist():
    b=zf.read(n); x1,y1,x2,y2=struct.unpack(">4f",b[18:34]); lines.append((n[:-4],x1,y1,x2,y2))
ys=[v for l in lines for v in (l[2],l[4])]; r0=max(int(min(ys))-60,0); r1=min(int(max(ys))+60,H)
C=3; a=data0+offs[stage-1]+r0*W*C; b=data0+offs[stage-1]+r1*W*C-1
chunks=[(s,min(s+2_000_000-1,b)) for s in range(a,b+1,2_000_000)]
buf=bytearray(b-a+1)
def get(c): buf[c[0]-a:c[1]-a+1]=rng(*c)
t0=time.time()
with ThreadPoolExecutor(8) as ex: list(ex.map(get,chunks))
arr=np.frombuffer(bytes(buf),np.uint8).reshape(r1-r0,W,C)
gray=(0.299*arr[...,0]+0.587*arr[...,1]+0.114*arr[...,2]).astype(np.uint8)
tag=f"{series}_img{img_no}_stage{stage}"
np.save(os.path.join(outdir,tag+".npy"),gray)
json.dump(dict(series=series,image=img_no,stage=stage,row0=r0,row1=r1,width=W,height=H,lines=lines),open(os.path.join(outdir,tag+".json"),"w"))
print("saved",tag,gray.shape,round(time.time()-t0),"s")
