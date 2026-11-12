"""GameCube CMPR (GX_TF_CMPR, fmt 14) decoder -> RGBA numpy array."""
import struct
import numpy as np

def _rgb565(v):
    r=(v>>11)&31; g=(v>>5)&63; b=v&31
    return ((r<<3)|(r>>2), (g<<2)|(g>>4), (b<<3)|(b>>2))

def decode_cmpr(data, w, h):
    W=(w+7)//8*8; H=(h+7)//8*8
    img=np.zeros((H,W,4),np.uint8); pos=0
    for ty in range(0,H,8):
        for tx in range(0,W,8):
            for sb in range(4):
                c0,c1=struct.unpack('>HH',data[pos:pos+4]); idx=data[pos+4:pos+8]; pos+=8
                p0=_rgb565(c0); p1=_rgb565(c1)
                pal=[p0+(255,),p1+(255,)]
                if c0>c1:
                    pal.append(tuple((2*a+b)//3 for a,b in zip(p0,p1))+(255,))
                    pal.append(tuple((a+2*b)//3 for a,b in zip(p0,p1))+(255,))
                else:
                    pal.append(tuple((a+b)//2 for a,b in zip(p0,p1))+(255,))
                    pal.append((0,0,0,0))
                bx=tx+(sb&1)*4; by=ty+(sb>>1)*4
                for y in range(4):
                    row=idx[y]
                    for x in range(4):
                        img[by+y,bx+x]=pal[(row>>(6-2*x))&3]
    return img[:h,:w]
