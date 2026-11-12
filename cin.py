"""Eternal Darkness cinematic frame stream (cinNNNN, the extension-less file next to cinNNNN.bin).

Layout (checked on cin0002):
  file (after SK_ASC):  +0 u32 size, +4 u32 hdr_size, ... , +0x38 u32 nframes, +0x4C u32 frame_size[nframes],
                        frames start at hdr_size, each frame padded to 32 bytes.
  frame: u32 (unused), 04 03 02 01, then Gage-BPE blocks (bpe_safe) expanding to the frame structure.
  frame structure (offsets relative to frame start):
    0x00 camera/scene floats ... 0x50 f32 far (1000) 0x54 f32 fov 0x58 f32 ...
    0x5C u32 entity table, 0x70 u32 list of object ids, 0x7C u8 entity count, 0x6C u32 effects block
    entity record (32 B): f32 x,y,z (world root pos), u8 nA, u8 nB, u8 flags, u8 nBones, u32 A, u32 B, u32 C, u32 D
      A: nA x (u32 id, f32 value)       B: nB x (s16 value, u8, u8)  (face/texture channel updates)
      C: nBones x 16 B: s16 qx,qy,qz,qw (/16384), s16 px,py,pz (/16 = parent-local offset), u8 flag, u8 bone index
      D: bit mask over bones (bone>>3, 1<<(bone&7)); 0 = all bones present.  Frames update only present bones.
"""
import struct
import numpy as np
try: from .bpe_safe import bpe_expand_safe
except ImportError: from bpe_safe import bpe_expand_safe

def frame_table(d):
    n = struct.unpack('>I', d[0x38:0x3C])[0]
    hdr = struct.unpack('>I', d[4:8])[0]
    sizes = np.frombuffer(d[0x4C:0x4C + 4 * n], '>u4').astype(int)
    starts = []; off = hdr
    for s in sizes: starts.append(off); off += (int(s) + 31) & ~31
    return starts, sizes

def decode_frame(d, start, size):
    fr = d[start:start + size]
    if fr[4:8] != b'\x04\x03\x02\x01': raise ValueError("bad frame marker")
    o, p, st = bpe_expand_safe(fr[8:])
    return o

def parse_frame(o):
    u32 = lambda a: struct.unpack('>I', o[a:a+4])[0]
    cam = struct.unpack('>20f', o[0:80])
    ent_off, lst_off, n = u32(0x5C), u32(0x70), o[0x7C]
    ents = []
    for k in range(n):
        e = ent_off + 32 * k
        x, y, z = struct.unpack('>3f', o[e:e+12])
        nA, nB, flags, nBones = o[e+12], o[e+13], o[e+14], o[e+15]
        A, B, C, D = struct.unpack('>4I', o[e+16:e+32])
        obj = u32(lst_off + 4 * k)
        mask = None if D == 0 else o[D:D + (nBones + 7) // 8 + 8]
        bones = {}
        for r in range(nBones):
            q = struct.unpack('>4h3h', o[C + 16*r: C + 16*r + 14])
            fl, bid = o[C + 16*r + 14], o[C + 16*r + 15]
            bones[bid] = (np.array(q[:4]) / 16384.0, np.array(q[4:7]) / 16.0, fl)
        ents.append(dict(obj=obj, pos=(x, y, z), nA=nA, nB=nB, flags=flags, bones=bones,
                         A=[struct.unpack('>If', o[A+8*i:A+8*i+8]) for i in range(nA)] if A else [],
                         B=[struct.unpack('>hBB', o[B+4*i:B+4*i+4]) for i in range(nB)] if B else []))
    return dict(cam=cam, fov=struct.unpack('>f', o[0x54:0x58])[0], ents=ents)
