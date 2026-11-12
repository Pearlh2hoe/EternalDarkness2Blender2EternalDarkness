"""Eternal Darkness character animation packs (chars/charNNNN.cpk).  Pure Python/numpy (no bpy).

Layout (big-endian):
  FILE  +0x08 u32 model hash (== NpcModel.hash of the character the pack belongs to)
        +0x0C u32 alias index (32-byte stub files: use that charNNNN instead)
        +0x10 u32 nparts, +0x14 u32 part table offset
  PART TABLE: nparts x (u32 offset, u32 size, u16 slot_type, u16 ?, u32 ?)   slot_type = clip set / stance
  PART  +0x14 u16 nrec, +0x18 u32 record table offset (records are 0x128 B)
  RECORD +6 u16 clip id (shared between characters), +0x118 payload offset, +0x11C packed size, +0x120 unpacked size
  PAYLOAD = Gage BPE.  Unpacked: u32 0x10, u32 end, u32 table; at `table`: (u16 count, u16 0, u32 offset) per track.
  Tracks alternate per bone: 2*i = position keys (kind 1), 2*i+1 = rotation keys (kind 2); bone i of the model.
  Key = (u16 tick, u16 kind, s16 a, b, c, d); rotation = quaternion xyzw = (a,b,c,d)/16384; position = (a,b,c)/16,
  both relative to the parent bone (root: absolute).  240 ticks per second (one 30 fps frame = 8 ticks).
"""
import struct
import numpy as np
from .bpe_safe import bpe_expand_safe

TICKS_PER_FRAME = 8


class CpkError(Exception):
    pass


def header(d):
    if len(d) < 0x20: raise CpkError("not a character pack (too small)")
    u32 = lambda o: struct.unpack('>I', d[o:o+4])[0]
    if len(d) == 0x20 and u32(0x10) == 0:
        return dict(hash=0, alias=u32(0x0C), nparts=0)
    return dict(hash=u32(8), alias=None, nparts=u32(0x10))


def load(d, only_set=None):
    """Return dict(hash, alias, sets=[(slot_type, [(clip_id, tracks)])]).  only_set limits the sets that get unpacked."""
    h = header(d)
    if h['alias'] is not None: return dict(hash=0, alias=h['alias'], sets=[])
    u32 = lambda o: struct.unpack('>I', d[o:o+4])[0]
    u16 = lambda o: struct.unpack('>H', d[o:o+2])[0]
    to = u32(0x14); sets = []
    for i in range(h['nparts']):
        e = to + i * 16; off = u32(e); typ = u16(e + 8)
        if only_set is not None and only_set >= 0 and typ != only_set: continue
        nrec, ro = u16(off + 0x14), u32(off + 0x18); clips = []
        for k in range(nrec):
            r = off + ro + k * 0x128
            po, cs = u32(r + 0x118), u32(r + 0x11C)
            try:
                data, used, _st = bpe_expand_safe(d[off + po: off + po + cs])
                tr = _tracks(data)
            except Exception:
                continue
            if tr: clips.append((u16(r + 6), tr))
        sets.append((typ, clips))
    return dict(hash=h['hash'], alias=None, sets=sets)


def _tracks(b):
    _, end, start = struct.unpack('>III', b[:12])
    out = []
    for i in range((end - start) // 8):
        c, _, off = struct.unpack('>HHI', b[start + i * 8: start + i * 8 + 8])
        if c == 0 or off + c * 12 > len(b): out.append(None); continue
        a = np.frombuffer(b, np.dtype([('t', '>u2'), ('k', '>u2'), ('v', '>i2', 4)]), count=c, offset=off)
        out.append((a['t'].astype(np.float64), a['v'].astype(np.float64)))
    return out


def clip_info(tracks):
    """(first tick, last tick, number of frames at 30 fps)"""
    t0 = min(t[0][0] for t in tracks if t is not None); t1 = max(t[0][-1] for t in tracks if t is not None)
    return t0, t1, int(round((t1 - t0) / TICKS_PER_FRAME)) + 1


def _is_const(v):
    return len(v) < 2 or bool(np.all(v == v[0]))


def sample(tracks, nbones, step=1):
    """Sample a clip at 30 fps.  Returns (frames, quat[f, bone, 4] xyzw, pos[f, bone, 3] in game units, const[bone])
    where const[bone] is True when neither the rotation nor the position of that bone ever changes."""
    t0, t1, nf = clip_info(tracks)
    fr = np.arange(0, nf, max(1, step));
    if fr[-1] != nf - 1: fr = np.append(fr, nf - 1)
    ticks = t0 + fr * float(TICKS_PER_FRAME)
    Q = np.zeros((len(fr), nbones, 4)); Q[..., 3] = 1.0; P = np.zeros((len(fr), nbones, 3)); const = np.ones(nbones, bool)
    for i in range(nbones):
        pt = tracks[2 * i] if 2 * i < len(tracks) else None
        rt = tracks[2 * i + 1] if 2 * i + 1 < len(tracks) else None
        if pt is not None:
            t, v = pt
            for c in range(3): P[:, i, c] = np.interp(ticks, t, v[:, c] / 16.0)
            const[i] &= _is_const(v[:, :3])
        if rt is not None:
            t, v = rt; q = v / 16384.0
            if len(t) == 1: Q[:, i] = q[0]
            else:
                idx = np.clip(np.searchsorted(t, ticks, side='right') - 1, 0, len(t) - 2)
                span = t[idx + 1] - t[idx]; w = np.clip((ticks - t[idx]) / np.where(span == 0, 1, span), 0, 1)[:, None]
                a, b = q[idx], q[idx + 1]
                b = np.where((a * b).sum(1, keepdims=True) < 0, -b, b)          # shortest arc
                m = a * (1 - w) + b * w
                Q[:, i] = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
            const[i] &= _is_const(v)
    return fr, Q, P, const
