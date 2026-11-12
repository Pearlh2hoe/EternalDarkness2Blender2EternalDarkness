"""Eternal Darkness (GameCube) NPC model resource -> mesh/skeleton/skin/textures.

Resource = slot payload of a decoded .gpk.  All offsets in the header are relative to (resource + 4).
Header (relative to B = resource+4), big-endian:
  +0x04 u32 hash          +0x08 u16 n_bones        +0x0A u16 n_infl_sets   +0x0E u16 n_vrange
  +0x10 u16 n_vert        +0x16 u16 n_prim         +0x18 u16 n_corner      +0x1A u16 n_uv   +0x1C u16 n_infl
  +0x24 bones   (n_bones x 80 B)      +0x28 ?          +0x2C infl sets (n_infl_sets x (u16 count,u16 first))
  +0x30 vranges (n_vrange x (u16 set,u16 count,u16 start))
  +0x34/38/3C 3 x (n_vert x s16[3]) unit vectors (6-bit fixed) -- meaning unresolved (NOT per-vertex normals in world/bone space)
  +0x40 prims   (n_prim x (u16 count|group<<8,u16 start))  = GX-style triangle stripS over the corner list (alternate winding)
  +0x44 corners (n_corner x (u16 vertex,u16 uv))
  +0x48 uvs     (n_uv x s16[2])  u=U/512, v=V/512 (V down from top)
  +0x4C infl    (n_infl x (f32 weight,u16 bone,u16 pad))
  +0x50 pos     (n_vert x s16[3])  value/16, Z up
  +0x68 texture block: u32 ?, u32 n_tex, u32 list_off(0xC), then list of (u32 entry_off,u32 0); entry (36 B):
        u32 (height<<16|width), u32 gx_format (14 = CMPR), u32 data_off (rel. to block), ...
Bone (80 B): 3 floats (scale), 12 floats = 3x4 row-major INVERSE-BIND matrix (world->bone), s32 parent, u32 name_hash, ...
"""
import struct
import numpy as np
try:
    from .gxtex import decode_cmpr
except ImportError:
    from gxtex import decode_cmpr

POS_SCALE = 16.0
UV_SCALE = 512.0

class ModelError(Exception): pass

class NpcModel:
    def __init__(self, res):
        if len(res) < 0x70 or res[4:8] != b'\x00\xfc\x00\x2e':
            raise ModelError("not a model resource")
        B = self.B = res[4:]
        u32 = lambda o: struct.unpack('>I', B[o:o+4])[0]
        u16 = lambda o: struct.unpack('>H', B[o:o+2])[0]
        self.hash = u32(4)
        nb, nset, nrng, nv = u16(8), u16(0xA), u16(0xE), u16(0x10)
        npr, nco, nuv, nin = u16(0x16), u16(0x18), u16(0x1A), u16(0x1C)
        if nv == 0 or npr == 0: raise ModelError("no geometry")
        def arr(off, n, fmt, size):
            if off + n*np.dtype(fmt).itemsize > len(B): raise ModelError("table out of range")
            return np.frombuffer(B, dtype=fmt, count=n, offset=off)
        self.pos = arr(u32(0x50), nv*3, '>i2', 6).reshape(-1, 3).astype(np.float32) / POS_SCALE
        self.vec = [arr(u32(o), nv*3, '>i2', 6).reshape(-1, 3).astype(np.float32) / 64 for o in (0x34, 0x38, 0x3C)]
        self.corner = arr(u32(0x44), nco*2, '>u2', 4).reshape(-1, 2)
        self.prim = arr(u32(0x40), npr*2, '>u2', 4).reshape(-1, 2)
        self.uv = arr(u32(0x48), nuv*2, '>i2', 4).reshape(-1, 2).astype(np.float32) / UV_SCALE
        if self.corner[:, 0].max() >= nv or self.corner[:, 1].max() >= nuv: raise ModelError("corner index out of range")
        self.prim_group = (self.prim[:, 0] >> 8).astype(np.int32)      # high byte = submesh / material group
        self.prim_count = (self.prim[:, 0] & 0xFF).astype(np.int32)
        if int((self.prim_count + self.prim[:, 1].astype(int)).max()) > nco: raise ModelError("prim out of range")
        # ---- skeleton
        bo = u32(0x24); self.bones = []
        for i in range(nb):
            f = struct.unpack('>15f', B[bo+i*80: bo+i*80+60])
            par = struct.unpack('>i', B[bo+i*80+60: bo+i*80+64])[0]
            M = np.array(f[3:15], np.float64).reshape(3, 4)
            self.bones.append(dict(inv_bind=M, parent=par, name_hash=u32(bo+i*80+64)))
        # ---- skin: per-vertex list of (bone, weight)
        sets = arr(u32(0x2C), nset*2, '>u2', 4).reshape(-1, 2)
        rng = arr(u32(0x30), nrng*3, '>u2', 6).reshape(-1, 3)
        io = u32(0x4C)
        infl = [struct.unpack('>fHH', B[io+i*8: io+i*8+8]) for i in range(nin)]
        skin = [[] for _ in range(nv)]
        # mesh descriptors (276 B each at +0x28): +0x0A vertex base, +0x10 n_vranges, +0x12 first vrange.
        # later ranges overwrite earlier ones (the game's kernel writes sequentially; count==3 windows overlap).
        nmesh = u16(0x0C); dbase = u32(0x28)
        for k in range(max(nmesh, 1)):
            d = dbase + k*276
            if nmesh and d + 0x14 <= len(B): vb, nr, first = u16(d+0x0A), u16(d+0x10), u16(d+0x12)
            else: vb, nr, first = 0, nrng, 0
            for sid, cnt, start in rng[first:first+nr]:
                n, f = sets[sid]
                for v in range(int(start)+vb, min(int(start)+vb+int(cnt), nv)):
                    skin[v] = [(infl[f+j][1], infl[f+j][0]) for j in range(n)]
        # ---- vertex order: pos/skin/vec arrays are in *source* order; the corner list indexes *destination*
        # slots.  A u16 table (byte offsets = slot*6) at header+0xB0 gives dest slot per source vertex.
        lo = u32(0xB0); dest = None
        if 0 < lo and lo + nv*2 <= len(B):
            L = np.frombuffer(B, '>u2', count=nv, offset=lo).astype(np.int64)
            if (L % 6 == 0).all() and np.array_equal(np.sort(L // 6), np.arange(nv)): dest = L // 6
        if dest is None: dest = np.arange(nv)
        self.dest = dest
        def scatter(a):
            out = np.empty_like(a); out[dest] = a; return out
        self.pos = scatter(self.pos); self.vec = [scatter(v) for v in self.vec]
        sk = [None] * nv
        for v in range(nv): sk[dest[v]] = skin[v]
        self.skin = sk
        # ---- position fixed-point format.  Most models store 4 fractional bits (/16); a few (e.g. NPCCom 79/80)
        # are stored with 8 (/256).  Detect by checking which scale makes rigidly-skinned vertices lie on their
        # bone axis (+X) in bone space.
        self.pos_div = POS_SCALE
        try:
            rig = [(i, s[0][0]) for i, s in enumerate(self.skin) if s and len(s) == 1]
            if len(self.bones) > 1 and len(rig) >= 8:
                idx = np.array([i for i, b in rig]); bs = np.array([b for i, b in rig])
                inv = np.array([b['inv_bind'] for b in self.bones])
                def res(sc):
                    Q = np.c_[self.pos[idx] * sc, np.ones(len(idx))]
                    loc = np.einsum('nij,nj->ni', inv[bs], Q)
                    return float(np.median(np.hypot(loc[:, 1], loc[:, 2])))
                r1, r2 = res(1.0), res(1.0 / 16)
                Hh = np.array([w[:3, 3] for w in self.bone_world()])
                big = float(np.ptp(self.pos, axis=0).max()) / max(float(np.ptp(Hh, axis=0).max()), 1e-6) > 8
                if big and r2 * 3 < r1:
                    self.pos = self.pos / 16.0; self.vec = self.vec; self.pos_div = POS_SCALE * 16
        except Exception:
            pass
        # ---- morph targets (face blend shapes).  Mesh descriptor +0x14: 32 x (first record, count) per channel id
        # (first == 0xFFFFFFFF -> unused).  Records (12 B each, table at header+0xDC): u32 vertex (relative to the mesh's
        # vertex base, source order), s16 dx,dy,dz (same fixed point as positions), u16 pad.  The game adds weight*delta
        # to the rest positions.  self.morphs: channel id -> (dest vertex array, delta array in game units)
        self.morphs = {}
        try:
            tbl = u32(0xDC)
            if nmesh and tbl and 0xDC + 4 <= len(B):
                acc = {}
                for k in range(nmesh):
                    d = dbase + k*276
                    if d + 276 > len(B): break
                    vb = u16(d+0x0A)
                    for ch in range(32):
                        first, cnt = u32(d+0x14+8*ch), u32(d+0x18+8*ch)
                        if first == 0xFFFFFFFF or cnt == 0 or tbl + (first+cnt)*12 > len(B): continue
                        r = np.frombuffer(B, np.dtype([('v', '>u4'), ('d', '>i2', 3), ('p', '>u2')]), count=cnt, offset=tbl+first*12)
                        src = r['v'].astype(np.int64) + vb
                        ok = src < nv
                        acc.setdefault(ch, []).append((dest[src[ok]], r['d'][ok].astype(np.float32) / self.pos_div))
                for ch, parts in acc.items():
                    self.morphs[ch] = (np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts]))
        except Exception:
            self.morphs = {}
        # ---- triangles (strips, alternating winding) -> list of corner-index triples
        tris = []; tgrp = []
        for c, s, g in zip(self.prim_count, self.prim[:, 1], self.prim_group):
            c = int(c); s = int(s)
            for i in range(c-2):
                a, b, d = s+i, s+i+1, s+i+2
                tris.append((a, b, d) if i % 2 == 0 else (b, a, d)); tgrp.append(int(g))
        self.tri_corners = np.array(tris, np.int32).reshape(-1, 3)
        self.tri_group = np.array(tgrp, np.int32)
        self.tri_verts = self.corner[self.tri_corners][..., 0].astype(np.int32)
        # ---- textures
        self.textures = []
        to = u32(0x68)
        if to and to + 12 <= len(B):
            try:
                ntex = u32(to+4); lo = u32(to+8)
                for t in range(ntex):
                    eo = u32(to + lo + t*8)
                    h_w, fmt, doff = struct.unpack('>III', B[to+eo: to+eo+12])
                    w, h = h_w & 0xFFFF, h_w >> 16
                    if fmt == 14 and 0 < w <= 1024 and 0 < h <= 1024:
                        n = ((w+7)//8*8) * ((h+7)//8*8) // 2
                        if to+doff+n <= len(B):
                            self.textures.append(decode_cmpr(B[to+doff: to+doff+n], w, h))
                        else: self.textures.append(None)
                    else: self.textures.append(None)
            except Exception: self.textures = []

        # ---- materials: u32 count at +0x5C, pointer at +0x60, 32-byte entries; entry g (= prim group g) starts with
        # the u16 diffuse texture index (0xFFFF = none); the u16 at +0x0E is a secondary (detail/gloss) map.
        self.materials = []
        try:
            nmat, mo = u32(0x5C), u32(0x60)
            if 0 < nmat <= 64 and mo + nmat*32 <= len(B):
                for g in range(nmat):
                    e = mo + g*32
                    d, sec = u16(e), u16(e + 0x0E)
                    self.materials.append(dict(diffuse=d if d < len(self.textures) else None,
                                               secondary=sec if sec < len(self.textures) else None))
        except Exception:
            self.materials = []

    def diffuse_for_group(self, g):
        """texture index for prim group g (falls back to the first decodable texture)"""
        if 0 <= g < len(self.materials) and self.materials[g]['diffuse'] is not None \
                and self.textures[self.materials[g]['diffuse']] is not None:
            return self.materials[g]['diffuse']
        for i, t in enumerate(self.textures):
            if t is not None: return i
        return None

    def bone_world(self):
        """bind-pose world matrices (4x4) = inverse of the stored inverse-bind."""
        out = []
        for b in self.bones:
            M = np.eye(4); M[:3, :] = b['inv_bind']; out.append(np.linalg.inv(M))
        return out

    def stats(self):
        t = self.tri_verts
        E = {}
        for a, b, c in t:
            for e in ((a, b), (b, c), (c, a)):
                k = (min(e), max(e)); E[k] = E.get(k, 0) + 1
        two = sum(1 for v in E.values() if v == 2) / max(len(E), 1)
        return dict(verts=len(self.pos), tris=len(t), bones=len(self.bones), textures=len(self.textures), edge2=round(two, 3))
