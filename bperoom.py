"""Eternal Darkness (GameCube) room file (rmedNNNN.bpe) parser.  No Blender dependency.

File layout
  0x00 u32 nparts (2)   0x04 u32 0x40 (data start)   0x08 u32 / 0x0C u32 compressed offsets / total
  0x10 u32 size of part 0 (decoded)   0x14 u32 size of part 1 (decoded)
  0x40 ...  Gage BPE stream (multi block); the decoded stream = part0 ++ part1
Part 0 = the room ("level") data, a flat byte stream that the game walks with Parse_*_From_ByteStream:
  0x000  0x120-byte file header (byte 0 = version 0x17), everything else in it is overwritten
  then   Surface header (12 B, version 0x0C) + N x 32 B surfaces
         Atx header (16 B, version 0x10) + animated texture (.IFL) entries (272 B)
         Room header (128 B, version 0x0A) + pools and lists:
              A  s16[3]  positions                         (hdr+0x08 = count)
              B  f32[3]  normals                           (hdr+0x10 = count)
              C  s16[2]  UVs, /512 (diffuse, lightmap 1, lightmap 2 all live in this pool)
              E  convex polygons, 20 B each (count hdr+0x2C):
                     u16 ncorners, u16 0, u32 bytes (=12*n), u32 0, u32 surface, u32 0
                     + per polygon a blob of n x 6 u16 corners (a triangle strip of n corners, alternate winding): pos, normal, uv, lightmap1 uv, lightmap2 uv, 0
              G  11158 x 12 B lists (not needed for rendering)
              M  collision / structural polygons, 56 B each (count hdr+0x44)
              P  spatial grid (33 x 33)
         Level header (120 B, version 0x0E/0x0F) with the objects
              A  cameras (120 B)       C  regions (116 B, AABB)       0x38 markers (76 B)
Surface (16 x u16): [0] diffuse texture, [1] lightmap 1, [6] lightmap 2 (0xFFFF = none), [4]/[5] extra maps.
Part 1 = texture bank: u32 total, u32 count, count x (u32,u32 offset); record (36 B): u16 HEIGHT, u16 WIDTH, u32 fmt (14 = CMPR),
  u32 data offset, ...  Lightmaps are clamped (wrap flag 0), diffuse maps repeat (wrap flag 1).
"""
import struct
import numpy as np

try:
    from .bpe_safe import bpe_expand_safe
except ImportError:
    from bpe_safe import bpe_expand_safe


class RoomError(Exception):
    pass


# --- textures ---
def decode_cmpr_fast(data, w, h):
    """GX_TF_CMPR -> RGBA uint8 (h, w, 4), vectorised."""
    tx, ty = (w + 7) // 8, (h + 7) // 8
    need = tx * ty * 32
    buf = np.frombuffer(data[:need].ljust(need, b'\0'), np.uint8).reshape(ty, tx, 4, 8).astype(np.int32)
    c0 = (buf[..., 0] << 8) | buf[..., 1]
    c1 = (buf[..., 2] << 8) | buf[..., 3]

    def rgb(c):
        r = (c >> 11) & 31; g = (c >> 5) & 63; b = c & 31
        return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1)
    p0, p1 = rgb(c0), rgb(c1)
    four = (c0 > c1)[..., None]
    p2 = np.where(four, (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(four, (p0 + 2 * p1) // 3, 0)
    a3 = np.where(four[..., 0], 255, 0)
    pal = np.stack([p0, p1, p2, p3], -2)                                   # (ty,tx,4sub,4,3)
    a3 = np.where(four[..., 0], 255, 0)                                    # (ty,tx,4sub)
    alpha = np.stack([np.full_like(a3, 255)] * 3 + [a3], -1)               # (ty,tx,4sub,4)
    sh = np.array([6, 4, 2, 0], np.int32)
    idx = (buf[..., 4:8][..., :, None] >> sh) & 3                          # (ty,tx,4sub,4row,4col)
    ty_, tx_ = idx.shape[:2]
    flat = idx.reshape(ty_, tx_, 4, 16)
    rgbv = np.take_along_axis(pal, flat[..., None].repeat(3, -1), axis=3)  # (ty,tx,4sub,16,3)
    alv = np.take_along_axis(alpha, flat, axis=3)                          # (ty,tx,4sub,16)
    out = np.concatenate([rgbv, alv[..., None]], -1).astype(np.uint8)
    out = out.reshape(ty_, tx_, 2, 2, 4, 4, 4)                             # sub=(by,bx), row, col
    out = out.transpose(0, 2, 4, 1, 3, 5, 6).reshape(ty_ * 8, tx_ * 8, 4)  # (ty,by,row,tx,bx,col)
    return out[:h, :w]


def _decode_texture(data, w, h, fmt):
    if fmt == 14:
        return decode_cmpr_fast(data, w, h)
    raise RoomError("texture format %d is not supported" % fmt)


# --- stream walker (mirrors the game's loader) ---
class _Walker:
    def __init__(self, S):
        self.S = S

    def u32(self, o): return struct.unpack_from('>I', self.S, o)[0]
    def u16(self, o): return struct.unpack_from('>H', self.S, o)[0]
    def i16(self, o): return struct.unpack_from('>h', self.S, o)[0]
    def f32(self, o): return struct.unpack_from('>f', self.S, o)[0]


def _align4(x): return (x + 3) & ~3


def parse_part0(S):
    w = _Walker(S)
    if S[0] != 0x17:
        raise RoomError("not a room stream (version byte %02X, expected 17)" % S[0])
    R = {}
    # ---- surfaces
    p = 0x120
    if w.u16(p) != 0x0C: raise RoomError("bad surface header")
    nsurf, ssz = w.u32(p + 4), w.u32(p + 8)
    R['surf_off'] = p + _align4(12); R['nsurf'] = nsurf
    p = R['surf_off'] + ssz
    # ---- animated textures
    if w.u16(p) != 0x10: raise RoomError("bad atx header")
    ncnt, asz, n12, sz12 = w.u16(p + 2), w.u32(p + 4), w.u16(p + 8), w.u32(p + 12)
    cur = p + _align4(16); a0 = cur; cur += asz
    R['ifl'] = []; R['ifl_frames'] = []
    for i in range(ncnt):
        e = a0 + i * 272
        R['ifl'].append(bytes(S[e + 2:e + 36]).split(b'\0')[0].decode('latin1'))
        nfr = w.u32(e + 256); s0 = w.u32(e + 260)          # blob 0 = u16 frame texture ids, blob 1 = frame durations
        R['ifl_frames'].append([w.u16(cur + 2 * k) for k in range(min(nfr, s0 // 2))])
        for k in (260, 264, 268): cur += w.u32(e + k)
    cur += sz12
    # ---- room header
    h = cur
    if S[h] != 0x0A: raise RoomError("bad room header (version %02X)" % S[h])
    c = 128
    def blk(fo):
        nonlocal c
        s = w.u32(h + fo)
        o = h + c; c += s
        return o, s
    R['nA'] = w.u16(h + 0x08); R['nB'] = w.u32(h + 0x10); R['nC'] = w.u32(h + 0x18)
    R['A'], _ = blk(0x0C); R['B'], _ = blk(0x14); R['C'], _ = blk(0x1C); blk(0x28)
    nE = w.u32(h + 0x2C); E, _ = blk(0x30); R['nE'] = nE; R['E'] = E
    for i in range(nE): c += w.u32(E + i * 20 + 4)
    nF = w.u16(h + 0x34); F, _ = blk(0x38)
    for i in range(nF): c += w.u32(F + i * 20 + 16)
    nG = w.u16(h + 0x3C); G, _ = blk(0x40)
    for i in range(nG): c += w.u32(G + i * 12 + 4)
    # corner blobs follow, in polygon order, directly at the E-blob cursor: they were laid out right after the E array
    # (the loop above already advanced c past the per-E blobs, so remember where they start)
    R['corner_start'] = E + w.u32(h + 0x30)
    R['nM'] = w.u16(h + 0x44)
    R['mstart'] = h + c
    return w, R, h


def parse(raw):
    """raw = whole .bpe file.  Returns a dict describing the room."""
    if len(raw) < 0x48: raise RoomError("file too small")
    nparts, start = struct.unpack_from('>II', raw, 0)
    if nparts != 2 or start != 0x40:
        raise RoomError("not a room .bpe (header %08X %08X)" % (nparts, start))
    size0, size1 = struct.unpack_from('>II', raw, 0x10)
    dec, used, status = bpe_expand_safe(raw[0x40:], 1 << 28)
    if status != 'ok' and len(dec) < size0 + size1:
        raise RoomError("BPE decode failed (%s)" % status)
    S0, S1 = dec[:size0], dec[size0:size0 + size1]
    return _build(S0, S1)


def _build(S0, S1):
    w, R, h = parse_part0(S0)
    S = S0
    out = {}
    # ---- textures
    n = struct.unpack_from('>I', S1, 4)[0]
    tex = []
    for i in range(n):
        off = struct.unpack_from('>I', S1, 8 + i * 8 + 4)[0]
        th, tw, fmt, doff = struct.unpack_from('>HHII', S1, off)     # high u16 = height, low u16 = width
        wrap_s, wrap_t = struct.unpack_from('>II', S1, off + 20)
        nmip = struct.unpack_from('>I', S1, off + 28)[0]
        cw_s, cw_t = struct.unpack_from('>II', S1, off + 12)          # words 3,4: 1 = repeat (diffuse), 0 = clamp (lightmap class)
        tex.append(dict(w=tw, h=th, fmt=fmt, off=doff, wrap=(wrap_s, wrap_t), clamp=(cw_s == 0 and cw_t == 0)))
    out['tex_info'] = tex
    out['S1'] = S1
    # ---- surfaces: (diffuse, lm1, lm2)
    sf = np.frombuffer(S, '>u2', count=R['nsurf'] * 16, offset=R['surf_off']).reshape(-1, 16)
    out['surfaces'] = [dict(diffuse_missing=bool(r[0] == 0xFFFF), diffuse=int(r[0]) if r[0] < n else None, lm1=int(r[1]) if r[1] < n else None,
                            lm2=int(r[6]) if r[6] < n else None,
                            extra=[int(x) for x in r[4:6] if x < n]) for r in sf]
    # ---- pools
    out['pos'] = np.frombuffer(S, '>i2', count=R['nA'] * 3, offset=R['A']).reshape(-1, 3).astype(np.float32)
    out['nrm'] = np.frombuffer(S, '>f4', count=R['nB'] * 3, offset=R['B']).reshape(-1, 3).astype(np.float32)
    out['uv'] = np.frombuffer(S, '>i2', count=R['nC'] * 2, offset=R['C']).reshape(-1, 2).astype(np.float32) / 512.0
    # ---- polygons
    E, nE = R['E'], R['nE']
    ents = np.frombuffer(S, '>u4', count=nE * 5, offset=E).reshape(nE, 5)
    ncorn = (ents[:, 0] >> 16).astype(np.int64)
    surf = ents[:, 3].astype(np.int64)
    pos0 = R['corner_start']
    sizes = ents[:, 1].astype(np.int64)
    if not np.array_equal(sizes, ncorn * 12):
        raise RoomError("unexpected polygon blob sizes")
    total = int(sizes.sum())
    corners = np.frombuffer(S, '>u2', count=total // 2, offset=pos0).reshape(-1, 6).astype(np.int64)
    # each polygon record is a triangle strip of n corners (n-2 triangles, alternate winding), not a cyclic polygon
    starts = np.concatenate([[0], np.cumsum(ncorn)])
    ta, tb, tc, ts = [], [], [], []
    for i in range(len(ncorn)):
        base = int(starts[i])
        for k in range(int(ncorn[i]) - 2):
            if k & 1: ta.append(base + k); tb.append(base + k + 2); tc.append(base + k + 1)
            else: ta.append(base + k); tb.append(base + k + 1); tc.append(base + k + 2)
            ts.append(surf[i])
    tris = np.array([ta, tb, tc], np.int64).T.reshape(-1, 3)
    tsurf = np.array(ts, np.int64)
    pos = out['pos'].astype(np.float64); nrm = out['nrm'].astype(np.float64)
    P = pos[corners[tris, 0]]                                          # (T,3,3)
    g = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nn = nrm[np.clip(corners[tris, 1], 0, len(nrm) - 1)].sum(1)
    flip = (g * nn).sum(1) < 0                                         # winding follows the stored normals
    tris[flip] = tris[flip][:, [0, 2, 1]]
    keep = np.linalg.norm(g, axis=1) > 1e-9                            # drop zero-area triangles
    out['tris'] = tris[keep]; out['tri_surf'] = tsurf[keep]; out['flipped'] = int(flip.sum())
    out['poly_n'] = ncorn; out['poly_surf'] = surf; out['corners'] = corners
    # ---- collision / structural polygons (M)
    out['collision'] = _parse_m(w, S, R, h)
    # ---- level header: cameras, regions, markers
    out['objects'] = _parse_level(w, S, R, h, out['collision'][2])
    out['ifl'] = R['ifl']; out['ifl_frames'] = R['ifl_frames']
    return out


def _parse_m(w, S, R, h):
    nM = R['nM']; sz = w.u32(h + 0x48)
    arr = R['mstart']
    p = arr + sz
    polys = []
    sizes = []
    for i in range(nM):
        sizes.append(w.u32(arr + i * 56 + 0x2C))
    blobs = []
    for i in range(nM):
        blobs.append((p, sizes[i])); p += sizes[i]
    for i in range(nM):
        e = arr + i * 56
        n = w.u16(e + 0x28)
        o, s = blobs[i]
        idx = np.frombuffer(S, '>u2', count=n, offset=o) if s else np.zeros(0, np.uint16)
        polys.append((w.u16(e), idx.astype(np.int64)))
    # lists hanging off the polygons
    for i in range(nM):
        e = arr + i * 56
        nl = S[e + 0x30]; nl = nl - 256 if nl > 127 else nl
        ls = w.u32(e + 0x34)
        if ls:
            la = p; p += ls
            for k in range(max(nl, 0)): p += w.u32(la + k * 8 + 4)
    return polys, arr, p


def _parse_room_tail(w, S, R, h, p):
    """advance past the room-header tail structures (P grid) starting at p; returns the level-header offset."""
    nP = w.u16(h + 0x4C); cur = p - h
    if w.u32(h + 0x50):
        cur += 4 * nP
        for i in range(nP):
            q = h + cur; c = _align4(36); n = w.u32(q + 4); sub = w.u32(q + 0x20); ent = q + c
            if sub:
                c += sub
                for k in range(n):
                    for off in (16, 12): c += w.u32(ent + k * 20 + off)
            cur += c
    nQ = w.u16(h + 0x54)
    if w.u32(h + 0x58):
        cur += 4 * nQ
        for i in range(nQ):                     # embedded objects; word +8 = total size of each
            cur += w.u32(h + cur + 8)
    if w.u32(h + 0x5C): cur += 284
    cur += w.u32(h + 0x64)
    return h + cur


def _parse_level(w, S, R, h, mend):
    objs = dict(cameras=[], regions=[], markers=[])
    try:
        p = _parse_room_tail(w, S, R, h, mend)
        if S[p] not in (0x0E, 0x0F): raise RoomError("level header not found (%02X)" % S[p])
        cur = 120
        L = {}
        def take(name, fo):
            nonlocal cur
            s = w.u32(p + fo)
            if s: L[name] = (p + cur, s); cur += s
            return s
        def subs(arr, n, sz, offs, gate=None):
            nonlocal cur
            for i in range(n):
                e = arr + i * sz
                if gate and not gate(e): continue
                for o in offs: cur += w.u32(e + o)
        nA = w.u16(p + 2); take('A', 4)
        if 'A' in L: subs(L['A'][0], nA, 120, (100, 108, 116))
        nB = w.u16(p + 8); take('B', 0xC)
        if 'B' in L: subs(L['B'][0], nB, 120, (100, 108, 116))
        nC = w.u16(p + 0x10); take('C', 0x14)
        if 'C' in L: subs(L['C'][0], nC, 116, (0x34, 0x3C), lambda e: w.i16(e + 108) == 2)
        take('24', 0x24)
        def lst80(fo, n):
            nonlocal cur
            sz = w.u32(p + fo)
            if not sz: return
            arr = p + cur; cur += sz
            for i in range(n): cur += w.u32(arr + i * 56 + 0x2C)
            for i in range(n):
                e = arr + i * 56; nl = S[e + 0x30]; ls = w.u32(e + 0x34)
                if ls:
                    la = p + cur; cur += ls
                    for k in range(nl if nl < 128 else 0): cur += w.u32(la + k * 8 + 4)
        lst80(0x28, w.u16(p + 0x20))
        take('30', 0x30); take('38', 0x38); take('40', 0x40)
        lst80(0x44, w.u16(p + 0x3C))
        # cameras
        for key in ('A',):                 # list B holds unnamed entries of another type
            if key not in L: continue
            n = nA
            for i in range(n):
                e = L[key][0] + i * 120
                pos = [w.i16(e + 0), w.i16(e + 2), w.i16(e + 4)]
                d = [w.f32(e + 8), w.f32(e + 12), w.f32(e + 16)]
                name = bytes(S[e + 0x38:e + 0x58]).split(b'\0')[0].decode('latin1')
                objs['cameras'].append(dict(name=name or "Camera%s%d" % (key, i), pos=pos, dir=d,
                                            far=w.f32(e + 0x20), fov=w.f32(e + 0x24), angle=w.f32(e + 0x14), hash=w.u32(e + 0x34), list=key))
        if 'C' in L:
            for i in range(nC):
                e = L['C'][0] + i * 116
                name = bytes(S[e + 4:e + 36]).split(b'\0')[0].decode('latin1')
                bb = [w.i16(e + 0x30 + k * 2) for k in range(6)]
                objs['regions'].append(dict(name=name, min=bb[0:3], max=bb[3:6], hash=w.u32(e + 0x2C)))
        if '38' in L:
            o, sz = L['38']
            for i in range(sz // 76):
                e = o + i * 76
                name = bytes(S[e:e + 40]).split(b'\0')[0].decode('latin1')
                if not name: continue
                objs['markers'].append(dict(name=name, hash=w.u32(e + 0x28),
                                            pos=[w.i16(e + 0x2C), w.i16(e + 0x2E), w.i16(e + 0x30)],
                                            kind=w.u32(e + 0x34), flags=w.u32(e + 0x48),
                                            axis=w.f32(e + 0x40), angle=w.f32(e + 0x44)))
    except Exception as ex:           # objects are optional: geometry still imports
        objs['error'] = str(ex)
    return objs


# --- helpers for the importer ---
def texture_rgba(room, i):
    t = room['tex_info'][i]
    w_, h_ = t['w'], t['h']
    sz = ((w_ + 7) // 8) * ((h_ + 7) // 8) * 32
    return _decode_texture(room['S1'][t['off']:t['off'] + sz], w_, h_, t['fmt'])
