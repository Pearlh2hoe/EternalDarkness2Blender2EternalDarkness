"""Rebuild an Eternal Darkness model resource from edited geometry (pure Python/numpy, no bpy).

Strategy: every vertex-dependent table (scatter table +0xB0, per-vertex table +0xB4, positions +0x50, the three vector
arrays +0x34/38/3C, UVs +0x48, corners +0x44, prims +0x40, influence sets/ranges/weights +0x2C/+0x30/+0x4C, the morph
records +0xDC and, when the material count grows, the material table +0x60) is written as NEW data appended to the end of
the resource and the header pointers/counts are updated; the old tables stay as dead bytes.  Nothing else moves, so
every section that is not understood (bones, texture block, hashes, extra tables ...) is kept byte-for-byte.
The 276-byte mesh descriptors (header +0x28) are patched in place: each describes a contiguous run of prims, UVs and
vertices (+0 prim count/+2 prim start, +4 uv count/+6 uv start, +8 vertex count/+0xA vertex start, +0x10/+0x12 influence
ranges, +0x14.. 32 x (first, count) morph records).  Faces keep the descriptor of the original vertices they use.
"""
import struct
from collections import Counter, defaultdict
import numpy as np


class WriteError(Exception):
    pass


def geo_from_model(m):
    """Geometry description of an unmodified model (used for the identity test and as a reference of the format)."""
    keep = [i for i, f in enumerate(m.tri_verts) if len(set(int(x) for x in f)) == 3]
    tc = m.tri_corners[keep]
    tuv = m.uv[m.corner[tc][..., 1]]
    return dict(pos=m.pos.astype(np.float64), nrm=m.vec[0].astype(np.float64), vid=np.arange(len(m.pos)),
                skin=None, tris=m.tri_verts[keep], tuv=tuv.astype(np.float64), tgrp=m.tri_group[keep], morph=None, mats=None)


def material_capacity(res):
    """How many 32-byte material entries fit before the texture block (the table cannot be moved)."""
    B = res[4:]; mo, to = struct.unpack('>I', B[0x60:0x64])[0], struct.unpack('>I', B[0x68:0x6C])[0]
    return max((to - mo) // 32, struct.unpack('>I', B[0x5C:0x60])[0]) if to > mo else struct.unpack('>I', B[0x5C:0x60])[0]


def stripify(tris, maxlen=255, seeds=()):
    """tris: list of 3-tuples of hashable corner keys (orientation = winding).  Returns a list of strips (lists of corner
    keys) in the engine's convention: triangle i of a strip is (s[i], s[i+1], s[i+2]) for even i and (s[i+1], s[i], s[i+2]) for odd i."""
    n = len(tris)
    edges = defaultdict(list)
    for i, t in enumerate(tris):
        for k in range(3): edges[(t[k], t[(k + 1) % 3])].append(i)
    used = [False] * n; strips = []
    canon = defaultdict(list)
    def ck_(t):
        r = min(range(3), key=lambda i: repr(t[i:] + t[:i])); return t[r:] + t[:r]
    for i, t in enumerate(tris): canon[ck_(tuple(t))].append(i)
    # seeds: strips of the ORIGINAL model; kept whole when every triangle of them still exists (unedited regions keep the
    # original, usually better packed, strips)
    for sd in seeds:
        ids = []; ok = len(sd) >= 3
        for j in range(len(sd) - 2):
            t = (sd[j], sd[j + 1], sd[j + 2]) if j % 2 == 0 else (sd[j + 1], sd[j], sd[j + 2])
            lst = [x for x in canon.get(ck_(t), ()) if not used[x] and x not in ids]
            if len(set(t)) < 3 or not lst: 
                if len(set(t)) < 3: continue          # degenerate (restart) triangle inside the original strip: nothing to match
                ok = False; break
            ids.append(lst[0])
        if ok and ids:
            for x in ids: used[x] = True
            strips.append(list(sd))

    def grow(start, rot, taken):
        t = tris[start]; s = [t[rot], t[(rot + 1) % 3], t[(rot + 2) % 3]]; chain = [start]; taken.add(start)
        while len(s) < maxlen:
            j = len(s) - 2
            p, q = (s[-2], s[-1]) if j % 2 == 0 else (s[-1], s[-2])
            nxt = None
            for c in edges.get((p, q), ()):
                if not used[c] and c not in taken:
                    nxt = c; break
            if nxt is None: break
            tt = tris[nxt]; r = [x for x in tt if x != p and x != q]
            if len(r) != 1: break
            # the apex must complete the oriented triangle (p, q, r) -> make sure of the winding
            k = tt.index(p) if p in tt else -1
            if k < 0 or tt[(k + 1) % 3] != q: break
            s.append(r[0]); chain.append(nxt); taken.add(nxt)
        return s, chain

    # start from triangles with the fewest unused neighbours (strip ends / corners) - gives noticeably longer strips
    def free_nb(i):
        t = tris[i]; c = 0
        for k in range(3):
            for j in edges.get((t[(k + 1) % 3], t[k]), ()):
                if j != i and not used[j]: c += 1
        return c
    import heapq
    heap = [(free_nb(i), i) for i in range(n)]; heapq.heapify(heap)
    while heap:
        d, i = heapq.heappop(heap)
        if used[i]: continue
        nd = free_nb(i)
        if nd != d: heapq.heappush(heap, (nd, i)); continue          # stale priority
        best = None
        for rot in range(3):
            taken = set(); s, chain = grow(i, rot, taken)
            if best is None or len(s) > len(best[0]): best = (s, chain)
        s, chain = best
        for c in chain: used[c] = True
        strips.append(s)
    return strips


def rebuild(res, m, geo):
    """res: original model resource bytes (4-byte prefix + body), m: edmodel.NpcModel of it, geo: see geo_from_model
    (+ optional 'skin': per vertex [(bone, weight)], 'morph': {channel: (n,3) game-unit deltas}, 'mats': {group: dict}).
    Returns (new_resource_bytes, [report lines])."""
    rep = []
    B = bytearray(res[4:]); L0 = len(B); B0 = bytes(B)
    u32 = lambda o: struct.unpack('>I', B[o:o + 4])[0]
    u16 = lambda o: struct.unpack('>H', B[o:o + 2])[0]
    nv0 = len(m.pos); nmesh = u16(0xC); dbase = u32(0x28)
    if nmesh < 1 or dbase + nmesh * 276 > L0: raise WriteError("model has no mesh descriptors: geometry export is not supported for it")
    div = float(m.pos_div)
    pos = np.asarray(geo['pos'], np.float64); n = len(pos)
    nrm = geo.get('nrm'); vid = np.asarray(geo['vid'], np.int64)
    tris = np.asarray(geo['tris'], np.int64).reshape(-1, 3); tuv = np.asarray(geo['tuv'], np.float64).reshape(-1, 3, 2)
    tgrp = np.asarray(geo['tgrp'], np.int64)
    if len(tris) == 0: raise WriteError("the mesh has no faces")
    if tgrp.max() > 255 or tgrp.min() < 0: raise WriteError("too many materials (max 256)")

    # ---------- original per-vertex data in destination-slot space (the order of m.pos / m.skin)
    dest = m.dest
    def scat(off, dt, k):
        o = u32(off)
        if not (0 < o and o + nv0 * k * np.dtype(dt).itemsize <= L0): return None
        a = np.frombuffer(bytes(B[o:o + nv0 * k * np.dtype(dt).itemsize]), dt).reshape(nv0, k)
        out = np.empty_like(a); out[dest] = a; return out
    vec1, vec2 = scat(0x38, '>i2', 3), scat(0x3C, '>i2', 3)
    b4 = scat(0xB4, '>u4', 1)
    raw0 = scat(0x34, '>i2', 3)
    has_scatter = 0 < u32(0xB0) and u32(0xB0) + nv0 * 2 <= L0
    desc = []
    for k in range(nmesh):
        d = dbase + k * 276
        desc.append(dict(d=d, vb=u16(d + 0xA), nv=u16(d + 8)))
    desc_of_src = np.full(nv0, -1, np.int64)
    for k, ds in enumerate(desc): desc_of_src[ds['vb']:ds['vb'] + ds['nv']] = k
    desc_of_dest = np.empty(nv0, np.int64); desc_of_dest[dest] = desc_of_src
    if (desc_of_dest < 0).any(): desc_of_dest[desc_of_dest < 0] = 0

    # ---------- nearest original vertex (for vertices created in Blender)
    def nearest(idx):
        out = np.empty(len(idx), np.int64)
        P = m.pos.astype(np.float64)
        for a in range(0, len(idx), 256):
            q = pos[idx[a:a + 256]]
            dd = ((q[:, None, :] - P[None, :, :]) ** 2).sum(-1)
            out[a:a + 256] = dd.argmin(1)
        return out
    src = vid.copy(); bad = (src < 0) | (src >= nv0)
    new_idx = np.nonzero(bad)[0]
    if len(new_idx): src[new_idx] = nearest(new_idx); rep.append("%d new vertices: skin/auxiliary data taken from the nearest original vertex" % len(new_idx))
    home = desc_of_dest[src]
    # descriptor of a new vertex: that of the faces it belongs to when those use original vertices
    if len(new_idx):
        for _ in range(3):
            for t in tris:
                known = [int(home[v]) for v in t if not bad[v]]
                if known:
                    c = Counter(known).most_common(1)[0][0]
                    for v in t:
                        if bad[v]: home[v] = c; bad[v] = False
    # ---------- faces -> descriptor
    fdesc = np.array([Counter(int(home[v]) for v in t).most_common(1)[0][0] for t in tris], np.int64)

    # ---------- per vertex source data
    skin_geo = geo.get('skin')
    def skin_of(i):
        s = skin_geo[i] if skin_geo is not None else None
        if s: return s
        return list(m.skin[int(src[i])] or [])
    maxinf = 1
    for s in m.skin:
        if s: maxinf = max(maxinf, len(s))

    bv_of_vid = {}
    for bv_, v_ in enumerate(vid):
        if v_ >= 0: bv_of_vid.setdefault(int(v_), bv_)
    # ---------- build entries (descriptor, blender vertex) in descriptor order
    entries = []                       # (desc, bv)
    ent_of = {}
    uv_tabs = []; corner = []; prims = []     # prims: (count, group, start_corner)
    dinfo = []
    for k in range(nmesh):
        fs = np.nonzero(fdesc == k)[0]
        used = sorted({int(v) for f in fs for v in tris[f]}, key=lambda v: (0, int(vid[v])) if 0 <= vid[v] < nv0 else (1, v))
        vstart = len(entries)
        for v in used: ent_of[(k, v)] = len(entries); entries.append((k, v))
        uvmap = {}; uvl = []
        def uvidx(uv):
            key = (int(np.clip(round(uv[0] * 512.0), -32768, 32767)), int(np.clip(round(uv[1] * 512.0), -32768, 32767)))
            if key not in uvmap: uvmap[key] = len(uvl); uvl.append(key)
            return uvmap[key]
        ustart = sum(len(x) for x in uv_tabs)
        pstart = len(prims)
        bygrp = defaultdict(list)
        for f in fs:
            ck = tuple((ent_of[(k, int(tris[f][c]))], ustart + uvidx(tuv[f][c])) for c in range(3))
            if len({c[0] for c in ck}) < 3: continue
            bygrp[int(tgrp[f])].append(ck)
        # uv indices must be final before strips are made: uvidx above already used the global offset
        seeds = defaultdict(list)
        od = desc[k]['d']; onp, ops = u16(od), u16(od + 2)
        for pi in range(ops, ops + onp):
            if pi >= len(m.prim): break
            cnt_, st_ = int(m.prim[pi][0]) & 0xFF, int(m.prim[pi][1]); g_ = int(m.prim[pi][0]) >> 8
            sq = []
            for ci in range(st_, st_ + cnt_):
                v_, u_ = int(m.corner[ci][0]), int(m.corner[ci][1])
                bv_ = bv_of_vid.get(v_); uk_ = (int(round(float(m.uv[u_][0]) * 512)), int(round(float(m.uv[u_][1]) * 512)))
                if bv_ is None or (k, bv_) not in ent_of or uk_ not in uvmap: sq = None; break
                sq.append((ent_of[(k, bv_)], ustart + uvmap[uk_]))
            if sq: seeds[g_].append(sq)
        for g in sorted(bygrp):
            cand = [stripify(bygrp[g])] + ([stripify(bygrp[g], seeds=seeds[g])] if seeds.get(g) else [])
            for s in min(cand, key=lambda L: sum(len(x) for x in L)):
                prims.append((len(s), g, len(corner))); corner.extend(s)
        uv_tabs.append(uvl)
        dinfo.append(dict(vstart=vstart, nv=len(entries) - vstart, ustart=ustart, nu=len(uvl), pstart=pstart, np=len(prims) - pstart))
    nv = len(entries)
    old_ptrs_early = {o: struct.unpack('>I', B[o:o + 4])[0] for o in range(0x20, 0x100, 4)}
    if nv > 65535 or len(corner) > 65535 or len(prims) > 65535:
        raise WriteError("model too large for the format (vertices %d, corners %d, prims %d; limit 65535)" % (nv, len(corner), len(prims)))
    ent_bv = np.array([e[1] for e in entries], np.int64)
    # ---------- tables
    P = np.rint(pos[ent_bv] * div); clipped = int((np.abs(P) > 32767).sum()); P = np.clip(P, -32768, 32767).astype('>i2')
    if clipped: rep.append("WARNING: %d vertex coordinates are outside the +-%.0f unit range of the format and were clamped" % (clipped, 32767 / div))
    if nrm is None: N = raw0[src[ent_bv]].astype('>i2')
    else:
        nn = np.asarray(nrm, np.float64)[ent_bv]; nn = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
        N = np.clip(np.rint(nn * 64.0), -32768, 32767).astype('>i2')
    V1 = (vec1[src[ent_bv]] if vec1 is not None else None); V2 = (vec2[src[ent_bv]] if vec2 is not None else None)
    B4 = (b4[src[ent_bv], 0] if b4 is not None else None)
    UVt = np.array([uv for l in uv_tabs for uv in l], '>i2').reshape(-1, 2)
    C = np.array(corner, np.int64).reshape(-1, 2).astype('>u2')
    PR = np.array([(c | (g << 8), s) for c, g, s in prims], '>u2').reshape(-1, 2)
    # skin: influence sets / ranges.  The game keeps vertices in SOURCE order (grouped by influence set, so that few
    # ranges are needed) and maps them to the draw slots the corners use through the scatter table (+0xB0).
    skin_cap = (old_ptrs_early[0x28] - old_ptrs_early[0x2C]) if old_ptrs_early[0x28] > old_ptrs_early[0x2C] else 1 << 30
    def build_skin(q):
        sets = {}; set_list = []; infl = []; allr = []; ent_set = []
        for j in range(nv):
            bv = int(ent_bv[j]); s_ = skin_of(bv)
            o_ = m.skin[int(vid[bv])] if 0 <= vid[bv] < nv0 else None
            snap = None
            if o_ and s_:
                d_ = {int(b): float(w) for b, w in o_}; e_ = {int(b): float(w) for b, w in s_}
                tot_ = sum(e_.values()) or 1.0
                if set(d_) == set(e_) and all(abs(d_[b] - e_[b] / tot_) < 0.003 for b in d_): snap = tuple((int(b), float(w)) for b, w in o_)
            if snap is not None: s2 = snap
            else:
                top = sorted(s_, key=lambda bw: -bw[1])[:maxinf]
                if q: top = [(b, round(w / q) * q) for b, w in top]
                top = [(b, w) for b, w in top if w > 0]
                tot = sum(w for b, w in top)
                s2 = tuple(sorted((int(b), float(np.float32(w / tot))) for b, w in top)) if tot > 0 else ((0, 1.0),)
            if s2 not in sets:
                sets[s2] = len(set_list); set_list.append((len(s2), len(infl))); infl.extend(s2)
            ent_set.append(sets[s2])
        order = []
        for k in range(nmesh):
            di = dinfo[k]; idx = list(range(di['vstart'], di['vstart'] + di['nv']))
            if has_scatter: idx.sort(key=lambda j: ent_set[j])
            order.extend(idx)
        order = np.array(order, np.int64)
        for k in range(nmesh):
            di = dinfo[k]; runs = []
            for sp in range(di['vstart'], di['vstart'] + di['nv']):
                sid = ent_set[int(order[sp])]
                if runs and runs[-1][0] == sid and runs[-1][1] < 65535: runs[-1][1] += 1
                else: runs.append([sid, 1, sp - di['vstart']])
            di['vr0'] = len(allr); di['nvr'] = len(runs); allr.extend(runs)
        size = len(set_list) * 4 + len(allr) * 6 + len(infl) * 8
        return size, set_list, infl, allr, order
    for q in (None, 0.01, 0.02, 0.05, 0.1, 0.2, 0.34, 0.5):
        size, set_list, infl, allr, order = build_skin(q)
        if size <= skin_cap: break
    if q: rep.append("skin weights were rounded to steps of %g so the influence tables fit the game's fixed space" % q)
    if size > skin_cap: raise WriteError("The skin weights need %d bytes but the game reserves %d for them: use fewer bones per vertex / simpler weights" % (size, skin_cap))
    if len(infl) > 65535 or len(set_list) > 65535 or len(allr) > 65535: raise WriteError("skin tables too large")
    SETS = np.array(set_list, '>u2').reshape(-1, 2)
    VR = np.array(allr, '>u2').reshape(-1, 3)
    INF = np.zeros(len(infl), np.dtype([('w', '>f4'), ('b', '>u2'), ('p', '>u2')]))
    for i, (b, w) in enumerate(infl): INF[i] = (w, b, 0)
    # per-vertex arrays go out in source order
    P = P[order]; N = N[order]
    if V1 is not None: V1 = V1[order]
    if V2 is not None: V2 = V2[order]
    if B4 is not None: B4 = B4[order]
    if has_scatter and nv * 6 > 65535: raise WriteError("too many vertices for the scatter table (max %d)" % (65535 // 6))
    SCAT = order * 6                                # table[source position] = draw slot * 6
    # morph records
    morph = geo.get('morph')
    chans = {}
    if morph is not None:
        for ch, dl in morph.items():
            chans[int(ch)] = np.asarray(dl, np.float64)[ent_bv][order]
    else:
        for ch, (didx, dl) in m.morphs.items():
            full = np.zeros((nv0, 3)); np.add.at(full, didx, dl.astype(np.float64))
            chans[int(ch)] = full[src[ent_bv]][order]
            # a vertex that exists only through "nearest original" must not receive the deltas of that vertex
            fake = ((vid[ent_bv] < 0) | (vid[ent_bv] >= nv0))[order]
            chans[int(ch)][fake] = 0
    recs = []; mfirst = {}
    for k in range(nmesh):
        di = dinfo[k]
        for ch in range(32):
            if ch not in chans: mfirst[(k, ch)] = (0xFFFFFFFF, 0); continue
            seg = chans[ch][di['vstart']:di['vstart'] + di['nv']]
            raw = np.clip(np.rint(seg * div), -32768, 32767).astype(np.int64)
            nz = np.nonzero((raw != 0).any(1))[0]
            mfirst[(k, ch)] = (len(recs), len(nz)) if len(nz) else (0xFFFFFFFF, 0)
            for r in nz: recs.append((int(r), int(raw[r][0]), int(raw[r][1]), int(raw[r][2])))
    MR = np.zeros(len(recs), np.dtype([('v', '>u4'), ('d', '>i2', 3), ('p', '>u2')]))
    for i, (r, x, y, z) in enumerate(recs): MR[i] = (r, (x, y, z), 0)

    # ---------- append + patch
    # The game refuses resources that change size or position (checked in game), so everything must fit in the space
    # the original tables occupied.  Contiguous table groups are pooled: the vertex/face tables share one pool, the skin
    # tables another, so savings in one table (e.g. shorter strips) can pay for growth in another.
    def ptr(off, o): struct.pack_into('>I', B, off, o)
    def cnt(off, v): struct.pack_into('>H', B, off, v)
    nuv0, nco0, npr0, nset0, nrng0, nin0 = u16(0x1A), u16(0x18), u16(0x16), u16(0xA), u16(0xE), u16(0x1C)
    old_nv_fields = {o: u16(o) for o in (0x10, 0x12, 0x14, 0x1E)}
    old_ptrs = {o: u32(o) for o in range(0x20, 0x100, 4)}
    B_orig = bytes(B)
    budget = []
    def pool(specs, label):
        """specs: [(header offset, data, original size)] -> written inside the area of the original tables"""
        specs = [x for x in specs if x[1] is not None]
        starts = [old_ptrs[o] for o, _, _ in specs]
        lo = min(starts); covered = sum((z + 1) & ~1 for _, _, z in specs); slack = 4 * len(specs)
        outside = [v for o, v in old_ptrs.items() if v not in starts and lo + covered - slack <= v <= L0]
        hi = min(outside) if outside else L0
        np_ = geo.get('_nopool'); contiguous = (not (np_ is True or (np_ and label in np_))) and abs((hi - lo) - covered) <= slack and all(lo <= p < hi for p in starts)
        def alignment(p): return min(32, p & -p) if p else 2           # keep each table's original alignment (e.g. corners: 32)
        offs = []; tails = []; pos_ = lo
        srt = sorted(starts + [hi])
        for o, d, z in specs:
            # bytes the original keeps right after the table (e.g. the 0xFFFF terminator of the scatter table) travel with it
            nxt = min(v for v in srt if v > old_ptrs[o]); gap = B_orig[old_ptrs[o] + z: nxt] if 0 < nxt - old_ptrs[o] - z <= 32 else b''
            if gap and not any(gap): gap = b''
            al = alignment(old_ptrs[o]); pos_ = (pos_ + al - 1) // al * al; offs.append(pos_); tails.append(gap); pos_ += len(d) + len(gap)
        need = pos_ - lo
        if contiguous:
            if need > hi - lo:
                raise WriteError("The model is too big for the space the game reserves for it (%s: needs %d bytes, %d available). "
                                 "Reduce the vertex/face count by about %d%%." % (label, need, hi - lo, int(100 * (need - (hi - lo)) / max(need, 1)) + 1))
            B[lo:hi] = bytes(hi - lo)
            for (o, d, z), p_, t_ in zip(specs, offs, tails):
                ptr(o, p_); B[p_:p_ + len(d)] = d; B[p_ + len(d):p_ + len(d) + len(t_)] = t_
            budget.append("%s: %d of %d bytes used" % (label, need, hi - lo))
        else:
            for o, d, z in specs:
                p = old_ptrs[o]
                if not (0 < p and p + z <= L0 and len(d) <= z):
                    raise WriteError("A table of this model does not fit in its original space (%s, header +0x%X): %d > %d bytes" % (label, o, len(d), z))
                B[p:p + z] = d + bytes(z - len(d))
            budget.append("%s: tables written in their own spaces" % label)
    pool([(0xB0, SCAT.astype('>u2').tobytes() if has_scatter else None, nv0 * 2),
          (0xB4, B4.astype('>u4').tobytes() if B4 is not None else None, nv0 * 4),
          (0x50, P.tobytes(), nv0 * 6), (0x34, N.tobytes(), nv0 * 6),
          (0x38, V1.astype('>i2').tobytes() if V1 is not None else None, nv0 * 6),
          (0x3C, V2.astype('>i2').tobytes() if V2 is not None else None, nv0 * 6),
          (0x48, UVt.tobytes(), nuv0 * 4), (0x44, C.tobytes(), nco0 * 4)], "vertex/face data")
    pool([(0x40, PR.tobytes(), npr0 * 4)], "face strips")          # must stay at its original address (checked in game)
    pool([(0x2C, SETS.tobytes(), nset0 * 4), (0x30, VR.tobytes(), nrng0 * 6), (0x4C, INF.tobytes(), nin0 * 8)], "skin data")
    if old_ptrs.get(0x80) == old_ptrs[0x2C]: ptr(0x80, u32(0x2C))
    mp = old_ptrs[0xDC]
    if mp < 0x100 or mp >= L0:
        if len(MR): rep.append("WARNING: this model has no morph table: shape keys were ignored")
    else:
        morph_cap = (old_ptrs[0xE4] - mp) if mp < old_ptrs[0xE4] <= L0 else None
        mrb = MR.tobytes()
        if morph_cap is None or len(mrb) > morph_cap:
            raise WriteError("Too many morph (shape key) vertices for the original space: %d > %d records" % (len(MR), (morph_cap or 0) // 12))
        B[mp:mp + morph_cap] = mrb + bytes(morph_cap - len(mrb))
    for o, v in old_nv_fields.items():
        if v == nv0: cnt(o, nv)
    cnt(0x16, len(prims)); cnt(0x18, len(corner)); cnt(0x1A, len(UVt)); cnt(0x1C, len(infl)); cnt(0xA, len(set_list)); cnt(0xE, len(allr))
    for k in range(nmesh):
        d = desc[k]['d']; di = dinfo[k]
        struct.pack_into('>HHHHHH', B, d, di['np'], di['pstart'], di['nu'], di['ustart'], di['nv'], di['vstart'])
        cnt(d + 0x10, di['nvr']); cnt(d + 0x12, di['vr0'])
        for ch in range(32):
            f, c = mfirst[(k, ch)]; struct.pack_into('>II', B, d + 0x14 + 8 * ch, f, c)
    # ---------- materials
    mats = geo.get('mats')
    if mats:
        nmat, mo = u32(0x5C), u32(0x60)
        ng = max(max(mats) + 1, nmat, int(tgrp.max()) + 1)
        tab = bytearray(B[mo:mo + nmat * 32])
        for g in range(nmat, ng):
            t = mats.get(g, {}).get('template', 0)
            tab += tab[(t if t < nmat else 0) * 32:(t if t < nmat else 0) * 32 + 32] if nmat else bytes(32)
        for g, e in mats.items():
            if e.get('diffuse') is not None: struct.pack_into('>H', tab, g * 32, e['diffuse'])
            if e.get('secondary') is not None: struct.pack_into('>H', tab, g * 32 + 0x0E, e['secondary'])
        if ng > nmat:
            room = (old_ptrs[0x68] - mo) if old_ptrs.get(0x68, 0) > mo else 0         # the texture block follows the table
            if ng * 32 > room:
                raise WriteError("No room for %d extra material entries in the model (the game's material table is fixed in size). "
                                 "Reuse the existing materials or texture assignments." % (ng - nmat))
            B[mo:mo + ng * 32] = tab; struct.pack_into('>I', B, 0x5C, ng)
        else:
            B[mo:mo + nmat * 32] = tab
    if len(B) != L0: raise WriteError("internal error: resource size changed")
    rep.extend(budget)
    rep.append("rebuilt: %d vertices, %d corners, %d strips (%d triangles), %d uvs, %d influence sets, %d morph records"
               % (nv, len(corner), len(prims), len(tris), len(UVt), len(set_list), len(recs)))
    return res[:4] + bytes(B), rep
