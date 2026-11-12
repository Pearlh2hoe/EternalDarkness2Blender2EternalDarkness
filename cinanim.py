"""Cinematic animation support (pure numpy; no bpy).  See cin.py for the stream format."""
import os, struct
import numpy as np
try:
    from . import cin, skasc
except ImportError:
    import cin, skasc

def load_frames(stream_path):
    """Decode an extension-less cinNNNN file into a list of parsed frames (None when unreadable)."""
    d = open(stream_path, 'rb').read()
    if d[:8] == b'*SK_ASC*': d = skasc.decode(d)
    starts, sizes = cin.frame_table(d)
    frames = []
    for s, z in zip(starts, sizes):
        try: frames.append(cin.parse_frame(cin.decode_frame(d, s, int(z))))
        except Exception: frames.append(None)
    return frames

def tracks(frames):
    """obj id -> [(frame_index, entity_pos, {bone: (quat, pos, flag)})]"""
    out = {}
    for fi, f in enumerate(frames):
        if f is None: continue
        for e in f['ents']:
            out.setdefault(e['obj'], []).append((fi, e['pos'], e['bones']))
    return out

def morph_tracks(frames):
    """obj id -> [(frame_index, {channel id: value})] -- only the channels written in that frame (values persist)."""
    out = {}
    for fi, f in enumerate(frames):
        if f is None: continue
        for e in f['ents']:
            if e['A']: out.setdefault(e['obj'], []).append((fi, {int(c): float(v) for c, v in e['A']}))
    return out

def quat_to_mat(q):
    x, y, z, w = q; n = np.sqrt(x*x + y*y + z*z + w*w) or 1.0; x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2; q = [(R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s, 0.25*s]
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2]) * 2; q = [0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s]
    elif R[1,1] > R[2,2]:
        s = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2]) * 2; q = [(R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s]
    else:
        s = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1]) * 2; q = [(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s]
    q = np.array(q); return q / np.linalg.norm(q)

def bind_worlds(model):
    return [np.linalg.inv(np.vstack([b['inv_bind'], [0, 0, 0, 1]])) for b in model.bones]

def bind_local(model):
    """per-bone (quat, pos) of the bind pose relative to the parent (root: relative to the origin)"""
    W = bind_worlds(model); out = []
    for i, b in enumerate(model.bones):
        p = b['parent']
        L = W[i] if not (0 <= p < len(W)) or p == i else np.linalg.inv(W[p]) @ W[i]
        out.append((mat_to_quat(L[:3, :3]), L[:3, 3].copy()))
    return out

def match_model(track, models):
    """Pick the model (dict slot -> NpcModel) whose bone offsets agree with the keyframe records of an entity."""
    best = None
    kf = max(track, key=lambda t: len(t[2]))
    recs = kf[2]
    if len(recs) < 3: return None
    for slot, m in models.items():
        nb = len(m.bones)
        if nb < 3 or max(recs) >= nb + 2: continue
        W = bind_worlds(m); diffs = []
        for bid, (q, p, fl) in recs.items():
            if bid == 0 or bid >= nb: continue
            par = m.bones[bid]['parent']
            if not 0 <= par < nb: continue
            loc = np.linalg.inv(W[par]) @ W[bid]
            diffs.append(abs(np.linalg.norm(loc[:3, 3]) - np.linalg.norm(p)))
        if len(diffs) < 3: continue
        score = float(np.median(diffs)) + 0.02 * abs(nb - len(recs))
        if best is None or score < best[0]: best = (score, slot)
    return best[1] if best and best[0] < 1.5 else None

def entity_poses(model, track):
    """Yield (frame_index, [world 4x4 per bone]) for every frame the entity appears in (poses carry over)."""
    nb = len(model.bones); par = [b['parent'] for b in model.bones]
    loc = bind_local(model); state = {i: loc[i] for i in range(nb)}
    Wb = bind_worlds(model)
    for fi, epos, bones in track:
        for bid, (q, p, fl) in bones.items():
            if bid < nb: state[bid] = (q, p)
        W = [None] * nb
        for i in range(nb):
            M = np.eye(4); M[:3, :3] = quat_to_mat(state[i][0]); M[:3, 3] = state[i][1]
            if par[i] < 0 or par[i] == i:
                if i == 0:
                    T = np.eye(4); T[:3, 3] = epos; M = T @ M
                    W[i] = M
                else: W[i] = Wb[i]
            else: W[i] = W[par[i]] @ M
        yield fi, W

def camera_track(frames):
    """[(frame, pos, target, up, fov_degrees)] for every decodable frame.
    Header floats: pos f[0:3], up f[7:10] (always orthogonal to target-pos), target f[11:14]; fov (deg) at +0x54."""
    out = []
    for fi, f in enumerate(frames):
        if f is None: continue
        c = f['cam']
        pos, up, tgt = np.array(c[0:3]), np.array(c[7:10]), np.array(c[11:14])
        if np.linalg.norm(tgt - pos) < 1e-3 or np.linalg.norm(up) < 1e-3: continue
        out.append((fi, pos, tgt, up, float(f['fov'])))
    return out

def camera_matrix(pos, tgt, up):
    """4x4 (game axes) for a Blender-style camera: looks along -Z, +Y up."""
    d = tgt - pos; d = d / np.linalg.norm(d)
    y = up / np.linalg.norm(up)
    x = np.cross(d, y); x /= np.linalg.norm(x)
    y = np.cross(x, d)
    M = np.eye(4); M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = -d; M[:3, 3] = pos
    return M
