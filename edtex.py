"""Texture table of a model resource + in-place texture patching (same dimensions, CMPR)."""
import struct
import numpy as np
from . import cmpr


class TexError(Exception):
    pass


def texture_table(res):
    """[dict(index, w, h, fmt, off, size)] ; off = absolute offset of the pixel data inside `res`."""
    B = res[4:]
    u32 = lambda o: struct.unpack('>I', B[o:o+4])[0]
    to = u32(0x68)
    if not (to and to + 12 <= len(B)): return []
    ntex = u32(to + 4); lo = u32(to + 8); out = []
    for t in range(ntex):
        eo = u32(to + lo + t * 8)
        hw, fmt, doff = struct.unpack('>III', B[to + eo: to + eo + 12])
        w, h = hw & 0xFFFF, hw >> 16
        size = ((w + 7) // 8 * 8) * ((h + 7) // 8 * 8) // 2 if fmt == 14 else 0
        out.append(dict(index=t, w=w, h=h, fmt=fmt, off=4 + to + doff, size=size))
    return out


def read_texture(res, t):
    if t['fmt'] != 14: return None
    return cmpr.decode_cmpr(res[t['off']: t['off'] + t['size']], t['w'], t['h'])


def patch_textures(res, images):
    """images: {texture index: uint8 RGBA array (h, w, 4)}.  Only textures whose pixels differ from the stored ones are
    re-encoded; everything else (and every byte outside the pixel data) is untouched.  Returns (new_res, [indices changed])."""
    tab = {t['index']: t for t in texture_table(res)}
    out = bytearray(res); changed = []
    for idx, img in images.items():
        t = tab.get(idx)
        if t is None: raise TexError("texture %d does not exist in this model" % idx)
        if t['fmt'] != 14: raise TexError("texture %d is not CMPR (format %d): not supported" % (idx, t['fmt']))
        img = np.asarray(img, np.uint8)
        if img.shape[0] != t['h'] or img.shape[1] != t['w']:
            raise TexError("texture %d must be %dx%d, got %dx%d" % (idx, t['w'], t['h'], img.shape[1], img.shape[0]))
        if img.shape[2] == 3: img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        old = read_texture(res, t)
        # compare ignoring RGB of fully transparent pixels (it is not stored)
        a = old.astype(int); b = img.astype(int)
        mask = (old[..., 3] >= 128) | (img[..., 3] >= 128)
        if np.array_equal(old[..., 3] >= 128, img[..., 3] >= 128) and not np.any(np.abs(a - b)[..., :3][mask] > 0):
            continue
        out[t['off']: t['off'] + t['size']] = cmpr.encode_cmpr(img)
        changed.append(idx)
    return bytes(out), changed
