"""GameCube CMPR (GX_TF_CMPR, format 14) encoder and decoder (numpy).

Layout: 8x8 pixel tiles in row-major order; each tile = 4 sub-blocks (4x4 pixels) in the order
(0,0) (4,0) (0,4) (4,4); sub-block = u16 colour0, u16 colour1 (RGB565 big-endian), 4 index bytes (one per pixel row,
2 bits per pixel, leftmost pixel in the top bits).  colour0 > colour1: 4 colours (c0, c1, 2/3 c0 + 1/3 c1, 1/3 c0 + 2/3 c1);
colour0 <= colour1: 3 colours (c0, c1, mean) + index 3 = transparent (1-bit alpha).
"""
import numpy as np


def _expand565(v):
    r = (v >> 11) & 31; g = (v >> 5) & 63; b = v & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.int32)


def _quant565(rgb):
    """rgb float/int (...,3) 0..255 -> u16 RGB565 (nearest)"""
    r = np.clip(np.rint(rgb[..., 0] * 31.0 / 255.0), 0, 31).astype(np.int32)
    g = np.clip(np.rint(rgb[..., 1] * 63.0 / 255.0), 0, 63).astype(np.int32)
    b = np.clip(np.rint(rgb[..., 2] * 31.0 / 255.0), 0, 31).astype(np.int32)
    return (r << 11) | (g << 5) | b


def _palette(c0, c1):
    """c0,c1: (n,) u16 -> palette (n,4,3) int32 and 4-colour flag (n,)"""
    p0 = _expand565(c0); p1 = _expand565(c1)
    four = c0 > c1
    pal = np.zeros((len(c0), 4, 3), np.int32)
    pal[:, 0] = p0; pal[:, 1] = p1
    pal[:, 2] = np.where(four[:, None], (2 * p0 + p1) // 3, (p0 + p1) // 2)
    pal[:, 3] = np.where(four[:, None], (p0 + 2 * p1) // 3, 0)
    return pal, four


def decode_cmpr(data, w, h):
    W = (w + 7) // 8 * 8; H = (h + 7) // 8 * 8
    nb = W // 4 * (H // 4)
    a = np.frombuffer(bytes(data[:nb * 8]), np.uint8).reshape(nb, 8)
    c0 = (a[:, 0].astype(np.int32) << 8) | a[:, 1]; c1 = (a[:, 2].astype(np.int32) << 8) | a[:, 3]
    pal, four = _palette(c0, c1)
    idx = np.stack([(a[:, 4 + y, None] >> (6 - 2 * np.arange(4))) & 3 for y in range(4)], 1)       # (nb,4rows,4cols)
    rgb = pal[np.arange(nb)[:, None, None], idx]
    alpha = np.where((~four)[:, None, None] & (idx == 3), 0, 255)
    blk = np.concatenate([rgb, alpha[..., None]], -1).astype(np.uint8)                  # (nb,4,4,4)
    # block order -> image
    tiles_x = W // 8
    blk = blk.reshape(H // 8, tiles_x, 2, 2, 4, 4, 4)           # tileY, tileX, sbY, sbX, y, x, c
    img = blk.transpose(0, 2, 4, 1, 3, 5, 6).reshape(H, W, 4)
    return img[:h, :w]


def encode_cmpr(img):
    """img: uint8 (h,w,4) RGBA -> CMPR bytes for the padded (multiple of 8) size."""
    img = np.asarray(img, np.uint8)
    h, w = img.shape[:2]
    W = (w + 7) // 8 * 8; H = (h + 7) // 8 * 8
    if (W, H) != (w, h): img = np.pad(img, ((0, H - h), (0, W - w), (0, 0)), mode='edge')
    tiles_x = W // 8
    blk = img.reshape(H // 8, 2, 4, tiles_x, 2, 4, 4).transpose(0, 3, 1, 4, 2, 5, 6)   # tileY,tileX,sbY,sbX,y,x,c
    blk = blk.reshape(-1, 16, 4).astype(np.float64)
    nb = len(blk); rgb = blk[:, :, :3]; opaque = blk[:, :, 3] >= 128
    has_t = ~opaque.all(1)
    wgt = opaque.astype(np.float64)
    nop = wgt.sum(1)
    mean = (rgb * wgt[..., None]).sum(1) / np.maximum(nop, 1)[:, None]
    # principal axis of the opaque pixels (power iteration on the 3x3 covariance)
    d = (rgb - mean[:, None]) * wgt[..., None]
    cov = np.einsum('npi,npj->nij', d, d)
    v = np.tile(np.array([0.57735, 0.57735, 0.57735]), (nb, 1))
    for _ in range(6):
        v = np.einsum('nij,nj->ni', cov, v); nr = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(nr > 1e-9, v / np.maximum(nr, 1e-9), np.array([0.57735, 0.57735, 0.57735]))
    proj = np.einsum('npi,ni->np', rgb - mean[:, None], v)
    big = np.where(opaque, proj, -1e18).argmax(1); small = np.where(opaque, proj, 1e18).argmin(1)
    e0 = rgb[np.arange(nb), big]; e1 = rgb[np.arange(nb), small]

    def order(e0, e1):
        c0 = _quant565(e0); c1 = _quant565(e1)
        want4 = ~has_t
        swap = np.where(want4, c0 < c1, c0 > c1)
        c0n = np.where(swap, c1, c0); c1n = np.where(swap, c0, c1)
        c0, c1 = c0n, c1n
        same = (c0 == c1) & want4
        # an opaque block whose endpoints coincide must stay a valid 4-colour block or use index 0 only (3-colour mode is fine)
        return c0, c1

    def assign(c0, c1):
        pal, four = _palette(c0, c1)
        dist = ((rgb[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)               # (nb,16,4)
        dist[:, :, 3] = np.where(four[:, None], dist[:, :, 3], 1e18)                    # index 3 is "transparent" in 3-colour mode
        idx = dist.argmin(2)
        idx = np.where(opaque, idx, 3)
        err = np.where(opaque, np.take_along_axis(dist, np.minimum(idx, 3)[..., None], 2)[..., 0], 0).sum(1)
        return idx, err, four

    c0, c1 = order(e0, e1)
    idx, err, four = assign(c0, c1)
    best = (c0.copy(), c1.copy(), idx.copy(), err.copy())
    # least-squares refinement of the endpoints (iterated cluster fit)
    for _ in range(3):
        t = np.where(four[:, None], np.array([0, 1, 1 / 3, 2 / 3]), np.array([0, 1, 0.5, 0.5]))
        t = np.take_along_axis(t, np.where(idx > 3, 3, idx), 1)                     # position along c0->c1 (alpha for c1)
        a = (1 - t) * opaque; b = t * opaque
        aa = (a * a).sum(1); bb = (b * b).sum(1); ab = (a * b).sum(1)
        det = aa * bb - ab * ab
        ok = np.abs(det) > 1e-6
        ax = np.einsum('np,npc->nc', a, rgb); bx = np.einsum('np,npc->nc', b, rgb)
        n0 = np.where(ok[:, None], (bb[:, None] * ax - ab[:, None] * bx) / np.where(ok, det, 1)[:, None], e0)
        n1 = np.where(ok[:, None], (aa[:, None] * bx - ab[:, None] * ax) / np.where(ok, det, 1)[:, None], e1)
        c0n, c1n = order(n0, n1)
        idx_n, err_n, four_n = assign(c0n, c1n)
        better = err_n < best[3]
        best = (np.where(better, c0n, best[0]), np.where(better, c1n, best[1]),
                np.where(better[:, None], idx_n, best[2]), np.where(better, err_n, best[3]))
        c0, c1, idx, four = c0n, c1n, idx_n, four_n
    c0, c1, idx, err0 = best
    # opaque blocks may also use the 3-colour mode (c0 <= c1, mean as 3rd colour, index 3 unused): keep the better one
    alt0 = np.where(c0 > c1, c1, c0); alt1 = np.where(c0 > c1, c0, c1)
    a_idx, a_err, a_four = assign(alt0, alt1)
    use = (~has_t) & (a_err < err0) & (~a_four)
    c0 = np.where(use, alt0, c0); c1 = np.where(use, alt1, c1); idx = np.where(use[:, None], a_idx, idx)
    # a flat opaque block (c0 == c1) is a 3-colour block: index 0 gives exactly colour0; make sure nothing uses index 3
    pal, four = _palette(c0, c1)
    idx = np.where(opaque & (~four)[:, None] & (idx == 3), 0, idx)
    out = np.zeros((nb, 8), np.uint8)
    out[:, 0] = (c0 >> 8) & 255; out[:, 1] = c0 & 255; out[:, 2] = (c1 >> 8) & 255; out[:, 3] = c1 & 255
    ib = idx.reshape(nb, 4, 4).astype(np.uint8)
    for y in range(4):
        out[:, 4 + y] = (ib[:, y, 0] << 6) | (ib[:, y, 1] << 4) | (ib[:, y, 2] << 2) | ib[:, y, 3]
    return out.tobytes()
