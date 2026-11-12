"""Pure-Python decoder for Silicon Knights '*SK_ASC*' streams (Eternal Darkness, GameCube).

Port of the game's own decoder (0x8013F634): an LZ-style coder whose symbols are arithmetic coded
(16-bit low/high/code, Nelson-style) with adaptive frequency trees:
  flag  : literal(0) / match(1) / EOF, 4 contexts of (c0,c1) counts
  literal: tree T1 (recently used bytes, 256 leaves) with an escape count; escape -> T2 (unseen bytes, exclusion)
  match : distance class via T3 (16 leaves) + uniform extra bits; length via L1/L2 (64 leaves) + 4 extra bits
"""
M32 = 0xFFFFFFFF

class _Tree:
    """sum tree: t[1] root, t[n..2n-1] leaves, as in the game's 0x8013EC98 helper."""
    __slots__ = ('n', 't')
    def __init__(self, n, leaf):
        self.n = n; self.t = [0] * (2 * n)
        for i in range(n): self.t[n + i] = leaf
        for i in range(n - 1, 0, -1): self.t[i] = self.t[2*i] + self.t[2*i+1]
    def add(self, idx, inc, limit):
        t = self.t; n = self.n; i = idx + n
        while i: t[i] += inc; i >>= 1
        if t[1] >= limit:
            for j in range(2*n - 1, n - 1, -1):
                if t[j] > 1: t[j] >>= 1
            for j in range(n - 1, 0, -1): t[j] = t[2*j] + t[2*j+1]
    def remove(self, sym):
        t = self.t; v = t[self.n + sym]; i = sym + self.n
        while i: t[i] -= v; i >>= 1
    def find(self, scaled):
        """descend: returns (symbol, cumulative_below)"""
        t = self.t; n = self.n; node = 2; cum = 0
        while True:
            v = t[node]
            if cum + v <= scaled: cum += v; node += 1
            if node >= n: return node - n, cum
            node <<= 1

def decode(data, skip_magic=True):
    if skip_magic:
        if data[:8] != b'*SK_ASC*': raise ValueError("not an SK_ASC stream")
        data = data[8:]
    n_in = len(data); ip = 0
    def getbyte():
        nonlocal ip
        if ip < n_in: b = data[ip]; ip += 1; return b
        return -1
    b0 = getbyte(); b1 = getbyte()
    L, H = 0, 0xFFFF
    C = ((b0 & 0xFF) << 8 | (b1 & 0xFF)) & 0xFFFF
    buf = 0

    def arith(lo_c, hi_c, total):
        """narrow [L,H] to the symbol range and renormalise exactly like the game code."""
        nonlocal L, H, C, buf
        rng = H - L + 1
        r23 = L + rng * hi_c // total - 1          # high
        r24 = L + rng * lo_c // total              # low
        r28 = C; r4 = buf
        while not ((r23 ^ r24) & 0x8000):
            r4 = (r4 << 1) & M32; r23 = ((r23 << 1) | 1) & M32
            r24 = (r24 << 1) & M32; r28 = (r28 << 1) & M32
            if not (r4 & 0xFF):
                b = getbyte(); r4 = 256 if (b & 0x100) else ((b << 1) | 1)   # b=-1 -> 256
            r28 |= (r4 >> 8) & 1
        comb = ((r23 << 16) | r24) & M32
        while (comb & 0x40004000) == 0x4000:
            r4 = (r4 << 1) & M32; comb = (comb << 1) & M32; r28 = (r28 << 1) & M32
            comb = (comb & 0xFFFF7FFF & M32) | 0x80010000
            r28 ^= 0x8000
            if not (r4 & 0xFF):
                b = getbyte(); r4 = 256 if (b & 0x100) else ((b << 1) | 1)
            r28 |= (r4 >> 8) & 1
        L = comb & 0xFFFF; H = comb >> 16; C = r28 & 0xFFFF; buf = r4

    def scaled(total):
        return ((C - L + 1) * total - 1) // (H - L + 1)

    # ---- model state (values from game init 0x8013EDE8)
    cnt = [40] * 8                    # [ctx*2+flag]
    T1 = _Tree(256, 0); T2 = _Tree(256, 1)
    T3 = _Tree(16, 0); T3.t[16] = 24
    for i in range(15, 0, -1): T3.t[i] = T3.t[2*i] + T3.t[2*i+1]
    L1 = _Tree(64, 0); L2 = _Tree(64, 1)
    A50 = 0; A54 = 1; A58 = 1; A5C = 1; A60 = 8; ctx = 0
    hist = 0            # r29
    out = bytearray()

    while True:
        c0 = cnt[ctx*2]; c1 = cnt[ctx*2+1]; tot = c0 + c1
        s = scaled(tot + 1)
        if s >= tot:                                   # EOF
            arith(tot, tot + 1, tot + 1); break
        if s < c0:                                     # ------- literal
            arith(0, c0, tot + 1)
            cnt[ctx*2] = c0 + 40
            if tot >= 6000:
                cnt[ctx*2] = max(cnt[ctx*2] >> 1, 1); cnt[ctx*2+1] = max(c1 >> 1, 1)
            ctx = (ctx << 1) & 2
            t1 = T1.t[1]; total = t1 + A5C
            s2 = scaled(total)
            if s2 < t1:
                sym, cum = T1.find(s2)
                arith(cum, cum + T1.t[256 + sym], total)
            else:
                arith(t1, total, total)
                t2 = T2.t[1]
                sym, cum = T2.find(scaled(t2))
                arith(cum, cum + T2.t[256 + sym], t2)
                T2.remove(sym)
                A5C = A5C + 1 if T2.t[1] else 0
                for i in range(max(sym - 8, 0), min(sym + 8, 255)):
                    if T2.t[256 + i]: T2.add(i, 1, 1000)
            T1.add(sym, 1, 1000)
            if T1.t[256 + sym] == 3:
                A5C = A5C - 1 if A5C > 1 else 1
            out.append(sym)
            if hist < 31200: hist += 1
            continue
        # ------- match
        arith(c0, tot, tot + 1)
        cnt[ctx*2+1] = c1 + 40
        if tot >= 6000:
            cnt[ctx*2] = max(c0 >> 1, 1); cnt[ctx*2+1] = max(cnt[ctx*2+1] >> 1, 1)
        ctx = ((ctx << 1) | 1) & 3
        while hist > A54:
            T3.add(A58, 24, 6000); A58 += 1; A54 <<= 1
        t3 = T3.t[1]
        k, cum = T3.find(scaled(t3))
        arith(cum, cum + T3.t[16 + k], t3)
        T3.add(k, 24, 6000)
        if k <= 1:
            off = k
        else:
            half = 1 << (k - 1)
            tot_u = half
            if half == (A54 >> 1): tot_u = hist - half
            rng = H - L + 1
            v = ((C - L + 1) * tot_u - 1) // rng
            arith(v, v + 1, tot_u)
            off = v + half
        # length
        l1 = L1.t[1]; total = l1 + A60
        s2 = scaled(total)
        if s2 < l1:
            sym, cum = L1.find(s2)
            arith(cum, cum + L1.t[64 + sym], total)
        else:
            arith(l1, total, total)
            l2 = L2.t[1]
            sym, cum = L2.find(scaled(l2))
            arith(cum, cum + L2.t[64 + sym], l2)
            L2.remove(sym)
            A60 = A60 + 8 if L2.t[1] else 0
            for i in range(max(sym - 4, 0), min(sym + 4, 63)):
                if L2.t[64 + i]: L2.add(i, 1, 6000)
        L1.add(sym, 8, 6000)
        if L1.t[64 + sym] == 24:
            r3 = 8 if A60 > 8 else A60 - 1
            A60 -= r3
        if sym == 15: ln = 783
        elif sym < 16: ln = sym
        else:
            v = scaled(16)
            arith(v, v + 1, 16)
            ln = ((sym - 16) << 4) + v + 15
        if hist < 31200:
            hist += ln + 3
            if hist > 31200: hist = 31200
        n = ln + 3
        if off + 1 > len(out): out.extend(b'\0' * (off + 1 - len(out)))  # unreachable in valid streams
        for _ in range(n): out.append(out[-(off + 1)])
    return bytes(out)
