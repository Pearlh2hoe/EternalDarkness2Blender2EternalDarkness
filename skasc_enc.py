"""Pure-Python ENCODER for Silicon Knights '*SK_ASC*' streams (Eternal Darkness, GameCube).

Exact mirror of skasc.decode (which is byte-exact against the game's decoder at 0x8013F634): same adaptive models
(flag counts, byte trees T1/T2, distance-class tree T3, length trees L1/L2), same arithmetic coder
(16-bit low/high, underflow handling).  Symbols: literal / match(distance <= 31200, length 3..785) / EOF.
Output decodes with skasc.decode() and with the game's own decoder.
"""
M32 = 0xFFFFFFFF
MAX_DIST = 31200


class _Tree:
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

    def below(self, sym):
        """sum of the leaves left of `sym`"""
        t = self.t; i = sym + self.n; cum = 0
        while i > 1:
            if i & 1: cum += t[i - 1]
            i >>= 1
        return cum


class _BitOut:
    def __init__(self):
        self.buf = bytearray(); self.acc = 0; self.nacc = 0

    def bit(self, b):
        self.acc = (self.acc << 1) | b; self.nacc += 1
        if self.nacc == 8:
            self.buf.append(self.acc); self.acc = 0; self.nacc = 0

    def finish(self):
        while self.nacc:                      # pad with 1 bits: the decoder reads 1s past the end
            self.bit(1)
        return bytes(self.buf)


def encode(data, matches=True, chain=24):
    data = bytes(data)
    out = _BitOut(); L = 0; H = 0xFFFF; pending = 0

    def arith(lo_c, hi_c, total):
        nonlocal L, H, pending
        rng = H - L + 1
        nH = L + rng * hi_c // total - 1
        nL = L + rng * lo_c // total
        L, H = nL, nH
        while not ((H ^ L) & 0x8000):
            b = H >> 15; out.bit(b)
            while pending: out.bit(b ^ 1); pending -= 1
            L = (L << 1) & 0xFFFF; H = ((H << 1) | 1) & 0xFFFF
        while (L & 0x4000) and not (H & 0x4000):
            pending += 1
            L = (L << 1) & 0x7FFF; H = (((H | 0x4000) << 1) | 1) & 0xFFFF

    cnt = [40] * 8
    T1 = _Tree(256, 0); T2 = _Tree(256, 1)
    T3 = _Tree(16, 0); T3.t[16] = 24
    for i in range(15, 0, -1): T3.t[i] = T3.t[2*i] + T3.t[2*i+1]
    L1 = _Tree(64, 0); L2 = _Tree(64, 1)
    A54 = 1; A58 = 1; A5C = 1; A60 = 8; ctx = 0; hist = 0

    def flag(is_match, is_eof=False):
        nonlocal ctx
        c0 = cnt[ctx*2]; c1 = cnt[ctx*2+1]; tot = c0 + c1
        if is_eof:
            arith(tot, tot + 1, tot + 1); return
        if not is_match:
            arith(0, c0, tot + 1)
            cnt[ctx*2] = c0 + 40
            if tot >= 6000:
                cnt[ctx*2] = max(cnt[ctx*2] >> 1, 1); cnt[ctx*2+1] = max(c1 >> 1, 1)
            ctx = (ctx << 1) & 2
        else:
            arith(c0, tot, tot + 1)
            cnt[ctx*2+1] = c1 + 40
            if tot >= 6000:
                cnt[ctx*2] = max(c0 >> 1, 1); cnt[ctx*2+1] = max(cnt[ctx*2+1] >> 1, 1)
            ctx = ((ctx << 1) | 1) & 3

    def literal(sym):
        nonlocal A5C
        flag(False)
        t1 = T1.t[1]; total = t1 + A5C
        leaf = T1.t[256 + sym]
        if leaf:
            cum = T1.below(sym); arith(cum, cum + leaf, total)
        else:
            arith(t1, total, total)
            t2 = T2.t[1]; cum = T2.below(sym)
            arith(cum, cum + T2.t[256 + sym], t2)
            T2.remove(sym)
            A5C = A5C + 1 if T2.t[1] else 0
            for i in range(max(sym - 8, 0), min(sym + 8, 255)):
                if T2.t[256 + i]: T2.add(i, 1, 1000)
        T1.add(sym, 1, 1000)
        if T1.t[256 + sym] == 3:
            A5C = A5C - 1 if A5C > 1 else 1

    def match(off, ln):
        """off = distance-1, ln = length-3"""
        nonlocal A54, A58, A60
        flag(True)
        while hist > A54:
            T3.add(A58, 24, 6000); A58 += 1; A54 <<= 1
        k = 0 if off == 0 else (1 if off == 1 else off.bit_length())
        t3 = T3.t[1]; cum = T3.below(k)
        arith(cum, cum + T3.t[16 + k], t3)
        T3.add(k, 24, 6000)
        if k > 1:
            half = 1 << (k - 1); tot_u = half
            if half == (A54 >> 1): tot_u = hist - half
            v = off - half
            arith(v, v + 1, tot_u)
        # length symbol
        if ln <= 14: sym = ln; extra = None
        elif ln == 783: sym = 15; extra = None
        else: sym = 16 + ((ln - 15) >> 4); extra = (ln - 15) & 15
        l1 = L1.t[1]; total = l1 + A60
        leaf = L1.t[64 + sym]
        if leaf:
            cum = L1.below(sym); arith(cum, cum + leaf, total)
        else:
            arith(l1, total, total)
            l2 = L2.t[1]; cum = L2.below(sym)
            arith(cum, cum + L2.t[64 + sym], l2)
            L2.remove(sym)
            A60 = A60 + 8 if L2.t[1] else 0
            for i in range(max(sym - 4, 0), min(sym + 4, 63)):
                if L2.t[64 + i]: L2.add(i, 1, 6000)
        L1.add(sym, 8, 6000)
        if L1.t[64 + sym] == 24:
            r3 = 8 if A60 > 8 else A60 - 1
            A60 -= r3
        if extra is not None: arith(extra, extra + 1, 16)

    n = len(data); pos = 0
    table = {}                                  # 3-byte key -> recent positions (newest last)
    def insert(p):
        if p + 3 <= n:
            key = data[p:p+3]
            lst = table.get(key)
            if lst is None: table[key] = [p]
            else:
                lst.append(p)
                if len(lst) > chain: del lst[0]
    while pos < n:
        best_len = 0; best_off = 0
        if matches and pos + 3 <= n:
            lst = table.get(data[pos:pos+3])
            if lst:
                maxlen = min(785, n - pos)
                if maxlen == 786 - 3: pass
                for cand in reversed(lst):
                    dist = pos - cand
                    if dist > hist or dist > MAX_DIST: break
                    l = 3
                    while l < maxlen and data[cand + l] == data[pos + l]: l += 1
                    if l > best_len:
                        best_len = l; best_off = dist - 1
                        if l == maxlen: break
            if best_len == 786: best_len = 785          # ln 783 is a special symbol: keep ln <= 782
        if best_len >= 3 and not (best_len - 3 == 783):
            match(best_off, best_len - 3)
            for p in range(pos, pos + best_len): insert(p)
            pos += best_len
            hist = min(hist + best_len, MAX_DIST) if hist < MAX_DIST else hist
        else:
            literal(data[pos]); insert(pos); pos += 1
            if hist < MAX_DIST: hist += 1
    flag(False, is_eof=True)
    pending += 1
    if L < 0x4000:
        out.bit(0)
        for _ in range(pending): out.bit(1)
    else:
        out.bit(1)
        for _ in range(pending): out.bit(0)
    return b'*SK_ASC*' + out.finish()
