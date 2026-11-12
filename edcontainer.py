"""Eternal Darkness container reader/writer (decoded, i.e. after SK_ASC).

One layout covers U_LevelNN.bin, NPC .gpk and cinNNNN.bin ("flat" container, possibly nested):
    u32 first word (top level: file_len-4)     u32 nslots     u32 0x6B5
    nslots x (u32 offset, u32 size)            offsets relative to the container start; (0,0) = empty slot
    resources at 32-byte aligned offsets in ascending order, first one at ceil32(8*nslots + 8): when that is
    8*nslots + 8 itself the table's last size word overlaps the first 4 bytes of the first resource (packer quirk).
    The size of a container = end of its last resource (unpadded).  Padding bytes are garbage in the originals.
A resource can itself be a flat container: U_Level = 3 nested containers (key sub*1000 + slot), cinNNNN.bin slot 0
(key slot*1000 + j).  Keys are the same as gpkdec.split.

repack(buf, {key: new_bytes}) -> new container.  With no replacements the result is byte-identical.
"""
import struct

MAGIC = 0x6B5


def _c32(x): return (x + 31) & ~31


def _u32(b, o): return struct.unpack('>I', b[o:o+4])[0]


def flat_entries(buf):
    n = _u32(buf, 4)
    if len(buf) < 12 or _u32(buf, 8) != MAGIC or not (0 < n < 100000): raise ValueError("not a flat container")
    return n, [struct.unpack('>II', buf[12 + i*8: 20 + i*8]) for i in range(n)]


def _is_flat(data):
    try:
        n = _u32(data, 4)
        return len(data) > 12 and _u32(data, 8) == MAGIC and 0 < n < 100000 and 12 + 8 * n <= len(data) + 4
    except Exception:
        return False


def _repack_flat(buf, repl, first_word_keep=True):
    """repl: key (slot or slot*1000+j) -> bytes.  Returns the container bytes (unpadded)."""
    n, ent = flat_entries(buf)
    nested = {}; direct = {}
    is_cont = {i for i in range(n) if ent[i][1] and _is_flat(buf[ent[i][0]: ent[i][0] + ent[i][1]])}
    for k, v in repl.items():
        if k >= 1000:
            if k // 1000 not in is_cont: raise KeyError("slot %d is not a container" % (k // 1000))
            nested.setdefault(k // 1000, {})[k % 1000] = v
        elif k < n and ent[k][1] and k not in is_cont: direct[k] = v
        elif 0 in is_cont: nested.setdefault(0, {})[k] = v          # children of slot 0 are numbered 0*1000 + j
        else: raise KeyError("no such resource: %d" % k)
    order = sorted(range(n), key=lambda i: (ent[i][0] if ent[i][1] else 1 << 60, i))
    items = {}                                    # slot -> (data, original_padding)
    for i in range(n):
        off, size = ent[i]
        if not size: items[i] = (b'', b''); continue
        data = buf[off: off + size]; pad = buf[off + size: _c32(off + size)]
        if i in direct:
            if len(direct[i]) != len(data): pad = b''          # keep the original padding bytes when the size is unchanged
            data = direct[i]
        elif i in nested: data = _repack_flat(data, nested[i])      # padding after a nested container is kept too
        items[i] = (data, pad)
    pos = _c32(8 * n + 8); table = {}; chunks = []
    for slot in order:
        data, pad = items[slot]
        if not data: table[slot] = (0, 0); continue
        table[slot] = (pos, len(data)); chunks.append((pos, data, pad)); pos = _c32(pos + len(data))
    end = max([o + z for o, z in table.values() if z] or [12 + 8 * n])
    out = bytearray(max(end, 12 + 8 * n))
    for o, d, pad in chunks:
        out[o:o+len(d)] = d
        k = min(len(pad), len(out) - (o + len(d)))
        if k > 0: out[o+len(d): o+len(d)+k] = pad[:k]
    struct.pack_into('>III', out, 0, _u32(buf, 0), n, MAGIC)
    for i in range(n): struct.pack_into('>II', out, 12 + i*8, *table[i])      # written last: may overlap the first resource
    return bytes(out)


def repack(buf, repl):
    """Return the container with the given resources replaced (keys as in gpkdec.split)."""
    buf = bytes(buf)
    # nested containers are found by key range: any key >= 1000 addresses slot//1000 as a container
    n, ent = flat_entries(buf)
    end_src = max([o + z for o, z in ent if z] or [0])
    out = bytearray(_repack_flat(buf, repl)) + buf[end_src:]
    struct.pack_into('>I', out, 0, len(out) - 4)
    return bytes(out)


def _locate(buf, key):
    """(absolute position of the (offset, size) table entry, absolute start of the container that owns it)"""
    n, ent = flat_entries(buf)
    is_cont = {i for i in range(n) if ent[i][1] and _is_flat(buf[ent[i][0]: ent[i][0] + ent[i][1]])}
    if key >= 1000:
        slot, ch = key // 1000, key % 1000
        if slot not in is_cont: raise KeyError("slot %d is not a container" % slot)
        cs = ent[slot][0]
    elif key < n and ent[key][1] and key not in is_cont: return 12 + 8 * key, 0
    elif 0 in is_cont: cs, ch = ent[0][0], key
    else: raise KeyError("no such resource: %d" % key)
    return cs + 12 + 8 * ch, cs


def relocate(buf, repl):
    """Put resources that grew at the END of the file and point their table entries there.  Nothing else moves
    (every other resource keeps its offset, which matters because the game refers to them by position)."""
    out = bytearray(buf)
    for key, data in repl.items():
        epos, cs = _locate(bytes(buf), key)
        while len(out) % 32: out.append(0)
        off = len(out); out.extend(data)
        struct.pack_into('>II', out, epos, off - cs, len(data))
    while len(out) % 32: out.append(0)
    struct.pack_into('>I', out, 0, len(out) - 4)
    return bytes(out)
