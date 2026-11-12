"""Resource access for the importer.

Accepted inputs:
  * .res / .edres  - a single model resource (as written by gpk.py extract)
  * .gpk           - decoded in pure Python (skasc.py), no game files needed
  * .dec           - an already decoded container
"""
import struct, subprocess, os, tempfile

def _split_level(buf):
    """U_LevelNN.bin: 3 sub-containers; keys are sub*1000+slot."""
    res = {}
    for si in range(3):
        so = struct.unpack('>I', buf[12 + si*8: 16 + si*8])[0]
        n, ver = struct.unpack('>II', buf[so + 4: so + 12])
        for i in range(n):
            off, size = struct.unpack('>II', buf[so + 12 + i*8: so + 20 + i*8])
            if off and size and so + off + size <= len(buf):
                res[si*1000 + i] = buf[so + off: so + off + size]
    return res

def _is_level(buf):
    if len(buf) < 0x30 or struct.unpack('>I', buf[4:8])[0] != 3: return False
    so = struct.unpack('>I', buf[12:16])[0]
    return so + 12 <= len(buf) and struct.unpack('>I', buf[so + 8: so + 12])[0] == 0x6B5

def split(buf):
    if _is_level(buf): return _split_level(buf)
    if struct.unpack('>I', buf[4:8])[0] == 3000 or struct.unpack('>I', buf[:4])[0] == 3000:
        return _split_scr(buf)
    tot, cnt, ver = struct.unpack('>III', buf[:12])
    res = {}
    for i in range(cnt):
        off, size = struct.unpack('>II', buf[12 + i*8: 20 + i*8])
        if off or size: res[i] = buf[off:off + size]
    # cinematic packs (cinNNNN.bin): a slot may itself be a container (offsets relative to that slot)
    for i in list(res):
        sub = res[i]
        if len(sub) > 12 and struct.unpack('>I', sub[8:12])[0] == 0x6B5 and 0 < struct.unpack('>I', sub[4:8])[0] < 4096:
            n = struct.unpack('>I', sub[4:8])[0]
            for j in range(n):
                if 12 + j*8 + 8 > len(sub): break
                o, sz = struct.unpack('>II', sub[12 + j*8: 20 + j*8])
                if o and sz and o + sz <= len(sub): res[i*1000 + j] = sub[o:o + sz]
            del res[i]
    return res

def _split_scr(buf):
    cnt = struct.unpack('>I', buf[:4])[0]
    res = {}
    for i in range(cnt):
        off, size = struct.unpack('>II', buf[8 + i*8: 16 + i*8])
        if off or size: res[i] = buf[off:off + size]
    return res

def decode_gpk(path):
    data = open(path, 'rb').read()
    if data[:8] == b'*SK_ASC*':
        from . import skasc
        return skasc.decode(data)
    return data            # already a decoded container

def load_resources(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.gpk', '.bin'):
        return split(decode_gpk(path))
    if ext == '.dec':
        return split(open(path, 'rb').read())
    return {0: open(path, 'rb').read()}
