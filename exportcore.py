"""File-level export logic (no bpy): load the original pack, apply per-resource edits, repack, recompress."""
import os
from . import skasc, skasc_enc, gpkdec, edcontainer, edmodel, edtex, modelwrite


class ExportError(Exception):
    pass


class Source:
    def __init__(self, path):
        if not os.path.isfile(path): raise ExportError("Original file not found: %s" % path)
        raw = open(path, 'rb').read()
        self.path = path; self.compressed = raw[:8] == b'*SK_ASC*'
        self.dec = skasc.decode(raw) if self.compressed else raw
        try:
            edcontainer.flat_entries(self.dec); self.container = True
        except ValueError:
            self.container = False
        self.res = gpkdec.split(self.dec) if self.container else {0: self.dec}
        self._models = {}

    def model(self, key):
        if key not in self.res: raise ExportError("%s has no resource %d (models are in slots %s)" % (os.path.basename(self.path), key, self.model_keys()))
        if key not in self._models: self._models[key] = edmodel.NpcModel(self.res[key])
        return self._models[key]

    def model_keys(self):
        out = []
        for k, r in self.res.items():
            try: edmodel.NpcModel(r); out.append(k)
            except Exception: pass
        return sorted(out)

    def texture_sizes(self, key):
        return {t['index']: (t['w'], t['h']) for t in edtex.texture_table(self.res[key]) if t['fmt'] == 14}


def write(src, jobs, out_path):
    """jobs: {key: dict(geo=<modelwrite geo> | None, images={texture index: uint8 RGBA (h, w, 4)})}.  Returns report lines."""
    rep = []; repl = {}
    for key, job in sorted(jobs.items()):
        r = src.res[key]; m = src.model(key)
        if job.get('geo') is not None:
            r, lines = modelwrite.rebuild(r, m, job['geo']); rep += ["[%d] %s" % (key, l) for l in lines]
        if job.get('images'):
            r, changed = edtex.patch_textures(r, job['images'])
            rep.append("[%d] textures re-encoded: %s" % (key, changed if changed else "none changed"))
        repl[key] = r
    for k, v in repl.items():
        if len(v) != len(src.res[k]): raise ExportError("internal error: resource %d changed size" % k)
    if src.container: new = edcontainer.repack(src.dec, repl)
    else: new = next(iter(repl.values()))
    data = skasc_enc.encode(new) if src.compressed else new
    with open(out_path, 'wb') as f: f.write(data)
    rep.append("wrote %s (%d bytes%s)" % (os.path.basename(out_path), len(data), ", SK_ASC compressed" if src.compressed else ""))
    return rep
