import os
import re
import shutil
import bpy
import numpy as np
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ExportHelper

from . import exportcore, modelwrite, icons

# ---- shape-key name -> morph channel id (same table as the importer)
CHANNEL_NAMES = {
    0: "Viseme_MBP", 1: "Viseme_AH", 2: "Viseme_OH", 3: "Viseme_OO", 4: "Viseme_EH", 5: "Viseme_EE",
    6: "Viseme_TH", 7: "Viseme_FV", 8: "Viseme_AW", 9: "Viseme_AO", 10: "Viseme_W", 11: "Viseme_IH",
    12: "Viseme_NDL", 13: "Viseme_ER", 14: "Viseme_SH",
    15: "Blink_R", 16: "Blink_L", 17: "BrowRaise_R", 18: "BrowRaise_L", 19: "JawOpen",
    21: "Glare", 22: "Grimace", 25: "Frown", 27: "Squint_R", 28: "Squint_L",
    29: "Scared", 30: "Smirk", 31: "Displeasure",
}
SHAPE_TO_CHANNEL = {"ED_" + v: k for k, v in CHANNEL_NAMES.items()}


def channel_of(shape_name):
    if shape_name in SHAPE_TO_CHANNEL: return SHAPE_TO_CHANNEL[shape_name]
    if shape_name.startswith("ED_Expr"):
        try: return int(shape_name[7:9])
        except ValueError: pass
    return None


# --- Blender -> plain data ---

def rig_of(obj):
    p = obj.parent
    if p is not None and p.type == 'ARMATURE' and "ed2b_bones" in p: return p
    for md in obj.modifiers:
        if md.type == 'ARMATURE' and md.object is not None and "ed2b_bones" in md.object: return md.object
    return None


def _upstream_image(sock, depth=4):
    """first Image Texture feeding a shader socket (looks through a few helper nodes: mix, colour ramp, ...)"""
    if sock is None: return None
    for l in sock.links:
        n = l.from_node
        if n.type == 'TEX_IMAGE' and n.image is not None: return n.image
        if depth > 0:
            for i in n.inputs:
                im = _upstream_image(i, depth - 1)
                if im is not None: return im
    return None


def mat_images(mat):
    """(diffuse image, secondary image, [every image of the material]).  What is actually wired into the shader wins
    (Base Color -> diffuse, Specular -> secondary); the importer's node labels are the fallback."""
    if mat is None or not mat.use_nodes or mat.node_tree is None: return None, None, []
    nodes = [n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image is not None]
    diffuse = secondary = None
    bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is not None:
        diffuse = _upstream_image(bsdf.inputs.get('Base Color'))
        secondary = _upstream_image(bsdf.inputs.get('Specular IOR Level') or bsdf.inputs.get('Specular'))
    for n in nodes:
        lab = (n.label or "").lower()
        if diffuse is None and lab == "diffuse": diffuse = n.image
        elif secondary is None and lab.startswith("secondary"): secondary = n.image
    return diffuse, secondary, [n.image for n in nodes]


def _srgb_encode(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)


def image_rgba(im, size=None):
    """uint8 RGBA (h, w, 4), top row first, from any Blender image (packed, generated, or loaded from a file on disk).
    Resized (bilinear, by Blender) to `size` = (w, h) when it differs."""
    tmp = None
    if size is not None and tuple(im.size) != tuple(size):
        tmp = im.copy(); tmp.scale(int(size[0]), int(size[1])); im = tmp
    w, h = im.size
    if w == 0 or h == 0: raise exportcore.ExportError("image '%s' has no pixel data (missing file?)" % im.name)
    buf = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(buf)
    a = buf.reshape(h, w, 4).astype(np.float64)
    try:
        if getattr(im, "is_float", False) and im.colorspace_settings.name.lower().startswith("srgb"):
            a[..., :3] = _srgb_encode(np.clip(a[..., :3], 0, 1))               # float buffers are linear: store as sRGB bytes
    except Exception: pass
    if tmp is not None: bpy.data.images.remove(tmp)
    a = np.rint(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    return a[::-1].copy()


def plan_materials(obj, m, sizes, rep, cap=64):
    """Returns (slot -> prim group, {group: dict(diffuse, secondary, template)}, {texture index: rgba array}).
    The images are whatever the Blender materials use: the imported ones (edited), or images you loaded from anywhere on disk and
    plugged into a material's Base Color / Specular -- those are written into the game texture the material used."""
    explicit = {}                      # texture index -> image that a material explicitly assigns to it (disk files, new images)
    implicit = {}                      # texture index -> imported (tagged) image
    claimed = {}; mats = {}; slot_group = {}
    nmat = len(m.materials); nxt = max(nmat, 1)
    def tag(im):
        t = im.get("ed2b_tex") if im is not None else None
        return int(t) if t is not None else None
    def assign(tex, im, mat, what):
        prev = explicit.get(tex)
        if prev is not None and prev is not im:
            rep.append("WARNING: texture %d is used by several materials with different images ('%s' and '%s'): '%s' wins" % (tex, prev.name, im.name, prev.name))
            return
        explicit[tex] = im
        src = (bpy.path.abspath(im.filepath) if getattr(im, "filepath", "") else "") or ("packed/generated image")
        rep.append("material '%s': %s '%s' (%s) is written into game texture %d" % (mat.name, what, im.name, src, tex))
    for si, slot in enumerate(obj.material_slots):
        mat = slot.material
        g0 = mat.get("ed2b_group") if mat is not None else None
        g0 = int(g0) if g0 is not None else None
        orig = m.materials[g0] if (g0 is not None and g0 < nmat) else None
        d_img, s_img, every = mat_images(mat)
        d_idx = orig['diffuse'] if orig else None
        s_idx = orig['secondary'] if orig else None
        for img, kind in ((d_img, "diffuse"), (s_img, "secondary")):
            if img is None: continue
            t = tag(img); cur = d_idx if kind == "diffuse" else s_idx
            if t is not None and t in sizes: new = t                      # one of the game's own textures
            elif cur is not None and cur in sizes:
                new = cur; assign(cur, img, mat, kind + " image")          # any other image: replaces what the material used
            else:
                rep.append("WARNING: material '%s': %s image '%s' cannot be used (the material has no game texture to replace; "
                           "assign it to an imported material instead)" % (mat.name, kind, img.name)); continue
            if kind == "diffuse": d_idx = new
            else: s_idx = new
        for im in every:
            t = tag(im)
            if t is not None and t in sizes: implicit.setdefault(t, im)
        k = (g0, d_idx, s_idx)
        if k in claimed: g = claimed[k]
        else:
            same = [kk for kk in claimed if kk[0] == g0 and g0 is not None]
            if g0 is not None and not same and g0 not in claimed.values(): g = g0
            elif nxt >= cap:
                g = g0 if g0 is not None else 0
                rep.append("WARNING: material '%s' needs a new entry but the model has no room for more than %d: it shares group %d (textures of that group are used)" % (mat.name if mat else "?", cap, g))
                claimed[k] = g; slot_group[si] = g; continue
            else:
                g = nxt; nxt += 1
                rep.append("material '%s' gets a new material entry (group %d)" % (mat.name if mat else "?", g))
            claimed[k] = g
            mats[g] = dict(diffuse=d_idx, secondary=s_idx, template=g0 if g0 is not None else 0)
        slot_group[si] = g
    images = {}
    for tex, im in list(implicit.items()) + list(explicit.items()):          # explicit assignments override imported images
        try: images[tex] = image_rgba(im, sizes[tex])
        except Exception as ex: rep.append("WARNING: could not read image '%s' (%s)" % (im.name, ex))
    return slot_group, mats, images


def vertex_normals(mesh, nv):
    try:
        nl = len(mesh.loops); arr = np.empty(nl * 3, np.float32)
        if hasattr(mesh, "corner_normals"): mesh.corner_normals.foreach_get('vector', arr)
        else:
            mesh.calc_normals_split(); mesh.loops.foreach_get('normal', arr)
        vi = np.empty(nl, np.int32); mesh.loops.foreach_get('vertex_index', vi)
        acc = np.zeros((nv, 3)); np.add.at(acc, vi, arr.reshape(-1, 3).astype(np.float64))
    except Exception:
        arr = np.empty(nv * 3, np.float32); mesh.vertices.foreach_get('normal', arr); acc = arr.reshape(-1, 3).astype(np.float64)
    return acc


def extract_geo(obj, m, scale, rep):
    """Blender mesh -> modelwrite geometry description (game units)."""
    mesh = obj.data; nv = len(mesh.vertices)
    rig = rig_of(obj)
    M = np.array(obj.matrix_world, np.float64)
    if rig is not None: M = np.linalg.inv(np.array(rig.matrix_world, np.float64)) @ M
    R = M[:3, :3]; identity = np.allclose(M, np.eye(4), atol=1e-6)
    co = np.empty(nv * 3, np.float32); mesh.vertices.foreach_get('co', co); co = co.reshape(-1, 3).astype(np.float64)
    if not identity: rep.append("object transform is applied to the exported vertices")
    pos = ((co @ R.T + M[:3, 3]) if not identity else co) / scale
    nrm = vertex_normals(mesh, nv)
    if not identity: nrm = nrm @ np.linalg.inv(R)
    vid = np.full(nv, -1, np.int64)
    attr = mesh.attributes.get("ed2b_vid") if hasattr(mesh, "attributes") else None
    if attr is not None:
        v = np.empty(nv, np.int32); attr.data.foreach_get("value", v); vid = v.astype(np.int64) - 1                    # stored as index + 1; 0 = created in Blender
    elif nv == len(m.pos):
        vid = np.arange(nv)
    # triangles
    mesh.calc_loop_triangles(); nt = len(mesh.loop_triangles)
    tv = np.empty(nt * 3, np.int32); mesh.loop_triangles.foreach_get('vertices', tv); tv = tv.reshape(-1, 3).astype(np.int64)
    tl = np.empty(nt * 3, np.int32); mesh.loop_triangles.foreach_get('loops', tl); tl = tl.reshape(-1, 3).astype(np.int64)
    tm = np.empty(nt, np.int32); mesh.loop_triangles.foreach_get('material_index', tm)
    uvl = mesh.uv_layers.active or (mesh.uv_layers[0] if len(mesh.uv_layers) else None)
    if uvl is None: raise exportcore.ExportError("Object '%s' has no UV map" % obj.name)
    uv = np.empty(len(mesh.loops) * 2, np.float32); uvl.data.foreach_get('uv', uv); uv = uv.reshape(-1, 2).astype(np.float64)
    tuv = uv[tl].copy(); tuv[..., 1] = 1.0 - tuv[..., 1]            # V is flipped like on import
    # skin
    skin = None
    if rig is not None and len(obj.vertex_groups):
        names = list(rig["ed2b_bones"]); bone_of = {n: i for i, n in enumerate(names)}
        gb = {vg.index: bone_of.get(vg.name) for vg in obj.vertex_groups}
        skin = []; unweighted = 0
        for v in mesh.vertices:
            s = [(gb[g.group], float(g.weight)) for g in v.groups if gb.get(g.group) is not None and g.weight > 0]
            skin.append(s)
            if not s: unweighted += 1
        if unweighted: rep.append("%d vertices have no bone weights: they take the weights of the nearest original vertex" % unweighted)
    # morph targets from the ED_* shape keys
    morph = None
    sk = mesh.shape_keys
    if sk is not None and len(sk.key_blocks) > 1:
        ref = sk.reference_key; r = np.empty(nv * 3, np.float32); ref.data.foreach_get('co', r); r = r.reshape(-1, 3).astype(np.float64)
        morph = {}
        for kb in sk.key_blocks:
            if kb == ref: continue
            ch = channel_of(kb.name)
            if ch is None: continue
            c = np.empty(nv * 3, np.float32); kb.data.foreach_get('co', c)
            d = (c.reshape(-1, 3).astype(np.float64) - r)
            d = (d @ R.T if not identity else d) / scale
            morph[ch] = d
        if not morph: morph = None
    return dict(pos=pos, nrm=nrm, vid=vid, skin=skin, tris=tv, tuv=tuv, tmat=tm, morph=morph, mats=None, tgrp=None)


TITLE = "EternalDarkness2Blender2EternalDarkness"

# --- operators ---

_NAME_RE = re.compile(r"_(\d{1,4})(?:\.\d+)?$")


def _key_of(obj):
    """Resource key of an imported model mesh: the importer's tag, or (older imports) the number at the end of the name."""
    if "ed2b_key" in obj: return int(obj["ed2b_key"])
    mt = _NAME_RE.search(obj.name)
    return int(mt.group(1)) if mt else None


def _targets(context):
    """Every model mesh in the blend file (selection is ignored)."""
    pool = list(bpy.data.objects) if hasattr(bpy.data, "objects") else list(context.scene.objects)
    return [o for o in pool if o.type == 'MESH' and _key_of(o) is not None]


class _ExportBase(ExportHelper):
    filename_ext = ".bin"
    filter_glob: StringProperty(default="*.bin;*.gpk", options={'HIDDEN'})
    source: StringProperty(name="Original file", subtype='FILE_PATH',
                           description="The original .bin/.gpk the model was imported from (read for everything that is not edited)")
    with_geometry = False

    def invoke(self, context, event):
        objs = _targets(context)
        if objs:
            src = next((o.get("ed2b_srcpath", "") for o in objs if o.get("ed2b_srcpath")), "")
            if src and not self.source: self.source = src
            if src and not self.filepath:
                b, e = os.path.splitext(src); self.filepath = b + "_edited" + e
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        icons.dialog_header(self.layout, TITLE)
        self.layout.prop(self, "source")

    def execute(self, context):
        objs = _targets(context)
        if not objs:
            self.report({'ERROR'}, "No model meshes found in this blend file (import a model first)"); return {'CANCELLED'}
        src_path = bpy.path.abspath(self.source) if self.source else next((o.get("ed2b_srcpath", "") for o in objs if o.get("ed2b_srcpath")), "")
        if not src_path or not os.path.isfile(src_path):
            self.report({'ERROR'}, "Set 'Original file' to the .bin/.gpk the model was imported from"); return {'CANCELLED'}
        try:
            src = exportcore.Source(src_path)
            rep = []; jobs = {}
            for obj in objs:
                key = _key_of(obj)
                if "ed2b_srcpath" in obj and os.path.basename(obj["ed2b_srcpath"]) != os.path.basename(src_path):
                    rep.append("skipped '%s' (imported from another file)" % obj.name); continue
                m = src.model(key); sizes = src.texture_sizes(key)
                rig = rig_of(obj)
                scale = float(obj.get("ed2b_scale", rig.get("ed2b_scale", 0.01) if rig else 0.01))
                slot_group, mats, images = plan_materials(obj, m, sizes, rep, modelwrite.material_capacity(src.res[key]))
                job = dict(images=images, geo=None)
                if self.with_geometry:
                    geo = extract_geo(obj, m, scale, rep)
                    tm = geo.pop("tmat"); lut = np.array([slot_group.get(i, 0) for i in range(max(len(obj.material_slots), 1))] or [0])
                    geo["tgrp"] = lut[np.clip(tm, 0, len(lut) - 1)] if len(lut) else np.zeros(len(tm), np.int64)
                    geo["mats"] = mats
                    job["geo"] = geo
                if key in jobs: rep.append("WARNING: two objects use resource %d: the last one wins" % key)
                jobs[key] = job
            if not jobs:
                self.report({'ERROR'}, "Nothing to export"); return {'CANCELLED'}
            out = self.filepath
            if os.path.abspath(out) == os.path.abspath(src_path) and not os.path.exists(src_path + ".bak"):
                shutil.copyfile(src_path, src_path + ".bak"); rep.append("backup of the original: %s.bak" % os.path.basename(src_path))
            rep += exportcore.write(src, jobs, out)
        except (exportcore.ExportError, modelwrite.WriteError) as e:
            self.report({'ERROR'}, str(e)); return {'CANCELLED'}
        except Exception as e:
            import traceback; traceback.print_exc()
            self.report({'ERROR'}, "Export failed: %s" % e); return {'CANCELLED'}
        for l in rep: print("ED export:", l)
        self.report({'INFO'}, rep[-1] + (" (see the system console for details)" if len(rep) > 1 else ""))
        return {'FINISHED'}


class ExportEDTextures(_ExportBase, bpy.types.Operator):
    bl_idname = "export_scene.eternal_darkness_textures"
    bl_label = "Textures only (.bin)"
    bl_description = "Write the edited textures of all imported model(s) in this file into a copy of the original .bin; geometry is not touched"
    with_geometry = False


class ExportEDModel(_ExportBase, bpy.types.Operator):
    bl_idname = "export_scene.eternal_darkness_model"
    bl_label = "Model (.bin)"
    bl_description = "Rebuild all imported model(s) in this file from the Blender mesh, materials, weights, shape keys and textures"
    with_geometry = True


def menu(self, context):
    self.layout.operator(ExportEDModel.bl_idname, text="EternalDarkness2Blender2EternalDarkness: Model (.bin)", **icons.kw("menu"))
    self.layout.operator(ExportEDTextures.bl_idname, text="EternalDarkness2Blender2EternalDarkness: Textures only (.bin)", **icons.kw("menu"))


classes = (ExportEDTextures, ExportEDModel)
