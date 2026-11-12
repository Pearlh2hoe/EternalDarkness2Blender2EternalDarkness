"""Importer for Eternal Darkness room files (rmedNNNN.bpe): geometry, materials, cameras, regions and markers."""
import os
import math
import bpy
import numpy as np
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector, Quaternion

from . import bperoom, icons


# --- small helpers ---
def _collection(ctx, name, parent=None):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        (parent or ctx.scene.collection).children.link(coll)
    return coll


def _png_bytes(rgba):
    """Minimal PNG encoder (RGBA8, top row first) - no PIL needed."""
    import zlib, struct
    h, w = rgba.shape[:2]
    raw = np.concatenate([np.zeros((h, 1), np.uint8), np.ascontiguousarray(rgba, np.uint8).reshape(h, w * 4)], 1).tobytes()
    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def _image_from_rgba(name, rgba, noncolor=False):
    """Create the image by loading a PNG (reliable on every Blender version), then pack it into the .blend."""
    import tempfile
    h, w = rgba.shape[:2]
    try:
        d = bpy.app.tempdir or tempfile.gettempdir()
        path = os.path.join(d, "ed2b_%s.png" % name)
        with open(path, 'wb') as f: f.write(_png_bytes(rgba))
        im = bpy.data.images.load(path)
        im.name = name
        if tuple(im.size) != (w, h): raise RuntimeError("loaded size %s != %s" % (tuple(im.size), (w, h)))
        try: im.colorspace_settings.name = 'Non-Color' if noncolor else 'sRGB'   # lightmap mode blends raw 8-bit values like the GameCube
        except Exception: pass
        try: im.alpha_mode = 'STRAIGHT'
        except Exception: pass
        im.pack()
        try: os.remove(path)
        except Exception: pass
    except Exception as ex:
        # fall back to filling the pixels directly
        im = bpy.data.images.new(name, w, h, alpha=True)
        if noncolor:
            try: im.colorspace_settings.name = 'Non-Color'
            except Exception: pass
        im.pixels = (rgba[::-1].astype(np.float32) / 255.0).ravel().tolist()
        try: im.pack()
        except Exception: pass
    return im


def _mix(nt, blend, a=None, b=None, x=0):
    """Colour mix node for 2.8-3.x (MixRGB) and 4.x (Mix).  a / b = socket or (r,g,b,a).  Returns the output socket."""
    try:
        n = nt.nodes.new('ShaderNodeMixRGB')
        n.blend_type = blend; n.inputs['Fac'].default_value = 1.0
        ia, ib, out = n.inputs['Color1'], n.inputs['Color2'], n.outputs['Color']
    except Exception:
        n = nt.nodes.new('ShaderNodeMix')
        n.data_type = 'RGBA'; n.blend_type = blend; n.inputs[0].default_value = 1.0
        ia, ib, out = n.inputs[6], n.inputs[7], n.outputs[2]
    n.location = (x, 150)
    for sock, v in ((ia, a), (ib, b)):
        if v is None: continue
        if isinstance(v, tuple): sock.default_value = v
        else: nt.links.new(v, sock)
    return out


def build_room_material(name, diffuse, lm1=None, lm2=None, has_alpha=False, view_color=None, cull=True, gain=1.0, ambient=0.1, lm0=None, mask=None):
    """Principled BSDF material.  Base colour = diffuse, or with lightmaps diffuse x (lightmaps x gain + ambient).
    Textures with transparent texels (CMPR 1-bit alpha) drive the alpha channel."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (900, 0)

    def tex(image, layer, y, label, repeat):
        uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = layer; uvn.location = (-700, y)
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = image; t.label = label; t.location = (-450, y)
        t.interpolation = 'Linear'; t.extension = 'REPEAT' if repeat else 'EXTEND'
        nt.links.new(uvn.outputs['UV'], t.inputs['Vector'])
        return t

    t = tex(diffuse, "UVMap", 300, "diffuse", True) if diffuse is not None else None
    l1 = tex(lm1, "Lightmap", 0, "lightmap 1", False) if lm1 is not None else None
    l2 = tex(lm2, "Lightmap2", -300, "lightmap 2", False) if (lm2 is not None and lm2 is not lm1) else None
    l0 = tex(lm0, "UVMap", 600, "baked layer 0", False) if lm0 is not None else None     # lightmap-class texture in the main slot (own UVs)
    lsum = None
    for ln in (l0, l1, l2):
        if ln is None: continue
        lsum = ln.outputs['Color'] if lsum is None else _mix(nt, 'ADD', lsum, ln.outputs['Color'], -150)
    if lsum is not None:
        if gain != 1.0: lsum = _mix(nt, 'MULTIPLY', lsum, (gain, gain, gain, 1.0), 0)
        if ambient > 0: lsum = _mix(nt, 'ADD', lsum, (ambient, ambient, ambient, 1.0), 150)
    if t is not None and lsum is not None: col = _mix(nt, 'MULTIPLY', t.outputs['Color'], lsum, 300)
    elif t is not None: col = t.outputs['Color']
    else: col = lsum
    bs = nt.nodes.new('ShaderNodeBsdfPrincipled'); bs.location = (600, 0)
    for sock, val in (('Roughness', 1.0), ('Specular IOR Level', 0.0), ('Specular', 0.0)):
        try: bs.inputs[sock].default_value = val
        except Exception: pass
    if col is not None and lsum is not None:
        # raw (gamma-space) product -> linear, so the standard view transform shows what the game shows
        g = nt.nodes.new('ShaderNodeGamma'); g.location = (450, 150); g.inputs['Gamma'].default_value = 2.2
        nt.links.new(col, g.inputs['Color']); col = g.outputs['Color']
        # the baked light is the shading: feed the same colour to the BSDF's own emission input so the render needs no lamps
        for sock in ('Emission Color', 'Emission'):
            if sock in bs.inputs:
                nt.links.new(col, bs.inputs[sock]); break
        try: bs.inputs['Emission Strength'].default_value = 1.0
        except Exception: pass
    if col is not None: nt.links.new(col, bs.inputs['Base Color'])
    alpha = t.outputs['Alpha'] if (t is not None and has_alpha) else None
    if mask is not None:
        # separate alpha-channel texture of the surface (grey image: white = opaque)
        mt = tex(mask, "UVMap", 450, "alpha mask", True)
        bw = nt.nodes.new('ShaderNodeRGBToBW'); bw.location = (-200, 450)
        nt.links.new(mt.outputs['Color'], bw.inputs['Color'])
        ma = bw.outputs['Val']
        if alpha is not None:
            mm = nt.nodes.new('ShaderNodeMath'); mm.operation = 'MULTIPLY'; mm.location = (-50, 450)
            nt.links.new(alpha, mm.inputs[0]); nt.links.new(ma, mm.inputs[1]); ma = mm.outputs['Value']
        alpha = ma
    if alpha is not None: nt.links.new(alpha, bs.inputs['Alpha'])
    nt.links.new(bs.outputs['BSDF'], out.inputs['Surface'])
    if view_color is not None:
        try: mat.diffuse_color = (view_color[0], view_color[1], view_color[2], 1.0)
        except Exception: pass
    try: mat.use_backface_culling = bool(cull)
    except Exception: pass
    if alpha is not None:
        # 1-bit alpha: hard cut-out in every Blender version (4.2+ uses surface_render_method, older ones blend_method)
        for attr, val in (('surface_render_method', 'DITHERED'), ('blend_method', 'CLIP'), ('shadow_method', 'CLIP'),
                          ('alpha_threshold', 0.5), ('show_transparent_back', False), ('use_transparency_overlap', False)):
            try: setattr(mat, attr, val)
            except Exception: pass
    return mat


# --- geometry ---
def build_room_mesh(name, room, mats, scale, use_normals):
    """Triangle mesh built with from_pydata + per-loop UV assignment (the same calls the model importer uses)."""
    pos, nrm, uv = room['pos'], room['nrm'], room['uv']
    tris = room['tris']
    cs = room['corners'][tris.ravel()]                 # one loop per triangle corner, loop order = tri*3 + k
    ntri = len(tris)
    mesh = bpy.data.meshes.new(name)
    verts = [(float(p[0]) * scale, float(p[1]) * scale, float(p[2]) * scale) for p in pos]
    faces = [(int(cs[i * 3, 0]), int(cs[i * 3 + 1, 0]), int(cs[i * 3 + 2, 0])) for i in range(ntri)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for m in mats: mesh.materials.append(m)
    ms = room['tri_surf']
    for i, poly in enumerate(mesh.polygons):
        poly.material_index = int(ms[i])
        poly.use_smooth = True
    nuv = len(uv)
    def layer(lname, idx):
        idx = np.clip(idx, 0, nuv - 1)
        a = uv[idx]
        l = mesh.uv_layers.new(name=lname)
        d = l.data
        for k in range(len(a)):                        # V flipped for Blender
            d[k].uv = (float(a[k, 0]), 1.0 - float(a[k, 1]))
        return a
    layer("UVMap", cs[:, 2])
    layer("Lightmap", cs[:, 3])
    # surfaces that have a single lightmap use junk in the second slot -> fall back to lightmap 1's coordinates
    layer("Lightmap2", np.where(cs[:, 4] < nuv, cs[:, 4], cs[:, 3]))
    if use_normals:
        try:
            if hasattr(mesh, "use_auto_smooth"): mesh.use_auto_smooth = True       # Blender < 4.1
            ln = nrm[np.clip(cs[:, 1], 0, len(nrm) - 1)].astype(np.float64)
            ln /= np.linalg.norm(ln, axis=1, keepdims=True) + 1e-9
            mesh.normals_split_custom_set([tuple(map(float, n)) for n in ln])
        except Exception:
            pass                                  # keep the default normals
    return mesh


def build_collision_mesh(name, room, scale):
    polys, _, _ = room['collision']
    faces = [(k, p) for k, (kind, p) in enumerate(polys) if len(p) >= 3]
    pos = room['pos']
    mesh = bpy.data.meshes.new(name)
    verts = [tuple(float(c) * scale for c in v) for v in pos]
    mesh.from_pydata(verts, [], [tuple(int(i) for i in p) for k, p in faces])
    mesh.update()
    try:
        at = mesh.attributes.new("ed_kind", 'INT', 'FACE')
        at.data.foreach_set("value", np.array([polys[k][0] for k, p in faces], np.int32))
    except Exception:
        pass
    return mesh


# --- objects ---
_MARKER_STYLE = (            # (name prefix, empty display type, size in game units)
    ("Entry", 'ARROWS', 30), ("Axis", 'ARROWS', 20), ("Gravity", 'SINGLE_ARROW', 30), ("Spawn", 'PLAIN_AXES', 20),
    ("Blood", 'SPHERE', 8), ("Pool", 'SPHERE', 12), ("Sanity", 'SPHERE', 10),
)


def _marker_style(name):
    for pre, disp, size in _MARKER_STYLE:
        if name.startswith(pre): return disp, size
    return 'PLAIN_AXES', 15


def build_objects(ctx, room, base, scale):
    objs = room['objects']
    n = {'cameras': 0, 'regions': 0, 'markers': 0}
    ccam = _collection(ctx, base + "_Cameras")
    creg = _collection(ctx, base + "_Regions")
    cmrk = _collection(ctx, base + "_Markers")
    first_cam = None
    for c in objs.get('cameras', []):
        cam = bpy.data.cameras.new(c['name'])
        try:
            cam.sensor_fit = 'VERTICAL'
            cam.angle = math.radians(c['fov'])
        except Exception: pass
        cam.clip_end = max(c['far'] * scale, 200.0); cam.clip_start = 0.05
        o = bpy.data.objects.new(c['name'], cam)
        o.location = tuple(float(v) * scale for v in c['pos'])
        d = Vector(c['dir'])
        if d.length > 1e-6:
            o.rotation_mode = 'QUATERNION'
            # the stored vector is a rotation AXIS and the float after it the ANGLE (radians): camera-to-world = rotation(axis, -angle),
            # in the usual GL camera frame (looks down -Z, up +Y) - exactly Blender's camera frame.
            ang = float(c.get('angle', 0.0))
            if abs(ang) > 1e-6 and ang == ang:
                ax = d.normalized(); s = math.sin(-ang * 0.5)
                o.rotation_quaternion = Quaternion((math.cos(ang * 0.5), ax.x * s, ax.y * s, ax.z * s))
            else:
                o.rotation_quaternion = d.to_track_quat('-Z', 'Y')
        o["ed2b_hash"] = "%08X" % c['hash']
        ccam.objects.link(o); n['cameras'] += 1
        if first_cam is None: first_cam = o
    if first_cam is not None and ctx.scene.camera is None:
        ctx.scene.camera = first_cam
    for r in objs.get('regions', []):
        o = bpy.data.objects.new(r['name'], None)
        mn, mx = Vector(r['min']), Vector(r['max'])
        o.empty_display_type = 'CUBE'; o.empty_display_size = 1.0
        o.location = tuple(float(v) * scale for v in (mn + mx) * 0.5)
        o.scale = tuple(max(float(v) * scale * 0.5, 1e-4) for v in (mx - mn))
        o.show_wire = True
        o["ed2b_hash"] = "%08X" % r['hash']
        creg.objects.link(o); n['regions'] += 1
    for m in objs.get('markers', []):
        o = bpy.data.objects.new(m['name'], None)
        disp, size = _marker_style(m['name'])
        o.empty_display_type = disp; o.empty_display_size = size * scale
        o.location = tuple(float(v) * scale for v in m['pos'])
        sign = -1.0 if m['axis'] < 0 else 1.0
        o.rotation_euler = (0.0, 0.0, sign * float(m['angle']))
        o["ed2b_hash"] = "%08X" % m['hash']; o["ed2b_kind"] = int(m['kind']); o["ed2b_flags"] = int(m['flags'])
        cmrk.objects.link(o); n['markers'] += 1
    return n


# --- operator ---
class ImportEDRoom(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.eternal_darkness_room"
    bl_label = "Import Eternal Darkness Room"
    filename_ext = ".bpe"
    filter_glob: StringProperty(default="*.bpe", options={'HIDDEN'})
    scale: FloatProperty(name="Scale", default=0.01, min=1e-6, max=100.0, precision=4,
                         description="Import scale (game units -> Blender units); 0.01 matches the character importer")
    lightmaps: BoolProperty(name="Lightmaps", default=True,
                            description="Multiply the baked lightmaps of every surface into the base colour (own UV maps, bilinear): diffuse x (lightmaps x gain + ambient)")
    lm_gain: FloatProperty(name="Lightmap gain", default=1.0, min=0.0, max=32.0,
                           description="Brightness multiplier on the lightmaps (the GameCube lightmaps are stored dark)")
    ambient: FloatProperty(name="Ambient", default=0.1, min=0.0, max=4.0,
                           description="Constant light added to the lightmaps; 0 = lightmaps only")
    cull: BoolProperty(name="Hide back faces", default=True,
                       description="Materials cull back faces, so looking from outside you see into the room")
    game_normals: BoolProperty(name="Use game normals", default=True, description="Apply the vertex normals stored in the room")
    objects: BoolProperty(name="Cameras, regions and markers", default=True,
                          description="Create the room's cameras, trigger regions (boxes) and markers (entry/spawn/blood/...) as objects")
    collision: BoolProperty(name="Collision polygons", default=False,
                            description="Also import the structural/collision polygon list as a hidden wire mesh")
    reset_scene: BoolProperty(name="Reset scene before import", default=True,
                              description="Delete all objects and unused data (meshes, materials, images) first")

    def draw(self, ctx):
        layout = self.layout
        icons.dialog_header(layout, "EternalDarkness2Blender2EternalDarkness")
        for p in ("scale", "lightmaps", "lm_gain", "ambient", "cull", "game_normals", "objects", "collision", "reset_scene"):
            layout.prop(self, p)

    def execute(self, ctx):
        try:
            with open(self.filepath, 'rb') as f: raw = f.read()
            room = bperoom.parse(raw)
        except Exception as e:
            self.report({'ERROR'}, "Room import failed: %s" % e); return {'CANCELLED'}
        if self.reset_scene:
            if ctx.object and ctx.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
            for coll in (bpy.data.meshes, bpy.data.cameras, bpy.data.materials, bpy.data.images):
                for d in list(coll):
                    if d.users == 0: coll.remove(d)
        if self.lightmaps:
            try: ctx.scene.view_settings.view_transform = 'Standard'    # Filmic/AgX would crush the dark lightmaps
            except Exception: pass
        base = os.path.splitext(os.path.basename(self.filepath))[0]
        images, rgba = {}, {}
        def get_img(i):
            if i is None: return None
            if i not in images:
                rgba[i] = bperoom.texture_rgba(room, i)
                im = _image_from_rgba("%s_tex%03d" % (base, i), rgba[i], self.lightmaps)
                im["ed2b_room_tex"] = int(i)
                images[i] = im
            return images[i]
        mats = []
        lm_used = set()
        for s in room['surfaces']:
            for k in ('lm1', 'lm2'):
                if s[k] is not None: lm_used.add(s[k])
        ifl_names = room.get('ifl') or []
        ifl_frames = room.get('ifl_frames') or []
        masks = {}
        def is_grey(i):
            a = rgba.get(i)
            if a is None: a = rgba[i] = bperoom.texture_rgba(room, i)
            c = a[..., :3].astype(np.int16)
            return int(np.abs(c[..., 0] - c[..., 1]).max()) < 10 and int(np.abs(c[..., 1] - c[..., 2]).max()) < 10
        def get_mask(i):
            if i not in masks:
                a = rgba.get(i)
                if a is None: a = rgba[i] = bperoom.texture_rgba(room, i)
                im = _image_from_rgba("%s_tex%03d_mask" % (base, i), a, True)      # Non-Color: the grey value is the alpha
                im["ed2b_room_tex"] = int(i); im["ed2b_alpha_mask"] = True
                masks[i] = im
            return masks[i]
        n_masks = 0
        for si, s in enumerate(room['surfaces']):
            lm0 = None
            ifl_used = None
            if s['diffuse'] is not None and room['tex_info'][s['diffuse']].get('clamp') and s['diffuse'] in lm_used:
                # the main texture is a baked-light atlas (clamped), not a repeating diffuse map
                lm0 = get_img(s['diffuse']) if self.lightmaps else None
                d = None; dtex = None
            elif s['diffuse'] is None and s.get('diffuse_missing') and len(ifl_frames) == 1 and ifl_frames[0]:
                dtex = ifl_frames[0][0]                    # animated texture: the first frame goes on the material
                d = get_img(dtex); ifl_used = ifl_names[0]
            else:
                dtex = s['diffuse']
                d = get_img(dtex)
            lm1 = get_img(s['lm1']) if self.lightmaps else None
            lm2 = get_img(s['lm2']) if self.lightmaps else None
            mk = None
            for xi in s['extra']:
                if mk is None and is_grey(xi):
                    mk = get_mask(xi); n_masks += 1
            has_alpha = bool(d is not None and (rgba[dtex][..., 3] < 255).any())
            vc = None
            if d is not None:
                px = rgba[dtex][..., :3].reshape(-1, 3).mean(0) / 255.0
                vc = tuple(float(x) ** 2.2 for x in px)
            mat = build_room_material("%s_surf%03d" % (base, si), d, lm1, lm2, has_alpha, vc, self.cull, self.lm_gain, self.ambient, lm0, mk)
            mat["ed2b_surface"] = int(si)
            if has_alpha or mk is not None: mat["ed2b_alpha"] = True
            if ifl_used: mat["ed2b_ifl"] = ifl_used
            mats.append(mat)
        # ---- animated textures: one material per image list with its first frame assigned (plus the matching ...OP list as alpha)
        n_ifl = 0
        by_name = {nm.upper(): fr for nm, fr in zip(ifl_names, ifl_frames)}
        for nm, fr in zip(ifl_names, ifl_frames):
            if not fr or nm.upper().endswith("OP.IFL"): continue
            op = by_name.get(nm.upper()[:-4] + "OP.IFL")
            mk = None
            if op:
                a = rgba.get(op[0])
                if a is None: a = rgba[op[0]] = bperoom.texture_rgba(room, op[0])
                mk = get_mask(op[0])
            first = get_img(fr[0])
            ha = bool((rgba[fr[0]][..., 3] < 255).any())
            vc = tuple(float(x) ** 2.2 for x in rgba[fr[0]][..., :3].reshape(-1, 3).mean(0) / 255.0)
            m = build_room_material("%s_IFL_%s" % (base, nm[:-4]), first, None, None, ha, vc, self.cull, self.lm_gain, self.ambient, None, mk)
            m["ed2b_ifl"] = nm; m["ed2b_ifl_frames"] = ",".join(str(x) for x in fr)
            m.use_fake_user = True
            n_ifl += 1
        # ---- every texture of the bank becomes an image (unused ones are kept with a fake user)
        for i in range(len(room['tex_info'])):
            im = get_img(i)
            if im is not None and im.users == 0: im.use_fake_user = True
        for nm, fr in zip(ifl_names, ifl_frames):
            for k, ti_ in enumerate(fr):
                im = images.get(ti_)
                if im is not None and "ed2b_ifl" not in im:
                    im["ed2b_ifl"] = nm; im["ed2b_ifl_frame"] = k
        # ---- mesh
        mesh = build_room_mesh(base + "_room", room, mats, self.scale, self.game_normals)
        obj = bpy.data.objects.new(base + "_room", mesh)
        _collection(ctx, base).objects.link(obj)
        obj["ed2b_room"] = True; obj["ed2b_srcpath"] = os.path.abspath(self.filepath); obj["ed2b_scale"] = float(self.scale)
        if room.get('ifl'): obj["ed2b_ifl"] = ", ".join(room['ifl'])
        msg = "room: %d triangles, %d surfaces, %d textures, %d alpha masks, %d animated texture(s) (%d with alpha)" % (len(room['tris']), len(mats), len(images), len(masks), n_ifl, sum(1 for m in mats if m.get("ed2b_alpha")))
        if self.objects:
            n = build_objects(ctx, room, base, self.scale)
            msg += ", %(cameras)d cameras, %(regions)d regions, %(markers)d markers" % n
            if room['objects'].get('error'): msg += " (object table: %s)" % room['objects']['error']
        if self.collision:
            cm = build_collision_mesh(base + "_collision", room, self.scale)
            co = bpy.data.objects.new(base + "_collision", cm)
            _collection(ctx, base + "_Collision").objects.link(co)
            co.display_type = 'WIRE'
            try: co.hide_set(True)
            except Exception: pass
            co.hide_render = True
        self.report({'INFO'}, "Imported " + msg)
        return {'FINISHED'}



# --- texture export ---
def room_objects(scene):
    return [o for o in scene.objects if o.get("ed2b_room")]


def pick_room_object(ctx):
    """The room mesh to export from: the active object if it is a room (or its child), else the only room in the scene."""
    a = ctx.object
    while a is not None:
        if a.get("ed2b_room"): return a
        a = a.parent
    rooms = room_objects(ctx.scene)
    return rooms[0] if len(rooms) == 1 else None


def classify_textures(room):
    """-> (set of lightmap texture ids, set of ordinary texture ids)."""
    ti = room['tex_info']
    diffuse, lm = set(), set()
    for s in room['surfaces']:
        for k in ('lm1', 'lm2'):
            if s[k] is not None: lm.add(s[k])
    for s in room['surfaces']:
        d = s['diffuse']
        if d is None: continue
        if ti[d].get('clamp') and d in lm: continue                 # baked-light atlas in the main slot
        diffuse.add(d)
    lights, textures = set(), set()
    for i, t in enumerate(ti):
        if i in diffuse: textures.add(i)
        elif i in lm or t.get('clamp'): lights.add(i)
        else: textures.add(i)
    return lights, textures


def export_room_textures(srcpath, outdir, base):
    """Write every texture of the room file as PNG: <outdir>/textures and <outdir>/light maps."""
    with open(srcpath, 'rb') as f: room = bperoom.parse(f.read())
    lights, textures = classify_textures(room)
    dt, dl = os.path.join(outdir, "textures"), os.path.join(outdir, "light maps")
    os.makedirs(dt, exist_ok=True); os.makedirs(dl, exist_ok=True)
    n_t = n_l = 0
    for i in range(len(room['tex_info'])):
        rgba = bperoom.texture_rgba(room, i)
        folder = dl if i in lights else dt
        with open(os.path.join(folder, "%s_tex%03d.png" % (base, i)), 'wb') as f: f.write(_png_bytes(rgba))
        if i in lights: n_l += 1
        else: n_t += 1
    return n_t, n_l


class ED2B_OT_export_room_textures(bpy.types.Operator):
    bl_idname = "ed2b.export_room_textures"
    bl_label = "Export all textures from room"
    bl_description = ("Choose a folder: every texture of the imported room is written as PNG into two sub folders, "
                      "\"textures\" and \"light maps\"")
    directory: StringProperty(subtype='DIR_PATH')
    filter_folder: BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, ctx, event):
        o = pick_room_object(ctx)
        if o is None:
            self.report({'ERROR'}, "No imported room found (select the room object if there are several)")
            return {'CANCELLED'}
        if not os.path.isfile(o.get("ed2b_srcpath", "")):
            self.report({'ERROR'}, "The room's .bpe file is no longer at %s - re-import it from where it is now" % o.get("ed2b_srcpath", "?"))
            return {'CANCELLED'}
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        o = pick_room_object(ctx)
        if o is None: return {'CANCELLED'}
        out = bpy.path.abspath(self.directory)
        base = os.path.splitext(os.path.basename(o["ed2b_srcpath"]))[0]
        try:
            n_t, n_l = export_room_textures(o["ed2b_srcpath"], out, base)
        except Exception as e:
            self.report({'ERROR'}, "Export failed: %s" % e); return {'CANCELLED'}
        self.report({'INFO'}, "Exported %d texture(s) and %d light map(s) to %s" % (n_t, n_l, out))
        return {'FINISHED'}


classes = (ImportEDRoom, ED2B_OT_export_room_textures)
