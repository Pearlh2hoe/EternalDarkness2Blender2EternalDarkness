bl_info = {
    "name": "EternalDarkness2Blender2EternalDarkness",
    "author": "Pearl23",
    "version": (1, 0, 0),
    "blender": (2, 80, 0),
    "location": "File > Import / Export > EternalDarkness2Blender2EternalDarkness; sidebar tab \"ED Models\"",
    "description": "Eternal Darkness (GameCube): import models, skeletons, weights, textures, cutscene and character animation, and .bpe room files (geometry, textures, optional lightmaps, cameras, regions, markers); export edited models and textures back to .bin",
    "category": "Import-Export",
}
import os
import bpy
import numpy as np
from bpy.props import StringProperty, BoolProperty, IntProperty, FloatProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Vector

from . import edmodel, gpkdec, cinanim, cpkanim, icons, exporter, roomimport


BONE_SIZE = 0.5        # imported bone length relative to the distance to the child bone


# Shape key names for the face morph channels. The game only stores channel numbers, so the names come from what each
# channel does to the face; the mouth channels (0-14) are approximate viseme labels. Channels not listed are named
# ED_ExprNN. R/L is the character's right/left.
CHANNEL_NAMES = {
    0: "Viseme_MBP", 1: "Viseme_AH", 2: "Viseme_OH", 3: "Viseme_OO", 4: "Viseme_EH", 5: "Viseme_EE",
    6: "Viseme_TH", 7: "Viseme_FV", 8: "Viseme_AW", 9: "Viseme_AO", 10: "Viseme_W", 11: "Viseme_IH",
    12: "Viseme_NDL", 13: "Viseme_ER", 14: "Viseme_SH",
    15: "Blink_R", 16: "Blink_L", 17: "BrowRaise_R", 18: "BrowRaise_L", 19: "JawOpen",
    21: "Glare", 22: "Grimace", 25: "Frown", 27: "Squint_R", 28: "Squint_L",
    29: "Scared", 30: "Smirk", 31: "Displeasure",
}

def shape_name(ch):
    return "ED_" + CHANNEL_NAMES.get(ch, "Expr%02d" % ch)


def build_model(ctx, m, name, with_armature=True, scale=0.01, use_game_normals=True, bone_names=False):
    mesh = bpy.data.meshes.new(name)
    verts = [tuple(float(c) * scale for c in p) for p in m.pos]
    faces = [tuple(int(i) for i in t) for t in m.tri_verts]
    # drop degenerate triangles (strip restarts)
    keep = [i for i, f in enumerate(faces) if len(set(f)) == 3]
    faces = [faces[i] for i in keep]
    tri_corners = m.tri_corners[keep]
    groups = m.tri_group[keep]
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    # UVs (V is flipped for Blender)
    uvl = mesh.uv_layers.new(name="UVMap")
    for pi, poly in enumerate(mesh.polygons):
        for k, li in enumerate(poly.loop_indices):
            u, v = m.uv[m.corner[tri_corners[pi][k]][1]]
            uvl.data[li].uv = (float(u), 1.0 - float(v))
    # ---- shading: smooth everywhere; UV island borders marked sharp (+ seam)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    if hasattr(mesh, "use_auto_smooth"): mesh.use_auto_smooth = True      # Blender < 4.1
    uvmap = {}
    def uvkey(ci): return tuple(np.round(m.uv[m.corner[ci][1]], 4))
    for pi, f in enumerate(faces):
        tc = tri_corners[pi]
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            ua, ub = uvkey(tc[k]), uvkey(tc[(k + 1) % 3])
            if a > b: a, b, ua, ub = b, a, ub, ua
            uvmap.setdefault((a, b), set()).add((ua, ub))
    edge_index = {}
    for e in mesh.edges: edge_index[tuple(sorted(e.vertices))] = e.index
    for key, variants in uvmap.items():
        if len(variants) > 1 and key in edge_index:
            e = mesh.edges[edge_index[key]]
            e.use_edge_sharp = True; e.use_seam = True
    if use_game_normals:
        try:
            nrm = m.vec[0].astype('float64')
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
            mesh.normals_split_custom_set_from_vertices([tuple(map(float, n)) for n in nrm])
        except Exception:
            pass                                  # keep smooth shading with sharp UV borders
    # one material per prim group, each with its own diffuse texture (material table at header+0x60)
    imgs = {}
    def get_img(ti):
        if ti is None: return None
        if ti not in imgs:
            tex = m.textures[ti]
            h, w = tex.shape[:2]
            im = bpy.data.images.new("%s_tex%02d" % (name, ti), w, h, alpha=True)
            im.pixels = (tex[::-1].astype('float32') / 255.0).ravel().tolist()   # Blender rows start at the bottom
            im["ed2b_tex"] = int(ti)                                # exporter metadata: texture index in the model
            im.pack(); imgs[ti] = im
        return imgs[ti]
    gids = sorted(set(int(g) for g in groups)) or [0]
    for g in gids:
        mat = bpy.data.materials.new("%s_g%d" % (name, g))
        mat["ed2b_group"] = int(g)                                  # exporter metadata: prim group / material index
        mat.use_nodes = True
        mat.roughness = 1.0                      # viewport display roughness
        bs = mat.node_tree.nodes.get('Principled BSDF')
        if bs: bs.inputs['Roughness'].default_value = 1.0   # render roughness
        di = m.diffuse_for_group(g)
        im = get_img(di)
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if im:
            node = mat.node_tree.nodes.new('ShaderNodeTexImage'); node.image = im; node.label = "diffuse"
            node.location = (-420, 300)
            if bsdf:
                mat.node_tree.links.new(node.outputs['Color'], bsdf.inputs['Base Color'])
                if (m.textures[di][..., 3] < 255).any():            # CMPR 1-bit alpha is used by this texture
                    mat.node_tree.links.new(node.outputs['Alpha'], bsdf.inputs['Alpha'])
                    for attr, val in (('blend_method', 'HASHED'), ('shadow_method', 'HASHED')):
                        try: setattr(mat, attr, val)
                        except Exception: pass
        # secondary (grey) map -> specular intensity
        si = m.materials[g]['secondary'] if g < len(m.materials) else None
        if si is not None and si != di and m.textures[si] is not None:
            sn = mat.node_tree.nodes.new('ShaderNodeTexImage'); sn.image = get_img(si); sn.label = "secondary (specular)"
            sn.location = (-420, -60)
            try: sn.image.colorspace_settings.name = 'Non-Color'
            except Exception: pass
            if bsdf:
                sock = bsdf.inputs.get('Specular IOR Level') or bsdf.inputs.get('Specular')   # 4.x / 2.8-3.x
                if sock: mat.node_tree.links.new(sn.outputs['Color'], sock)
        # every remaining texture of the model as an unlinked node, laid out in a grid
        used = {di, si}
        col = 0
        for ti, tex in enumerate(m.textures):
            if tex is None or ti in used: continue
            un = mat.node_tree.nodes.new('ShaderNodeTexImage'); un.image = get_img(ti)
            un.label = "tex %02d" % ti
            un.location = (-1100 - (col % 6) * 280, 300 - (col // 6) * 300); col += 1
        mesh.materials.append(mat)
    for pi, poly in enumerate(mesh.polygons):
        poly.material_index = gids.index(int(groups[pi]))
    # exporter metadata: original vertex index per vertex (survives edits that reorder / delete vertices)
    try:
        a = mesh.attributes.new("ed2b_vid", 'INT', 'POINT'); a.data.foreach_set("value", list(range(1, len(verts) + 1)))      # original index + 1 (0 = vertex created in Blender)
    except Exception: pass
    obj = bpy.data.objects.new(name, mesh)
    ctx.collection.objects.link(obj)

    if with_armature and m.bones:
        arm = bpy.data.armatures.new(name + "_arm")
        arm.display_type = 'STICK'
        aobj = bpy.data.objects.new(name + "_rig", arm)
        ctx.collection.objects.link(aobj)
        ctx.view_layer.objects.active = aobj
        bpy.ops.object.mode_set(mode='EDIT')
        worlds = m.bone_world()
        bnames = ["b%02d_%08X" % (i, m.bones[i]['name_hash']) for i in range(len(m.bones))]
        if bone_names:
            from .bonemap import BONE_NAMES
            bnames = [BONE_NAMES.get(m.bones[i]['name_hash'], bnames[i]) for i in range(len(m.bones))]
        ebs = []
        # Blender bones point along local +Y; the game's bones point along local +X.
        # Bone frame = (X_b, Y_b, Z_b) = (game Y, game X, -game Z) so poses from the game map 1:1.
        CM3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
        for i, W in enumerate(worlds):
            eb = arm.edit_bones.new(bnames[i])
            M = np.eye(4); M[:3, :3] = W[:3, :3] @ CM3; M[:3, 3] = W[:3, 3] * scale
            ydir = Vector(M[:3, 1]).normalized()
            eb.head = Vector(M[:3, 3])
            eb.tail = eb.head + ydir * (10.0 * scale)
            eb.matrix = Matrix(M.tolist())
            eb.length = 10.0 * scale * BONE_SIZE
            ebs.append(eb)
        for i, b in enumerate(m.bones):
            p = b['parent']
            if 0 <= p < len(ebs) and p != i:
                ebs[i].parent = ebs[p]
        # lengthen towards the farthest child (keeps the bone's orientation/roll)
        for i, eb in enumerate(ebs):
            kids = [ebs[j] for j, bb in enumerate(m.bones) if bb['parent'] == i and j != i and 0 <= j < len(ebs)]
            if kids:
                far = max(kids, key=lambda c: (c.head - eb.head).length)
                L = (far.head - eb.head).length
                if L > 0.5 * scale: eb.length = L * BONE_SIZE
        bpy.ops.object.mode_set(mode='OBJECT')
        # only create vertex groups for bones that actually influence vertices
        used = sorted({int(b) for infl in m.skin for b, w in infl if 0 <= b < len(m.bones) and w > 0})
        groups_by_bone = {b: obj.vertex_groups.new(name=bnames[b]) for b in used}
        for v, infl in enumerate(m.skin):
            for bone, w in infl:
                if int(bone) in groups_by_bone and w > 0:
                    groups_by_bone[int(bone)].add([v], float(w), 'ADD')
        aobj["ed2b_hash"] = "%08X" % (int(m.hash) & 0xFFFFFFFF)      # lets a .cpk animation pack find this rig
        aobj["ed2b_scale"] = float(scale)
        aobj["ed2b_bones"] = list(bnames)
        aobj["ed2b_parents"] = [int(b['parent']) for b in m.bones]
        obj.parent = aobj
        mod = obj.modifiers.new("Armature", 'ARMATURE'); mod.object = aobj
    # face morph targets -> shape keys "ED_<name>" (channel ids as used by the cutscene stream)
    if getattr(m, 'morphs', None):
        import numpy as _np
        obj.shape_key_add(name="Basis", from_mix=False)
        base = _np.array([v.co[:] for v in mesh.vertices], _np.float32)
        for ch in sorted(m.morphs):
            idx, dl = m.morphs[ch]
            co = base.copy(); _np.add.at(co, idx, dl * scale)
            sk = obj.shape_key_add(name=shape_name(ch), from_mix=False)
            sk.data.foreach_set('co', co.ravel()); sk.slider_min = 0.0; sk.slider_max = 1.0; sk.value = 0.0
    return obj, (aobj if (with_armature and m.bones) else None), (bnames if (with_armature and m.bones) else None)


def apply_morphs(obj, m, mtrack, step=1):
    """Key the ED_<name> shape keys from the (frame, {channel: value}) list of a cinematic actor."""
    sks = obj.data.shape_keys
    if not sks or not mtrack: return 0
    kb = sks.key_blocks; cur = {}
    last = 0
    for fi, vals in mtrack:
        for ch, v in vals.items():
            k = kb.get(shape_name(ch))
            if k is None: continue
            if ch not in cur and fi > 0:             # hold the rest value up to the first write
                k.value = 0.0; k.keyframe_insert('value', frame=fi)
            k.value = v; k.keyframe_insert('value', frame=fi + 1); cur[ch] = v
        last = max(last, fi)
    for k in kb:
        if k.name.startswith('ED_'): k.value = 0.0      # animation drives it; the static value stays 0
    return last


def apply_animation(aobj, m, bnames, track, scale, step=1):
    """Key the armature's pose bones from a cinematic track. Returns the last cinematic frame keyed."""
    arm = aobj.data
    CM4 = Matrix(((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    rest = [Matrix(arm.bones[n].matrix_local) for n in bnames]
    par = [b['parent'] if 0 <= b['parent'] < len(bnames) and b['parent'] != i else -1 for i, b in enumerate(m.bones)]
    pbs = [aobj.pose.bones[n] for n in bnames]
    for pb in pbs: pb.rotation_mode = 'QUATERNION'
    aobj.animation_data_create()
    first = track[0][0]; last = first
    for k, (fi, W) in enumerate(cinanim.entity_poses(m, track)):
        if k % step: continue
        P = []
        for Wi in W:
            Ws = Wi.copy(); Ws[:3, 3] *= scale
            P.append(Matrix(Ws.tolist()) @ CM4)
        for i, pb in enumerate(pbs):
            if par[i] < 0 and i != 0: continue          # static helper roots keep their bind pose
            if par[i] >= 0:
                basis = rest[i].inverted() @ rest[par[i]] @ P[par[i]].inverted() @ P[i]
            else:
                basis = rest[i].inverted() @ P[i]
            loc, rot, _s = basis.decompose()
            pb.location = loc; pb.rotation_quaternion = rot
            pb.keyframe_insert('location', frame=fi + 1, group=pb.name)
            pb.keyframe_insert('rotation_quaternion', frame=fi + 1, group=pb.name)
        last = fi
    return last


def apply_cpk_clip(aobj, name, tracks, step=1):
    """Create an Action named `name` from one decoded .cpk clip and make it the rig's active action.
    Returns the number of frames (30 fps)."""
    names = list(aobj["ed2b_bones"]); par = list(aobj["ed2b_parents"]); scale = float(aobj["ed2b_scale"])
    nb = len(names)
    arm = aobj.data
    CM4 = Matrix(((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    rest = [Matrix(arm.bones[n].matrix_local) for n in names]
    pbs = [aobj.pose.bones[n] for n in names]
    for pb in pbs: pb.rotation_mode = 'QUATERNION'
    fr, Q, P, const = cpkanim.sample(tracks, nb, step)
    action = bpy.data.actions.new(name); action.use_fake_user = True
    aobj.animation_data_create(); aobj.animation_data.action = action
    for k, f in enumerate(fr):
        W = [None] * nb
        for i in range(nb):
            M = np.eye(4); M[:3, :3] = cinanim.quat_to_mat(Q[k, i]); M[:3, 3] = P[k, i]
            if par[i] < 0 or par[i] == i: W[i] = M
            else: W[i] = W[par[i]] @ M
        Pm = []
        for Wi in W:
            Ws = Wi.copy(); Ws[:3, 3] *= scale
            Pm.append(Matrix(Ws.tolist()) @ CM4)
        for i, pb in enumerate(pbs):
            if par[i] < 0 and i != 0: continue            # static helper roots keep their bind pose
            if k > 0 and const[i]: continue               # nothing changes on this bone: one key is enough
            if par[i] >= 0: basis = rest[i].inverted() @ rest[par[i]] @ Pm[par[i]].inverted() @ Pm[i]
            else: basis = rest[i].inverted() @ Pm[i]
            loc, rot, _s = basis.decompose()
            pb.location = loc; pb.rotation_quaternion = rot
            pb.keyframe_insert('location', frame=int(f) + 1, group=pb.name)
            pb.keyframe_insert('rotation_quaternion', frame=int(f) + 1, group=pb.name)
    return int(fr[-1]) + 1


class ImportEDAnim(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.eternal_darkness_anim"
    bl_label = "Import Animations (.cpk)"
    bl_description = "Import the clips of a charNNNN.cpk animation pack as Actions onto the matching, already imported character rig"
    filename_ext = ".cpk"
    filter_glob: StringProperty(default="*.cpk", options={'HIDDEN'})
    clip_set: IntProperty(name="Set (-1 = all)", default=0, min=-1,
                          description="Clip set (stance) to import: the pack's slot type (0, 1, 2, 4, 5, 14...). -1 imports every set (hundreds of clips: slow, raise the frame step)")
    clip: IntProperty(name="Clip id (-1 = all)", default=-1, min=-1,
                      description="Only import the clip with this id (ids are shared between characters)")
    anim_step: IntProperty(name="Animation frame step", default=1, min=1, max=30,
                           description="Keyframe every Nth frame (30 fps source)")

    def execute(self, ctx):
        try:
            with open(self.filepath, 'rb') as fh: d = fh.read()
            hd = cpkanim.header(d)
        except Exception as e:
            self.report({'ERROR'}, "Cannot read pack: %s" % e); return {'CANCELLED'}
        if hd['alias'] is not None:
            self.report({'ERROR'}, "This file is an alias stub: it reuses the animations of char%04d.cpk - import that file instead" % hd['alias'])
            return {'CANCELLED'}
        hx = "%08X" % hd['hash']
        rigs = [o for o in ctx.scene.objects if o.type == 'ARMATURE' and o.get("ed2b_hash") == hx]
        if not rigs:
            self.report({'ERROR'}, "No imported character with model hash %s in this scene. Import the matching model first "
                                   "(with 'Armature + weights' on), then import this pack." % hx)
            return {'CANCELLED'}
        sel = [o for o in rigs if o.select_get()]
        rigs = sel or rigs
        pack = cpkanim.load(d, self.clip_set)
        total = 0; longest = 0
        for aobj in rigs:
            for typ, clips in pack['sets']:
                for cid, tracks in clips:
                    if self.clip >= 0 and cid != self.clip: continue
                    longest = max(longest, apply_cpk_clip(aobj, "%s_set%02d_clip%03d" % (aobj.name, typ, cid), tracks, self.anim_step))
                    total += 1
            # leave the rig in its rest pose; the clips are listed in the Action editor (fake user keeps them)
            if aobj.animation_data: aobj.animation_data.action = None
            for pb in aobj.pose.bones:
                pb.location = (0, 0, 0); pb.rotation_quaternion = (1, 0, 0, 0)
        if total == 0:
            self.report({'WARNING'}, "No matching clips (set %d, clip %d)" % (self.clip_set, self.clip)); return {'CANCELLED'}
        ctx.scene.render.fps = 30
        ctx.scene.frame_start = 1; ctx.scene.frame_end = max(longest, 2)
        self.report({'INFO'}, "Imported %d clip(s) as Actions onto %d rig(s). Assign one in the Action editor / Dope Sheet." % (total, len(rigs)))
        return {'FINISHED'}


class ImportED(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.eternal_darkness"
    bl_label = "Import EternalDarkness2Blender"
    filename_ext = ".gpk"
    filter_glob: StringProperty(default="*.gpk;*.bin;*.dec;*.res;*.edres", options={'HIDDEN'})
    slot: IntProperty(name="Slot (-1 = all models)", default=-1, min=-1)
    armature: BoolProperty(name="Armature + weights", default=True)
    game_normals: BoolProperty(name="Use game normals", default=True,
                               description="Apply the normals stored in the model; off = smooth shading with UV borders sharp")
    bone_names: BoolProperty(name="Bayonetta 3 bone names (experimental)", default=False,
                             description="Name the known bones with the supplied Bayonetta 3 style mapping (Alex's skeleton); others keep b##_hash")
    reset_scene: BoolProperty(name="Reset scene before import", default=True,
                              description="Delete all objects and unused data (meshes, materials, images, armatures) first")
    models_only: BoolProperty(name="Models only (no animation, for editing)", default=False,
                              description="Cut-scene files (cinNNNN.bin): import just the models in their bind pose - no frame stream is read, no actor animation, no shape-key keyframes and no camera")
    animation: BoolProperty(name="Cinematic animation", default=True,
                            description="With a cinNNNN.bin, also read the extension-less cinNNNN file next to it and animate the actors")
    camera: BoolProperty(name="Cinematic camera", default=True,
                         description="Also create the cutscene camera (position, target, up, field of view) and make it the scene camera")
    anim_step: IntProperty(name="Animation frame step", default=1, min=1, max=30,
                           description="Keyframe every Nth cinematic frame (30 fps source)")
    scale: FloatProperty(name="Scale", default=0.01, min=1e-6, max=100.0, precision=4,
                         description="Import scale (game units -> Blender units)")

    def draw(self, ctx):
        layout = self.layout
        icons.dialog_header(layout, "EternalDarkness2Blender2EternalDarkness")
        for p in ("slot", "scale", "armature", "bone_names", "game_normals", "reset_scene", "models_only"):
            layout.prop(self, p)
        if not self.models_only:
            for p in ("animation", "camera", "anim_step"): layout.prop(self, p)

    def execute(self, ctx):
        if self.reset_scene:
            if ctx.object and ctx.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            for o in list(bpy.data.objects): bpy.data.objects.remove(o, do_unlink=True)
            for coll in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.images):
                for d in list(coll):
                    if d.users == 0: coll.remove(d)
        try:
            res = gpkdec.load_resources(self.filepath)
        except Exception as e:
            self.report({'ERROR'}, str(e)); return {'CANCELLED'}
        base = os.path.splitext(os.path.basename(self.filepath))[0]
        n = 0
        built = {}          # slot -> (m, aobj, bone names)
        objs = self._objs = {}
        parsed = {}
        for idx, data in res.items():
            try: parsed[idx] = edmodel.NpcModel(data)
            except edmodel.ModelError: continue
            except Exception: continue
        for idx, m in parsed.items():
            if self.slot >= 0 and idx != self.slot: continue
            obj, aobj, bnames = build_model(ctx, m, "%s_%03d" % (base, idx), self.armature, self.scale, self.game_normals, self.bone_names)
            built[idx] = (m, aobj, bnames); objs[idx] = obj; n += 1
            obj["ed2b_key"] = int(idx); obj["ed2b_src"] = os.path.basename(self.filepath); obj["ed2b_srcpath"] = os.path.abspath(self.filepath)
            obj["ed2b_hash"] = "%08X" % (int(m.hash) & 0xFFFFFFFF); obj["ed2b_scale"] = float(self.scale)
        if n == 0:
            avail = list(parsed)
            if avail and self.slot >= 0:
                self.report({'ERROR'}, "No model in slot %d. Models in this file are in slots: %s" % (self.slot, avail))
            elif not avail:
                self.report({'ERROR'}, "No models found in this file (use the .bin/.gpk pack, not the extension-less cinematic frame data)")
            return {'CANCELLED'}
        if len(built) == 1: ctx.scene.ed2b_slot = next(iter(built))
        msg = "Imported %d model(s)" % n
        stream = os.path.splitext(self.filepath)[0]
        if self.animation and not self.models_only and self.armature and self.filepath.lower().endswith('.bin') and os.path.isfile(stream):
            try:
                msg += "; " + self.import_animation(ctx, stream, parsed, built, base)
            except Exception as ex:
                import traceback; traceback.print_exc()
                msg += "; cinematic animation failed: %s" % ex
        self.report({'INFO'}, msg)
        return {'FINISHED'}

    def import_animation(self, ctx, stream, parsed, built, base):
        objs = self._objs
        frames = cinanim.load_frames(stream)
        tr = cinanim.tracks(frames); mtr = cinanim.morph_tracks(frames)
        used = set(); count = 0; last = 0
        for obj_id, track in sorted(tr.items()):
            slot = cinanim.match_model(track, parsed)
            if slot is None or slot not in built: continue
            m, aobj, bnames = built[slot]; mobj = objs.get(slot)
            if slot in used:        # same model animated by a second actor: make another copy
                mobj, aobj, bnames = build_model(ctx, m, "%s_%03d_obj%d" % (base, slot, obj_id), True, self.scale, self.game_normals, self.bone_names)
            used.add(slot)
            last = max(last, apply_animation(aobj, m, bnames, track, self.scale, self.anim_step))
            if mobj is not None: apply_morphs(mobj, m, mtr.get(obj_id))
            count += 1
        cam_msg = ""
        if self.camera:
            ctrack = cinanim.camera_track(frames)
            if ctrack:
                last = max(last, apply_camera(ctx, base + "_camera", ctrack, self.scale)); cam_msg = " + camera"
        if last:
            ctx.scene.render.fps = 30
            ctx.scene.frame_start = 1; ctx.scene.frame_end = last + 1
        return "animated %d actor(s)%s" % (count, cam_msg)


import re
_TEX_RE = re.compile(r"_(\d{3})(?:_obj\d+)?_tex(\d\d)$")

def slot_images(slot):
    """Images created by the importer for the given slot (from models already imported in this file)."""
    out = []
    for im in bpy.data.images:
        mt = _TEX_RE.search(im.name)
        if mt and int(mt.group(1)) == slot: out.append(im)
    return sorted(out, key=lambda i: i.name)


def apply_camera(ctx, name, track, scale):
    import math
    cam = bpy.data.cameras.new(name)
    cam.sensor_fit = 'VERTICAL'; cam.sensor_height = 24.0
    obj = bpy.data.objects.new(name, cam)
    ctx.collection.objects.link(obj)
    obj.rotation_mode = 'QUATERNION'
    obj.animation_data_create()
    last = 0
    for fi, pos, tgt, up, fov in track:
        M = cinanim.camera_matrix(pos, tgt, up)
        M[:3, 3] *= scale
        loc, rot, _s = Matrix(M.tolist()).decompose()
        obj.location = loc; obj.rotation_quaternion = rot
        obj.keyframe_insert('location', frame=fi + 1)
        obj.keyframe_insert('rotation_quaternion', frame=fi + 1)
        cam.lens = 0.5 * cam.sensor_height / math.tan(math.radians(fov) / 2.0)   # field of view treated as vertical
        cam.keyframe_insert('lens', frame=fi + 1)
        last = fi
    ctx.scene.camera = obj
    return last


class ED2B_OT_export_textures(bpy.types.Operator):
    bl_idname = "ed2b.export_textures"
    bl_label = "Export all textures from selected slot"
    bl_description = "Choose a folder and write every texture of the imported model in this slot as PNG"
    directory: StringProperty(subtype='DIR_PATH')
    filter_folder: BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, ctx, event):
        if not slot_images(ctx.scene.ed2b_slot):
            self.report({'ERROR'}, "No imported model textures found for slot %d" % ctx.scene.ed2b_slot)
            return {'CANCELLED'}
        ctx.window_manager.fileselect_add(self)          # opens the folder browser window
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        out = bpy.path.abspath(self.directory)
        os.makedirs(out, exist_ok=True)
        n = 0
        for im in slot_images(ctx.scene.ed2b_slot):
            old_path, old_fmt = im.filepath_raw, im.file_format
            try:
                im.filepath_raw = os.path.join(out, im.name + ".png")
                im.file_format = 'PNG'
                im.save(); n += 1
            finally:
                im.filepath_raw, im.file_format = old_path, old_fmt
        self.report({'INFO'}, "Exported %d texture(s) from slot %d to %s" % (n, ctx.scene.ed2b_slot, out))
        return {'FINISHED'}


class ED2B_PT_textures(bpy.types.Panel):
    bl_label = "EternalDarkness2Blender"
    bl_idname = "ED2B_PT_textures"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "material"

    def draw(self, ctx):
        col = self.layout.column()
        col.prop(ctx.scene, "ed2b_slot")
        col.label(text="%d texture(s) imported for this slot" % len(slot_images(ctx.scene.ed2b_slot)))
        col.operator(ED2B_OT_export_textures.bl_idname, icon='EXPORT')
        col.separator()
        ro = roomimport.pick_room_object(ctx)
        if ro is not None:
            col.label(text="Room: %s" % ro.name)
        elif roomimport.room_objects(ctx.scene):
            col.label(text="Select a room object to choose it")
        col.operator(roomimport.ED2B_OT_export_room_textures.bl_idname, icon='EXPORT')


# ---- sidebar tab: show / hide imported model slots ---------------------------------------------------------------
import re as _re
_SLOT_NAME_RE = _re.compile(r"_(\d{1,4})(?:\.\d+)?$")

def _model_objects(scene):
    """[(slot key, mesh object)] of every imported model in the scene, ordered by slot."""
    out = []
    for o in scene.objects:
        if o.type != 'MESH': continue
        if "ed2b_key" in o: out.append((int(o["ed2b_key"]), o))
        else:
            mt = _SLOT_NAME_RE.search(o.name)
            if mt and o.parent is not None and "ed2b_hash" in o.parent: out.append((int(mt.group(1)), o))
    return sorted(out, key=lambda t: (t[0], t[1].name))

def _slot_objects(o):
    """the mesh and its rig"""
    objs = [o]
    if o.parent is not None and "ed2b_hash" in o.parent: objs.append(o.parent)
    return objs

def _set_hidden(o, state):
    for x in _slot_objects(o):
        try: x.hide_set(state)
        except Exception: x.hide_viewport = state


class ED2B_OT_toggle_slot(bpy.types.Operator):
    bl_idname = "ed2b.toggle_slot"
    bl_label = "Show / hide model"
    bl_description = "Show or hide this model (and its armature) in the viewport"
    bl_options = {'INTERNAL'}
    obj: StringProperty()

    def execute(self, ctx):
        o = ctx.scene.objects.get(self.obj)
        if o is None: return {'CANCELLED'}
        _set_hidden(o, not o.hide_get())
        return {'FINISHED'}


class ED2B_OT_select_slot(bpy.types.Operator):
    bl_idname = "ed2b.select_slot"
    bl_label = "Select model"
    bl_description = "Select this model (it is shown first if hidden)"
    bl_options = {'INTERNAL'}
    obj: StringProperty()

    def execute(self, ctx):
        o = ctx.scene.objects.get(self.obj)
        if o is None: return {'CANCELLED'}
        if o.hide_get(): _set_hidden(o, False)
        for x in ctx.scene.objects:
            try: x.select_set(False)
            except Exception: pass
        o.select_set(True); ctx.view_layer.objects.active = o
        return {'FINISHED'}


class ED2B_OT_show_all(bpy.types.Operator):
    bl_idname = "ed2b.show_all"
    bl_label = "Show / hide all models"
    bl_options = {'INTERNAL'}
    hide: BoolProperty(default=False)

    def execute(self, ctx):
        for k, o in _model_objects(ctx.scene): _set_hidden(o, self.hide)
        return {'FINISHED'}


class ED2B_PT_slots(bpy.types.Panel):
    bl_label = "Model slots"
    bl_idname = "ED2B_PT_slots"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ED Models"            # a tab next to Item / Tool / View

    def draw(self, ctx):
        lay = self.layout
        models = _model_objects(ctx.scene)
        if not models:
            lay.label(text="No imported models in this scene", **icons.kw("menu"))
            return self._export_box(lay)
        row = lay.row(align=True)
        op = row.operator(ED2B_OT_show_all.bl_idname, text="Show all", icon='HIDE_OFF'); op.hide = False
        op = row.operator(ED2B_OT_show_all.bl_idname, text="Hide all", icon='HIDE_ON'); op.hide = True
        box = lay.box(); col = box.column(align=True)
        for key, o in models:
            r = col.row(align=True)
            hidden = o.hide_get()
            r.active = not hidden
            op = r.operator(ED2B_OT_select_slot.bl_idname, text="%d  %s" % (key, o.name), emboss=False,
                            icon='OUTLINER_OB_MESH'); op.obj = o.name
            op = r.operator(ED2B_OT_toggle_slot.bl_idname, text="", emboss=False,
                            icon='HIDE_ON' if hidden else 'HIDE_OFF'); op.obj = o.name
        self._export_box(lay)

    def _export_box(self, lay):
        box = lay.box()
        box.label(text="Export (all models in this file)", **icons.kw("menu"))
        box.operator(exporter.ExportEDModel.bl_idname, text="Model (.bin)", **icons.kw("menu"))
        box.operator(exporter.ExportEDTextures.bl_idname, text="Textures only (.bin)", **icons.kw("menu"))


def _icon_kw(name="menu"):
    return icons.kw(name)


def menu(self, ctx):
    self.layout.operator(ImportED.bl_idname, text="EternalDarkness2Blender2EternalDarkness (.gpk/.bin/.dec/.res)", **_icon_kw("menu"))
    self.layout.operator(ImportEDAnim.bl_idname, text="EternalDarkness2Blender2EternalDarkness animations (.cpk)", **_icon_kw("menu"))
    self.layout.operator(roomimport.ImportEDRoom.bl_idname, text="EternalDarkness2Blender2EternalDarkness room (.bpe)", **_icon_kw("menu"))

# ---- keep the scene frame range in step with the active clip ----------------------------------------------------
_last_action = {}          # rig name -> action name last seen

@bpy.app.handlers.persistent
def _sync_frame_range(scene, depsgraph=None):
    """When the action of one of our rigs changes (e.g. picked in the Action Editor), set the scene range to the clip."""
    try:
        changed = None
        for o in scene.objects:
            if o.type != 'ARMATURE' or "ed2b_hash" not in o: continue
            ad = o.animation_data; act = ad.action if ad else None
            nm = act.name if act else None
            if _last_action.get(o.name) != nm:
                _last_action[o.name] = nm
                if act is not None: changed = act
        if changed is not None:
            start, end = changed.frame_range
            start = int(start); end = max(int(round(end)), start + 1)
            if scene.frame_start != start: scene.frame_start = start
            if scene.frame_end != end: scene.frame_end = end
    except Exception:
        pass


classes = (ImportED, ImportEDAnim, ED2B_OT_export_textures, ED2B_PT_textures,
           ED2B_OT_toggle_slot, ED2B_OT_select_slot, ED2B_OT_show_all, ED2B_PT_slots) + exporter.classes + roomimport.classes
def register():
    icons.load()
    bpy.types.Scene.ed2b_slot = IntProperty(name="Slot", default=0, min=0, description="Model slot in the source pack")
    for c in classes: bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu)
    bpy.types.TOPBAR_MT_file_export.append(exporter.menu)
    if _sync_frame_range not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_sync_frame_range)
def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu)
    bpy.types.TOPBAR_MT_file_export.remove(exporter.menu)
    if _sync_frame_range in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_sync_frame_range)
    _last_action.clear()
    for c in reversed(classes): bpy.utils.unregister_class(c)
    del bpy.types.Scene.ed2b_slot
    icons.unload()
