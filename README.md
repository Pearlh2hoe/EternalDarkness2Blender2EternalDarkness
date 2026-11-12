# EternalDarkness2Blender2EternalDarkness

A Blender add-on for **Eternal Darkness: Sanity's Requiem**. It imports the game's models, animations and room files into Blender, and exports edited models and textures back into the game's `.bin` format.

> You need your own copy of the game files. This add-on does not include any game data.

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Importing models](#importing-models)
- [Importing animations](#importing-animations)
- [Importing rooms](#importing-rooms)
- [Exporting textures](#exporting-textures)
- [Exporting models](#exporting-models)
- [The ED Models sidebar tab](#the-ed-models-sidebar-tab)
- [Supported files](#supported-files)
- [Known limitations](#known-limitations)

---

## Features

**Models**
- Meshes, skeletons, vertex weights, materials and textures
- Game normals, with sharp edges at UV borders
- Face morph channels as shape keys
- Import a single model slot or every model in a file
- A "models only" mode for editing
- Automatic mapping to Bayonetta 3 bone names (made for my use only lmao)

**Animations**
- Cinematic actor animation and camera
- Gameplay animation packs (`charNNNN.cpk`) applied to an already imported rig

**Rooms (`.bpe`)**
- Full room geometry, with triangle strips expanded and faces oriented to match the stored normals
- Diffuse textures, alpha cut-outs and separate alpha mask textures
- Optional baked lightmaps
- Cameras, trigger regions and markers placed as Blender objects
- Optional collision mesh
- Every texture in the room is imported, including unused ones
- Animated textures: one material per image list, first frame assigned

**Export**
- Edited models back to `.bin`
- Textures only back to `.bin`
- All textures of a character slot or a room as PNG files

---

## Installation

1. Download `EternalDarkness2Blender2EternalDarkness.zip`.
2. In Blender, open **Edit > Preferences > Add-ons > Install...**
3. Select the zip and enable **EternalDarkness2Blender2EternalDarkness**.

The three importers appear under **File > Import**, and the exporters under **File > Export**.

---

## Importing models

**File > Import > EternalDarkness2Blender2EternalDarkness (.gpk/.bin/.dec/.res)**

Models live in `U_LevelNN.bin` (player and level models) and in the `NPC*.gpk` packs. Files compressed with `*SK_ASC*` are decoded by the add-on, so you can pick the original game file.

| Option | Default | What it does |
| --- | --- | --- |
| Slot (-1 = all models) | -1 | Import one model slot, or all of them |
| Armature + weights | on | Build the skeleton and skin weights |
| Use game normals | on | Apply the vertex normals stored in the file |
| Bayonetta 3 bone names (experimental) | off | Rename bones to Bayonetta 3 names |
| Reset scene before import | on | Clear the scene first |
| Models only (no animation, for editing) | off | Skip animation and cameras |
| Cinematic animation | on | Import actor animation from cinematic files |
| Cinematic camera | on | Import the camera from cinematic files |
| Animation frame step | 1 | Sample every Nth frame |
| Scale | 0.01 | Game units to Blender units |

---

## Importing animations

**File > Import > EternalDarkness2Blender2EternalDarkness animations (.cpk)**

Gameplay animation lives in `charNNNN.cpk`. Import the character first, then import its pack. The add-on matches the pack to the rig by hash and creates one Action per clip.

| Option | Default | What it does |
| --- | --- | --- |
| Set (-1 = all) | 0 | Which animation set to import |
| Clip id (-1 = all) | -1 | Import one clip, or all in the set |
| Animation frame step | 1 | Sample every Nth frame |

Some `.cpk` files are alias stubs that reuse the animations of another `charNNNN.cpk`. The importer tells you which file to use instead.

When you pick a different Action on one of these rigs, the scene frame range follows the clip.

---

## Importing rooms

**File > Import > EternalDarkness2Blender2EternalDarkness room (.bpe)**

Pick a room file from the game's `Rooms` folder (for example `rmed0000.bpe`).

| Option | Default | What it does |
| --- | --- | --- |
| Scale | 0.01 | Game units to Blender units |
| Lightmaps | on | Use the baked lightmaps in the materials |
| Lightmap gain | 1.0 | Brightness multiplier for the lightmaps |
| Ambient | 0.1 | Constant light added to the lightmaps |
| Hide back faces | on | Back-face culling on the materials |
| Use game normals | on | Apply the normals stored in the room |
| Cameras, regions and markers | on | Place the room's objects |
| Collision polygons | off | Import the collision polygons as a hidden wire mesh |
| Reset scene before import | on | Clear the scene first |

### Lightmaps

The GameCube blends raw 8-bit values, so with Lightmaps on the images are loaded as Non-Color and a gamma node converts the result back. If a room looks too dark or too bright, change **Lightmap gain** (brightness of the light) or **Ambient** (the minimum light) in the import dialog. Turn **Lightmaps** off to get plain diffuse materials.

Some surfaces use a baked light atlas where a diffuse texture would normally be. The add-on detects these and treats them as lightmaps, not as repeating textures.

### Textures and animated textures

Every texture in the room's texture bank becomes an image in the blend file. Images that no material uses are kept with a fake user so they survive saving.

For each animated texture list (`*.IFL`) the add-on creates a material named `<room>_IFL_<name>` with the **first frame** assigned. If a matching `…OP` list exists, its first frame is used as the alpha mask. All frames are imported as images and tagged with the list name and frame number.

Animated textures are not played back. Only the first frame is assigned.

---

## Exporting textures

In the **Properties editor > Material** tab there is a panel called *EternalDarkness2Blender* with two buttons.

**Export all textures from selected slot** writes every texture of an imported character slot as PNG into a folder you choose.

**Export all textures from room** writes every texture of the selected room into two sub folders:

```
<your folder>/
  textures/
    rmed0000_tex000.png
    ...
  light maps/
    rmed0000_tex001.png
    ...
```

The file number is the texture's position in the room's texture bank. Lightmaps are the images used as lightmaps by any surface, plus baked light atlases. Everything else goes to `textures`.

The room export reads the original `.bpe` again (the path is stored on the room object), so the file must still be where you imported it from. It does not depend on the Lightmaps option.

If the scene has several rooms, select the room object you want first.

---

## Exporting models

- **File > Export > Model (.bin)** writes an edited model back into a game `.bin`.
- **File > Export > Textures only (.bin)** writes edited textures back.

Both need the **original file** as a template: the game rejects resources that change size or position, so the add-on writes your changes into the original layout. If an edit does not fit (too many vertices, a texture of a different size, and so on) the exporter stops and says why. Details are printed to Blender's system console.

The same operators are available in the **ED Models** sidebar tab.

---

## The ED Models sidebar tab

Press `N` in the 3D viewport and open the **ED Models** tab.

- Lists every imported model slot
- Click a name to select it, use the eye icon to show or hide it
- **Show all** and **Hide all** buttons
- Export buttons for models and textures

---

## Supported files

| File | Used for |
| --- | --- |
| `U_LevelNN.bin` | Player and level models |
| `NPC*.gpk` | NPC models |
| `charNNNN.cpk` | Gameplay animation packs |
| `cinNNNN.bin` / `.bpe` | Cinematics (actor animation and camera) |
| `rmedNNNN.bpe` | Rooms |
| `.dec`, `.res` | Already decoded containers and single model resources |

`.gpk` and `.bin` files that start with `*SK_ASC*` are decompressed by the add-on. Room files are `.bpe` (Gage BPE compression).

---

## Known limitations

- **Animated textures** are not played. Only the first frame is assigned, and a surface is only linked to its animated texture when the room has a single image list.
- **Room items and props** that the game's scripts spawn are not imported. The room files only hold markers, not the item models.
- **Lightmap blending** is an approximation of the game's own lighting.
- **Colour "extra" textures** on room surfaces (for example lit stained glass) are imported as images but not connected to the material.
- **Blood decals** and similar overlays show as opaque cards, because the game blends them differently.
- Rooms with very large stray vertices (for example `rmed0037`) have a large bounding box.
