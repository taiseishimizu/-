"""3D 人体モデル版 ストレッチ動画レンダラー（Blender bpy）

make_video.py のポーズ定義を流用し、Skin modifier で作った人体を Cycles で描画する。
伸ばす部位は赤い発光レイヤーで重ね、右パネル（意識ポイント）は PIL で合成する。

usage:
  python3 render3d.py --still 1            # 種目1のキーポーズ静止画
  python3 render3d.py --video 1 [2 ...]    # 指定種目の動画（未指定なら全種目）
"""
import math
import os
import sys

import bpy
import imageio.v2 as imageio
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_video as mv  # noqa: E402

RW, RH = 760, 720  # 3D 描画領域（左側）
SAMPLES = int(os.environ.get("SAMPLES", "10"))

# ---------------------------------------------------------------- body graph
# (name, parent, radius) ; 位置は joints から計算
SKIN, SHIRT, SHORTS, SHOE, HAIR = (0.88, 0.70, 0.58), (0.10, 0.22, 0.42), (0.08, 0.08, 0.10), \
    (0.95, 0.95, 0.95), (0.06, 0.05, 0.05)


def body_points(j, view):
    """2D joints -> 3D 頂点 dict（x, depth, height）"""
    def p3(name, depth=0.0):
        x, y = j[name]
        return Vector((x, depth, y))

    def mix(a, b, t):
        return a.lerp(b, t)

    dl = 0.10 if view == "side" else 0.0  # 脚の奥行きオフセット
    da = 0.19 if view == "side" else 0.0  # 肩の奥行きオフセット
    P = p3("P")
    N = p3("N")
    pts = {"P": P, "abd": mix(P, N, 0.35), "chest": mix(P, N, 0.68), "N": mix(P, N, 0.93)}
    hd = (p3("Hc") - p3("N")).normalized()
    pts["neck"] = N + hd * 0.07
    for i, s in (("1", -1), ("2", 1)):
        H = p3("H" + i, s * dl)
        K = p3("K" + i, s * dl)
        A = p3("A" + i, s * dl)
        T = p3("T" + i, s * dl)
        pts["H" + i] = H
        pts["th" + i] = mix(H, K, 0.45)
        pts["K" + i] = K
        pts["calf" + i] = mix(K, A, 0.30)
        pts["A" + i] = A
        pts["T" + i] = T
        S = p3("S" + i, s * da)
        E = p3("E" + i, s * da)
        Wr = p3("W" + i, s * da)
        if view == "side":
            S = S + (N - P).normalized() * -0.04
        pts["S" + i] = S
        pts["ua" + i] = mix(S, E, 0.45)
        pts["E" + i] = E
        pts["W" + i] = Wr
        fdir = (Wr - E).normalized() if (Wr - E).length > 1e-4 else Vector((0, 0, -1))
        pts["hand" + i] = Wr + fdir * 0.08
    return pts, hd


EDGES = [("P", "abd"), ("abd", "chest"), ("chest", "N"), ("N", "neck"),
         ("P", "H1"), ("H1", "th1"), ("th1", "K1"), ("K1", "calf1"), ("calf1", "A1"), ("A1", "T1"),
         ("P", "H2"), ("H2", "th2"), ("th2", "K2"), ("K2", "calf2"), ("calf2", "A2"), ("A2", "T2"),
         ("N", "S1"), ("S1", "ua1"), ("ua1", "E1"), ("E1", "W1"), ("W1", "hand1"),
         ("N", "S2"), ("S2", "ua2"), ("ua2", "E2"), ("E2", "W2"), ("W2", "hand2")]

RAD = {"P": 0.135, "abd": 0.125, "chest": 0.15, "N": 0.10, "neck": 0.05,
       "H1": 0.095, "th1": 0.085, "K1": 0.055, "calf1": 0.06, "A1": 0.038, "T1": 0.035,
       "S1": 0.06, "ua1": 0.05, "E1": 0.04, "W1": 0.032, "hand1": 0.03}
for _k in list(RAD):
    if _k.endswith("1"):
        RAD[_k[:-1] + "2"] = RAD[_k]

# レイヤー（服・靴・赤ハイライト）は部分グラフ + 半径倍率
SHIRT_E = [("P", "abd"), ("abd", "chest"), ("chest", "N"), ("N", "S1"), ("S1", "ua1"),
           ("N", "S2"), ("S2", "ua2")]
SHORTS_E = [("abd", "P"), ("P", "H1"), ("H1", "th1"), ("P", "H2"), ("H2", "th2")]
SHOE_E = [("A1", "T1"), ("A2", "T2")]

HL_E = {"torso": [("P", "abd"), ("abd", "chest"), ("chest", "N")],
        "ua1": [("S1", "ua1"), ("ua1", "E1")], "fa1": [("E1", "W1")],
        "ua2": [("S2", "ua2"), ("ua2", "E2")], "fa2": [("E2", "W2")],
        "th1": [("H1", "th1"), ("th1", "K1")], "sh1": [("K1", "calf1"), ("calf1", "A1")],
        "ft1": [("A1", "T1")],
        "th2": [("H2", "th2"), ("th2", "K2")], "sh2": [("K2", "calf2"), ("calf2", "A2")],
        "ft2": [("A2", "T2")]}


# ---------------------------------------------------------------- scene
def material(name, color, emit=0.0, rough=0.55, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    if emit:
        b.inputs["Emission Color"].default_value = (*color, 1)
        b.inputs["Emission Strength"].default_value = emit
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
    return m


def srgb(c):
    return tuple(((v / 255) ** 2.2) for v in c)


class SkinPart:
    """Skin modifier で太さのあるチューブを作るオブジェクト"""

    def __init__(self, name, edges, scale, mat):
        self.names = []
        for a, b in edges:
            for n in (a, b):
                if n not in self.names:
                    self.names.append(n)
        self.idx = {n: i for i, n in enumerate(self.names)}
        me = bpy.data.meshes.new(name)
        me.from_pydata([(0, 0, 0)] * len(self.names), [(self.idx[a], self.idx[b]) for a, b in edges], [])
        self.obj = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(self.obj)
        sk = self.obj.modifiers.new("skin", "SKIN")
        sk.use_smooth_shade = True
        sk.branch_smoothing = 1.0
        sub = self.obj.modifiers.new("sub", "SUBSURF")
        sub.levels = sub.render_levels = 2
        for i, n in enumerate(self.names):
            r = RAD[n] * scale
            me.skin_vertices[0].data[i].radius = (r, r)
        me.skin_vertices[0].data[0].use_root = True
        me.materials.append(mat)

    def update(self, pts):
        me = self.obj.data
        for n, i in self.idx.items():
            me.vertices[i].co = pts[n]
        me.update()


class Scene:
    def __init__(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        sc = bpy.context.scene
        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.cycles.samples = SAMPLES
        sc.cycles.use_denoising = True
        sc.render.use_persistent_data = True
        sc.render.resolution_x, sc.render.resolution_y = RW, RH
        sc.render.film_transparent = False
        sc.view_settings.view_transform = "AgX"
        world = bpy.data.worlds.new("w")
        sc.world = world
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs[0].default_value = (*srgb(mv.BG), 1)
        world.node_tree.nodes["Background"].inputs[1].default_value = 0.8

        self.m_skin = material("skin", SKIN, rough=0.5)
        self.m_shirt = material("shirt", SHIRT, rough=0.8)
        self.m_shorts = material("shorts", SHORTS, rough=0.8)
        self.m_shoe = material("shoe", SHOE, rough=0.6)
        self.m_hair = material("hair", HAIR, rough=0.7)
        self.m_red = material("red", (0.85, 0.02, 0.02), emit=0.5, rough=0.35)
        self.m_floor = material("floor", srgb((226, 220, 208)), rough=0.9)
        self.m_wall = material("wall", srgb((214, 206, 192)), rough=0.9)
        self.m_mat = material("mat", srgb((120, 160, 200)), rough=0.8)

        bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
        bpy.context.object.data.materials.append(self.m_floor)

        self.body = SkinPart("body", EDGES, 1.0, self.m_skin)
        self.shirt = SkinPart("shirt", SHIRT_E, 1.06, self.m_shirt)
        self.shorts = SkinPart("shorts", SHORTS_E, 1.08, self.m_shorts)
        self.shoe = SkinPart("shoe", SHOE_E, 1.35, self.m_shoe)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1)
        self.head = bpy.context.object
        self.head.data.materials.append(self.m_skin)
        bpy.ops.object.shade_smooth()
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1)
        self.hair = bpy.context.object
        self.hair.data.materials.append(self.m_hair)
        bpy.ops.object.shade_smooth()
        self.hl_parts = {}
        self.hl_key = None
        self.joint_marks = []
        self.props = []

        # ライト
        key = bpy.data.lights.new("key", "AREA")
        key.energy, key.size = 260, 3
        ko = bpy.data.objects.new("key", key)
        ko.location = (-2.5, -3.5, 4)
        ko.rotation_euler = (math.radians(50), 0, math.radians(-35))
        bpy.context.collection.objects.link(ko)
        fill = bpy.data.lights.new("fill", "AREA")
        fill.energy, fill.size = 110, 4
        fo = bpy.data.objects.new("fill", fill)
        fo.location = (3, -3, 2.5)
        fo.rotation_euler = (math.radians(60), 0, math.radians(40))
        bpy.context.collection.objects.link(fo)

        cam = bpy.data.cameras.new("cam")
        cam.lens = 72
        self.cam = bpy.data.objects.new("cam", cam)
        bpy.context.collection.objects.link(self.cam)
        sc.camera = self.cam

    def set_camera(self):
        # 2D 版のフレーミング（地面 y=0 が画面下寄り、x=0 が中央）に近づける
        target = Vector((0.0, 0.0, 0.92))
        loc = Vector((0.0, -6.2, 1.25))
        self.cam.location = loc
        d = target - loc
        self.cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

    def set_props(self, props, view):
        for o in self.props:
            bpy.data.objects.remove(o, do_unlink=True)
        self.props = []
        for pr in props:
            if pr[0] == "wall":
                bpy.ops.mesh.primitive_cube_add(size=1, location=(pr[1] + 0.06, 0.45, 1.0))
                o = bpy.context.object
                o.scale = (0.12, 1.6, 2.0)
                o.data.materials.append(self.m_wall)
            elif pr[0] == "backwall":
                bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0.35, 1.1))
                o = bpy.context.object
                o.scale = (2.2, 0.1, 2.2)
                o.data.materials.append(self.m_wall)
            elif pr[0] == "mat":
                bpy.ops.mesh.primitive_cube_add(size=1, location=((pr[1] + pr[2]) / 2, 0, 0.005))
                o = bpy.context.object
                o.scale = (pr[2] - pr[1], 0.7, 0.01)
                o.data.materials.append(self.m_mat)
            else:
                continue
            self.props.append(o)

    def set_highlight(self, hl):
        key = tuple(hl)
        if key == self.hl_key:
            return
        self.hl_key = key
        for p in self.hl_parts.values():
            bpy.data.objects.remove(p.obj, do_unlink=True)
        for o in self.joint_marks:
            bpy.data.objects.remove(o, do_unlink=True)
        self.hl_parts, self.joint_marks = {}, []
        for h in hl:
            if h.startswith("j:"):
                bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1)
                o = bpy.context.object
                o.data.materials.append(self.m_red)
                bpy.ops.object.shade_smooth()
                o["joint"] = h[2:]
                self.joint_marks.append(o)
            else:
                scale = 1.12 if h == "torso" else 1.45
                self.hl_parts[h] = SkinPart("hl_" + h, HL_E[h], scale, self.m_red)

    def pose(self, j, view, pulse):
        pts, hd = body_points(j, view)
        for part in (self.body, self.shirt, self.shorts, self.shoe, *self.hl_parts.values()):
            part.update(pts)
        hc = pts["neck"] + hd * 0.11
        self.head.location = hc
        self.head.scale = (0.095, 0.1, 0.118)
        self.head.rotation_euler = Vector((0, 0, 1)).rotation_difference(hd).to_euler()
        self.hair.location = hc + hd * 0.025
        self.hair.scale = (0.1, 0.105, 0.11)
        self.hair.rotation_euler = self.head.rotation_euler
        # 横向きのとき髪を後頭部側へ
        if view == "side":
            back = Vector((-hd.z, 0, hd.x)) * -0.0
            self.hair.location = hc + hd * 0.02 + Vector((-0.018, 0, 0.0)) + back
        self.m_red.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 0.25 + 0.6 * pulse
        for o in self.joint_marks:
            name = o["joint"]
            depth = 0.0
            if view == "side" and name[-1] in "12":
                depth = (-1 if name[-1] == "1" else 1) * (0.19 if name[0] in "SEW" else 0.10)
            x, y = j[name]
            o.location = (x, depth - 0.02, y)
            r = 0.075
            o.scale = (r, r, r)

    def render(self, path):
        bpy.context.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        return Image.open(path).convert("RGB")


# ---------------------------------------------------------------- compositing
def panel_layer(ex, i, cap):
    layer = Image.new("RGBA", (mv.W * mv.SS, mv.H * mv.SS), (0, 0, 0, 0))
    mv.draw_panel(layer, ex, i, len(mv.EXERCISES), cap, 0)
    return layer.resize((mv.W, mv.H), Image.LANCZOS)


def compose(fig, panel, prog):
    canvas = Image.new("RGBA", (mv.W, mv.H), mv.BG + (255,))
    canvas.paste(fig, (0, 0))
    canvas.alpha_composite(panel)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, mv.H - 10, mv.W, mv.H], fill=(225, 220, 210))
    d.rectangle([0, mv.H - 10, int(mv.W * prog), mv.H], fill=mv.RED)
    return canvas.convert("RGB")


STEP = int(os.environ.get("STEP", "2"))  # N フレームごとに描画（間は複製）
TMP = os.environ.get("TMPDIR_3D", "/tmp/r3d")


def main():
    os.makedirs(TMP, exist_ok=True)
    args = sys.argv[1:]
    mode = args[0]
    ids = [int(a) for a in args[1:]] or list(range(1, len(mv.EXERCISES) + 1))
    sc = Scene()
    sc.set_camera()

    if mode == "--still":
        for i in ids:
            ex = mv.EXERCISES[i - 1]
            sc.set_props(ex["props"], ex["view"])
            for k, key in enumerate(ex["keys"]):
                sc.set_highlight(key[4])
                sc.pose(mv.solve(key[0]), ex["view"], 1.0)
                fig = sc.render(f"{TMP}/still.png")
                compose(fig, panel_layer(ex, i, key[3]), 0.5).save(f"still3d_{i}_{k}.png")
                print("still", i, k, flush=True)
        return

    for i in ids:
        ex = mv.EXERCISES[i - 1]
        seq = mv.ex_frames(ex)
        sc.set_props(ex["props"], ex["view"])
        out = f"seg3d_{i}.mp4"
        writer = imageio.get_writer(out, fps=mv.FPS, codec="libx264", quality=8, macro_block_size=16,
                                    ffmpeg_params=["-pix_fmt", "yuv420p"])
        nxt = mv.finalize(mv.next_card(i, ex))
        for _ in range(int(1.8 * mv.FPS)):
            writer.append_data(nxt)
        panels = {}
        for f, (j, cap, hl) in enumerate(seq):
            if cap not in panels:
                panels[cap] = panel_layer(ex, i, cap)
            sc.set_highlight(hl)
            pulse = 0.5 + 0.5 * math.sin(f / mv.FPS * 2 * math.pi * 0.8)
            sc.pose(j, ex["view"], pulse)
            if f % STEP == 0:
                fig = sc.render(f"{TMP}/f.png")
            writer.append_data(np.asarray(compose(fig, panels[cap], (f + 1) / len(seq))))
            if f % 30 == 0:
                print(f"ex{i} frame {f}/{len(seq)}", flush=True)
        writer.close()
        print(f"ex{i} DONE", flush=True)


if __name__ == "__main__":
    main()
