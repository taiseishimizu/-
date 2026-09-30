"""バドミントン向け 可動域UPストレッチ動画ジェネレーター

棒人間アニメーションで各ストレッチを再現し、伸ばす部位を赤でハイライト、
右パネルに意識ポイントを表示した MP4 を出力する。

usage: python3 make_video.py [--preview]
"""
import math
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720
SS = 2  # supersampling
FPS = 30
FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"

BG = (246, 243, 236)
INK = (46, 52, 64)
FAR = (150, 158, 170)
RED = (229, 57, 53)
PANEL = (255, 255, 255)
MUTED = (110, 116, 128)
ACCENT = (30, 90, 160)

# 体節の長さ（world 単位 ≒ m）
L = dict(torso=0.52, head=0.18, ua=0.29, fa=0.27, th=0.44, sh=0.42, ft=0.16)
HEAD_R = 0.10

# 描画域: 左側フィギュア領域
FIG_W = 760
GROUND_PX = 610
SCALE = 270
ORIGIN_X = 380


def font(size):
    return ImageFont.truetype(FONT, size * SS)


# ---------------------------------------------------------------- kinematics
def vec(a, l):
    r = math.radians(a)
    return (l * math.cos(r), l * math.sin(r))


def add(p, q):
    return (p[0] + q[0], p[1] + q[1])


def ik(root, target, l1, l2, s):
    dx, dy = target[0] - root[0], target[1] - root[1]
    d = max(1e-6, math.hypot(dx, dy))
    d = min(d, l1 + l2 - 1e-4)
    a = math.atan2(dy, dx)
    c = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
    alpha = math.acos(max(-1, min(1, c)))
    mid = (root[0] + l1 * math.cos(a + s * alpha), root[1] + l1 * math.sin(a + s * alpha))
    b = math.atan2(target[1] - mid[1], target[0] - mid[0])
    end = (mid[0] + l2 * math.cos(b), mid[1] + l2 * math.sin(b))
    return mid, end


def limb(root, spec, l1, l2):
    kind = spec[0]
    if kind == "ang":
        m = add(root, vec(spec[1], l1))
        return m, add(m, vec(spec[2], l2))
    if kind == "ik":
        return ik(root, spec[1], l1, l2, spec[2])
    if kind == "pts":
        return spec[1], spec[2]
    raise ValueError(kind)


def solve(pose):
    """pose dict -> joints dict"""
    j = {}
    P = pose["P"]
    t = pose["torso"]
    j["P"] = P
    j["N"] = add(P, vec(t, L["torso"]))
    j["Hc"] = add(j["N"], vec(pose.get("head", t), L["head"]))
    sw, hw = pose.get("sw", 0.0), pose.get("hw", 0.0)
    j["S1"] = add(j["N"], vec(t - 90, sw))
    j["S2"] = add(j["N"], vec(t + 90, sw))
    j["H1"] = add(P, vec(t - 90, hw))
    j["H2"] = add(P, vec(t + 90, hw))
    for i in "12":
        j["E" + i], j["W" + i] = limb(j["S" + i], pose["arm" + i], L["ua"], L["fa"])
        j["K" + i], j["A" + i] = limb(j["H" + i], pose["leg" + i], L["th"], L["sh"])
        fl = pose.get("fl", L["ft"])
        j["T" + i] = add(j["A" + i], vec(pose.get("ft" + i, -25), fl))
    return j


SEGS = [("P", "N"), ("N", "Hc"), ("N", "S1"), ("N", "S2"),
        ("S1", "E1"), ("E1", "W1"), ("S2", "E2"), ("E2", "W2"),
        ("P", "H1"), ("P", "H2"),
        ("H1", "K1"), ("K1", "A1"), ("A1", "T1"),
        ("H2", "K2"), ("K2", "A2"), ("A2", "T2")]


def to_polar(j):
    out = {}
    for a, b in SEGS:
        dx, dy = j[b][0] - j[a][0], j[b][1] - j[a][1]
        out[(a, b)] = (math.degrees(math.atan2(dy, dx)), math.hypot(dx, dy))
    return out


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_ang(a, b, t):
    d = (b - a + 180) % 360 - 180
    return a + d * t


def ease(t):
    return t * t * (3 - 2 * t)


def blend(j0, j1, t, plant):
    p0, p1 = to_polar(j0), to_polar(j1)
    j = {"P": (lerp(j0["P"][0], j1["P"][0], t), lerp(j0["P"][1], j1["P"][1], t))}
    lens = {}
    for a, b in SEGS:
        (a0, l0), (a1, l1) = p0[(a, b)], p1[(a, b)]
        if l0 < 1e-4:
            a0 = a1
        if l1 < 1e-4:
            a1 = a0
        ang, ln = lerp_ang(a0, a1, t), lerp(l0, l1, t)
        lens[(a, b)] = ln
        j[b] = add(j[a], vec(ang, ln))
    # 接地している手足は目標位置に固定（足が滑らないように IK で補正）
    for name in plant:
        i = name[-1]
        if name.startswith("leg"):
            r, m, e, toe = "H" + i, "K" + i, "A" + i, "T" + i
        else:
            r, m, e, toe = "S" + i, "E" + i, "W" + i, None
        tgt = (lerp(j0[e][0], j1[e][0], t), lerp(j0[e][1], j1[e][1], t))
        l1, l2 = lens[(r, m)], lens[(m, e)]
        # 現在の関節の曲がり方向を維持
        cr = ((j[m][0] - j[r][0]) * (j[e][1] - j[r][1]) - (j[m][1] - j[r][1]) * (j[e][0] - j[r][0]))
        s = -1 if cr > 0 else 1
        mid, end = ik(j[r], tgt, l1, l2, s)
        foot = None
        if toe:
            foot = (j[toe][0] - j[e][0], j[toe][1] - j[e][1])
        j[m], j[e] = mid, end
        if toe:
            j[toe] = add(end, foot)
    return j


# ---------------------------------------------------------------- drawing
def wp(p):
    return ((ORIGIN_X + p[0] * SCALE) * SS, (GROUND_PX - p[1] * SCALE) * SS)


def seg_ids():
    return {"torso": ("P", "N"), "ua1": ("S1", "E1"), "fa1": ("E1", "W1"),
            "ua2": ("S2", "E2"), "fa2": ("E2", "W2"), "th1": ("H1", "K1"),
            "sh1": ("K1", "A1"), "ft1": ("A1", "T1"), "th2": ("H2", "K2"),
            "sh2": ("K2", "A2"), "ft2": ("A2", "T2")}


def line(d, a, b, col, w):
    pa, pb = wp(a), wp(b)
    d.line([pa, pb], fill=col, width=int(w * SS))
    r = w * SS / 2
    for p in (pa, pb):
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=col)


def draw_props(d, props):
    for pr in props:
        if pr[0] == "wall":
            x0 = wp((pr[1], 0))[0]
            d.rectangle([x0, 60 * SS, x0 + 26 * SS, GROUND_PX * SS], fill=(214, 206, 192))
            d.text((x0 + 4 * SS, 70 * SS), "壁", font=font(18), fill=MUTED)
        if pr[0] == "backwall":
            d.rectangle([wp((-0.75, 0))[0], 50 * SS, wp((0.75, 0))[0], GROUND_PX * SS],
                        fill=(232, 225, 212))
            d.text((wp((-0.72, 0))[0], 58 * SS), "壁に背中をつける", font=font(18), fill=MUTED)
        if pr[0] == "mat":
            d.rounded_rectangle([wp((pr[1], 0))[0], (GROUND_PX - 6) * SS,
                                 wp((pr[2], 0))[0], (GROUND_PX + 4) * SS],
                                radius=4 * SS, fill=(120, 160, 200))
    d.line([(20 * SS, GROUND_PX * SS), ((FIG_W - 20) * SS, GROUND_PX * SS)], fill=(170, 160, 145),
           width=3 * SS)


def draw_figure(img, j, view, hl, pulse):
    d = ImageDraw.Draw(img)
    ids = seg_ids()
    far_col = FAR if view == "side" else INK
    near_w, far_w, torso_w = 17, 15, 26

    # 奥側の手足
    for s in ("th2", "sh2", "ft2", "ua2", "fa2"):
        a, b = ids[s]
        line(d, j[a], j[b], far_col, far_w)
    # 胴体
    if view == "front":
        poly = [wp(j["S1"]), wp(j["S2"]), wp(j["H2"]), wp(j["H1"])]
        d.polygon(poly, fill=INK)
        line(d, j["S1"], j["S2"], INK, 16)
        line(d, j["H1"], j["H2"], INK, 16)
        line(d, j["P"], j["N"], INK, torso_w)
    else:
        line(d, j["P"], j["N"], INK, torso_w)
    line(d, j["N"], add(j["N"], ((j["Hc"][0] - j["N"][0]) * 0.4, (j["Hc"][1] - j["N"][1]) * 0.4)), INK, 12)
    c = wp(j["Hc"])
    r = HEAD_R * SCALE * SS
    d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=INK)
    # 手前の手足
    for s in ("th1", "sh1", "ft1", "ua1", "fa1"):
        a, b = ids[s]
        line(d, j[a], j[b], INK, near_w)

    if not hl:
        return img
    # 赤ハイライト（グロー＋本体）
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    g = ImageDraw.Draw(glow)
    alpha = int(110 + 90 * pulse)
    for s in hl:
        if s.startswith("j:"):
            p = wp(j[s[2:]])
            rr = (26 + 6 * pulse) * SS
            g.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], fill=RED + (alpha,))
        else:
            a, b = ids[s]
            w = (torso_w if s == "torso" else near_w) + 16 + 6 * pulse
            line(g, j[a], j[b], RED + (alpha,), w)
    glow = glow.filter(ImageFilter.GaussianBlur(8 * SS))
    img.alpha_composite(glow)
    d = ImageDraw.Draw(img)
    for s in hl:
        if s.startswith("j:"):
            p = wp(j[s[2:]])
            rr = 11 * SS
            d.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], fill=RED)
        else:
            a, b = ids[s]
            w = (torso_w if s == "torso" else (far_w if s.endswith("2") else near_w))
            line(d, j[a], j[b], RED, w)
    # 頭は常に最前面（赤い胴体に隠れないように）
    if "torso" in hl:
        d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=INK)
    return img


def wrap(text, fnt, width):
    lines, cur = [], ""
    for ch in text:
        if fnt.getlength(cur + ch) > width and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def draw_panel(img, ex, idx, total, caption, prog):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 780, 24, 1256, 690
    d.rounded_rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], radius=18 * SS, fill=PANEL)
    x = (x0 + 24) * SS
    y = (y0 + 20) * SS
    d.text((x, y), f"{idx}/{total}  {ex['area_no']}  {ex['timing']}", font=font(18), fill=MUTED)
    y += 34 * SS
    for ln in wrap(ex["title"], font(32), (x1 - x0 - 48) * SS):
        d.text((x, y), ln, font=font(32), fill=INK, stroke_width=1, stroke_fill=INK)
        y += 42 * SS
    y += 6 * SS
    # 伸ばす部位
    bx1 = (x1 - 24) * SS
    lines = wrap("伸ばす部位： " + ex["target"], font(20), bx1 - x - 40 * SS)
    bh = (16 + 28 * len(lines)) * SS
    d.rounded_rectangle([x, y, bx1, y + bh], radius=10 * SS, fill=(253, 232, 231))
    d.ellipse([x + 12 * SS, y + 14 * SS, x + 26 * SS, y + 28 * SS], fill=RED)
    ty = y + 8 * SS
    for ln in lines:
        d.text((x + 34 * SS, ty), ln, font=font(20), fill=(170, 30, 30))
        ty += 28 * SS
    y += bh + 12 * SS
    d.text((x, y), "回数： " + ex["reps"], font=font(20), fill=ACCENT, stroke_width=1, stroke_fill=ACCENT)
    y += 40 * SS
    d.text((x, y), "意識するポイント", font=font(22), fill=INK, stroke_width=1, stroke_fill=INK)
    y += 36 * SS
    for k, pt in enumerate(ex["points"], 1):
        d.ellipse([x, y + 1 * SS, x + 26 * SS, y + 27 * SS], fill=INK)
        d.text((x + 8 * SS, y + 3 * SS), str(k), font=font(18), fill=PANEL)
        for ln in wrap(pt, font(20), (x1 - x0 - 90) * SS):
            d.text((x + 38 * SS, y + 2 * SS), ln, font=font(20), fill=INK)
            y += 29 * SS
        y += 12 * SS
    # 凡例
    d.rectangle([x, (y1 - 40) * SS, x + 34 * SS, (y1 - 28) * SS], fill=RED)
    d.text((x + 44 * SS, (y1 - 45) * SS), "赤 ＝ 伸ばす（意識する）部位", font=font(17), fill=MUTED)

    # 動作キャプション（フィギュア上部）
    if caption:
        cf = font(26)
        tw = cf.getlength(caption)
        cx = (FIG_W / 2) * SS
        d.rounded_rectangle([cx - tw / 2 - 20 * SS, 20 * SS, cx + tw / 2 + 20 * SS, 66 * SS],
                            radius=12 * SS, fill=INK)
        d.text((cx - tw / 2, 28 * SS), caption, font=cf, fill=(255, 255, 255))
    # 進捗バー
    d.rectangle([0, (H - 10) * SS, W * SS, H * SS], fill=(225, 220, 210))
    d.rectangle([0, (H - 10) * SS, W * SS * prog, H * SS], fill=RED)


# ---------------------------------------------------------------- exercises
F = -25  # 右向きのフラットな足
KNEE_B, ANK_B = (-0.27, 0.06), (-0.68, 0.08)

EXERCISES = []

# 1. ニー・トゥ・ウォール
_a = dict(P=(0.02, 0.80), torso=86, leg1=("ik", (0.30, 0.07), 1), ft1=F,
          leg2=("ik", (-0.35, 0.10), 1), ft2=-45,
          arm1=("ik", (0.60, 1.22), -1), arm2=("ik", (0.60, 1.17), -1))
_b = dict(_a, P=(0.18, 0.62), torso=80)
EXERCISES.append(dict(
    title="ニー・トゥ・ウォール", area_no="① 足首", timing="練習前・動的",
    target="足首（ふくらはぎ〜アキレス腱）", reps="左右各10回 × 2セット",
    points=["前足のかかとは絶対に床から浮かせない",
            "膝は人差し指〜中指の方向へまっすぐ出す",
            "楽にできたら、つま先を壁から1cmずつ遠ざける"],
    view="side", props=[("wall", 0.62)], plant=["leg1", "leg2", "arm1", "arm2"],
    keys=[(_a, 0.0, 0.8, "スタート：前足のかかとをベタっと床に", ["sh1", "ft1", "j:A1"]),
          (_b, 1.5, 1.5, "膝を壁へ押し出す（かかとは浮かせない）", ["sh1", "ft1", "j:A1"]),
          (_a, 1.2, 0.4, "ゆっくり戻す", ["sh1", "ft1", "j:A1"])], cycles=3))

# 2. 90/90 ヒップスイッチ（正面）
_arm = dict(arm1=("pts", (0.33, 0.40), (0.44, 0.10)), arm2=("pts", (-0.33, 0.40), (-0.44, 0.10)))
_c = dict(P=(0, 0.14), torso=90, sw=0.17, hw=0.11, fl=0.07, ft1=-40, ft2=-140,
          leg1=("pts", (0.30, 0.50), (0.45, 0.07)), leg2=("pts", (-0.30, 0.50), (-0.45, 0.07)), **_arm)
_r = dict(_c, leg1=("pts", (0.58, 0.22), (0.45, 0.07)), leg2=("pts", (0.05, 0.27), (-0.45, 0.07)), torso=88)
_l = dict(_c, leg1=("pts", (-0.05, 0.27), (0.45, 0.07)), leg2=("pts", (-0.58, 0.22), (-0.45, 0.07)), torso=92)
_hl = ["th1", "th2", "j:H1", "j:H2"]
EXERCISES.append(dict(
    title="90/90 ヒップスイッチ", area_no="② 股関節", timing="練習前・動的",
    target="股関節（お尻の奥・太ももの付け根）内旋／外旋", reps="左右交互10往復 × 2セット",
    points=["骨盤を立て、背すじを伸ばしたまま倒す",
            "内側に倒れる脚の付け根（内旋）を特に感じる",
            "痛みの手前で止め、反動は使わない"],
    view="front", props=[("mat", -0.8, 0.8)], plant=["leg1", "leg2", "arm1", "arm2"],
    keys=[(_c, 0.0, 0.6, "背すじを伸ばして座る", _hl),
          (_r, 1.4, 1.2, "両膝を右へパタンと倒す", _hl),
          (_c, 1.0, 0.2, "中央へ", _hl),
          (_l, 1.4, 1.2, "左へ倒す", _hl),
          (_c, 1.0, 0.2, "中央へ", _hl)], cycles=2))

# 3. ワールドグレイテストストレッチ
_back = ("pts", KNEE_B, ANK_B)
_a = dict(P=(0.0, 0.44), torso=85, leg1=("ik", (0.45, 0.07), 1), ft1=F, leg2=_back, ft2=200,
          arm1=("ang", -70, -60), arm2=("ang", -80, -70))
_b = dict(_a, P=(0.0, 0.36), torso=22, head=15,
          arm1=("pts", (0.44, 0.30), (0.50, 0.10)), arm2=("ik", (0.38, 0.03), 1))
_cc = dict(_b, torso=32, head=70, arm1=("ang", 95, 95))
EXERCISES.append(dict(
    title="ワールドグレイテスト・ストレッチ", area_no="②③ 股関節＋胸椎", timing="練習前・動的",
    target="後ろ脚の股関節前面 ＋ 胸椎（背中の上部）の回旋", reps="左右各5回（各姿勢2〜3秒）",
    points=["後ろ脚のお尻を締め、付け根の前側を伸ばす",
            "肘を前足の内側・床へできるだけ近づける",
            "腕を天井へ、目線は指先。胸から開いて捻る"],
    view="side", props=[], plant=["leg1", "leg2"],
    keys=[(_a, 0.0, 0.8, "大きく踏み込む", ["th2", "j:H2"]),
          (_b, 1.3, 1.5, "肘を前足の内側へ沈める", ["th2", "j:H2"]),
          (_cc, 1.5, 1.8, "腕を天井へ・目線は指先", ["torso", "ua1", "th2"]),
          (_b, 1.2, 0.3, "手を床へ戻す", ["th2", "j:H2"]),
          (_a, 1.2, 0.5, "上体を起こす", ["th2", "j:H2"])], cycles=2))

# 4. コサックスクワット（正面）
_c = dict(P=(0, 0.78), torso=90, sw=0.17, hw=0.11, fl=0.07, ft1=-30, ft2=-150,
          leg1=("ik", (0.55, 0.07), 1), leg2=("ik", (-0.55, 0.07), -1),
          arm1=("ang", -100, 145), arm2=("ang", -80, 35))
_r = dict(_c, P=(0.36, 0.33), torso=95, leg1=("pts", (0.66, 0.42), (0.55, 0.07)),
          leg2=("ik", (-0.55, 0.07), -1), ft2=100, fl=0.09)
_l = dict(_c, P=(-0.36, 0.33), torso=85, leg2=("pts", (-0.66, 0.42), (-0.55, 0.07)),
          leg1=("ik", (0.55, 0.07), 1), ft1=80, fl=0.09)
EXERCISES.append(dict(
    title="コサックスクワット", area_no="② 股関節", timing="練習前・動的",
    target="伸ばした脚の内もも（内転筋）＋ 曲げた側の股関節", reps="左右交互8回 × 2セット",
    points=["曲げる側のかかとを浮かせず、真下にお尻を落とす",
            "伸ばした脚はつま先を天井へ向ける",
            "胸を張り、膝とつま先の向きをそろえる"],
    view="front", props=[], plant=["leg1", "leg2"],
    keys=[(_c, 0.0, 0.6, "足を大きく開いて立つ", ["th1", "th2"]),
          (_r, 1.5, 1.5, "右へ深く沈む（左足つま先は天井）", ["th2", "j:H1"]),
          (_c, 1.2, 0.3, "中央へ", ["th1", "th2"]),
          (_l, 1.5, 1.5, "左へ深く沈む", ["th1", "j:H2"]),
          (_c, 1.2, 0.3, "中央へ", ["th1", "th2"])], cycles=2))

# 5. 四つ這い 胸椎回旋
_legs = dict(leg1=("pts", (-0.26, 0.06), (-0.68, 0.07)), leg2=("pts", (-0.22, 0.06), (-0.64, 0.07)),
             ft1=180, ft2=180, fl=0.14)
_a = dict(P=(-0.25, 0.50), torso=2, head=-5, arm2=("ik", (0.27, 0.03), 1),
          arm1=("pts", (0.20, 0.25), (0.38, 0.45)), **_legs)
_b = dict(_a, head=40, arm1=("pts", (0.22, 0.80), (0.40, 0.62)))
EXERCISES.append(dict(
    title="四つ這い 胸椎ローテーション", area_no="③ 胸椎", timing="練習前・動的",
    target="胸椎（みぞおちの裏〜肩甲骨の間）の回旋", reps="左右各8回",
    points=["腰は固定。みぞおちの裏から回す",
            "肘 → 胸 → 目線の順に天井へ開く",
            "開き切ったら息を吐き切って2秒キープ"],
    view="side", props=[("mat", -0.85, 0.55)], plant=["arm2", "leg1", "leg2"],
    keys=[(_a, 0.0, 0.6, "手を頭の後ろ、肘を床の方へ", ["torso"]),
          (_b, 1.6, 1.8, "肘→胸→目線の順で天井へ", ["torso"]),
          (_a, 1.4, 0.3, "ゆっくり閉じる", ["torso"])], cycles=3))

# 6. ウォールエンジェル（正面）
_w = dict(P=(0, 0.90), torso=90, sw=0.18, hw=0.10, fl=0.07, ft1=-30, ft2=-150,
          leg1=("ik", (0.16, 0.07), 1), leg2=("ik", (-0.16, 0.07), -1),
          arm1=("pts", (0.46, 1.24), (0.50, 1.50)), arm2=("pts", (-0.46, 1.24), (-0.50, 1.50)))
_y = dict(_w, arm1=("pts", (0.34, 1.68), (0.46, 1.92)), arm2=("pts", (-0.34, 1.68), (-0.46, 1.92)))
_hl = ["ua1", "ua2", "j:S1", "j:S2"]
EXERCISES.append(dict(
    title="ウォールエンジェル", area_no="④ 肩甲骨・肩", timing="練習前・動的",
    target="肩（外旋・挙上）＋ 肩甲骨まわり", reps="10回 × 2セット（上げ3秒・下げ3秒）",
    points=["後頭部・背中・お尻・手の甲を壁から離さない",
            "腰を反らさないよう、おへそを軽く引き込む",
            "下ろす時に肩甲骨を「下・内」へ寄せる"],
    view="front", props=[("backwall",)], plant=["leg1", "leg2"],
    keys=[(_w, 0.0, 0.8, "Wの形：手の甲と肘を壁につける", _hl),
          (_y, 2.2, 0.8, "壁から離さずYへスライド", _hl),
          (_w, 2.2, 0.6, "肩甲骨を下げながら戻す", _hl)], cycles=3))

# 7. 広背筋ストレッチ（肩の屈曲）
_legs = dict(leg1=("pts", (-0.26, 0.06), (-0.68, 0.07)), leg2=("pts", (-0.22, 0.06), (-0.64, 0.07)),
             ft1=180, ft2=180, fl=0.14)
_a = dict(P=(-0.25, 0.50), torso=-3, head=-10, arm1=("ik", (0.42, 0.03), -1),
          arm2=("ik", (0.46, 0.03), -1), **_legs)
_b = dict(_a, P=(-0.58, 0.32), torso=-8, head=-4,
          leg1=("pts", (-0.26, 0.06), (-0.68, 0.07)))
EXERCISES.append(dict(
    title="キャット・リーチ（広背筋）", area_no="④ 肩", timing="練習後・静的",
    target="脇の下〜広背筋（肩の屈曲・挙上）", reps="30秒 × 2セット（呼吸5回）",
    points=["手の小指側を床に押し、脇の下を床へ沈める",
            "お尻はかかとへ。みぞおちを床に近づける",
            "吐くたびに指先を遠くへ伸ばす"],
    view="side", props=[("mat", -0.85, 0.6)], plant=["arm1", "arm2", "leg1", "leg2"],
    keys=[(_a, 0.0, 0.8, "四つ這い：手を肩より前へ", ["ua1", "ua2", "j:S1"]),
          (_b, 2.5, 3.0, "お尻をかかとへ（実際は30秒キープ）", ["ua1", "ua2", "j:S1", "torso"]),
          (_a, 2.0, 0.5, "ゆっくり戻る", ["ua1", "ua2", "j:S1"])], cycles=2))

# 8. ランジ・ツイスト（連動）
_a = dict(P=(0.0, 0.74), torso=90, leg1=("ik", (0.35, 0.07), 1), ft1=F,
          leg2=("ik", (-0.35, 0.10), 1), ft2=-45, arm1=("ang", -90, -80), arm2=("ang", -95, -85))
_b = dict(_a, P=(-0.02, 0.50), torso=97, head=95, arm1=("ang", 140, 100), arm2=("ang", 55, 65))
_cc = dict(_a, P=(0.04, 0.48), torso=80, head=85, arm1=("ang", -30, -60), arm2=("ang", -120, -100))
_hl = ["torso", "th2", "j:H2"]
EXERCISES.append(dict(
    title="ランジ・ツイスト（連動）", area_no="⑤ 股関節＋体幹", timing="練習前・動的",
    target="後ろ脚の付け根 → 体幹 → 胸の捻り（連動）", reps="左右各8回",
    points=["骨盤 → 胸 → 腕の順に回す（腕から動かさない）",
            "前膝はつま先の真上でブレさせない",
            "胸を開く時に吸い、捻り切る時に吐く"],
    view="side", props=[], plant=["leg1", "leg2"],
    keys=[(_a, 0.0, 0.5, "スプリットスタンスで構える", _hl),
          (_b, 1.4, 1.0, "沈みながら胸を開く（吸う）", _hl),
          (_cc, 1.0, 1.0, "骨盤→胸→腕の順に捻る（吐く）", _hl),
          (_a, 1.2, 0.4, "元の姿勢へ", _hl)], cycles=3))


# ---------------------------------------------------------------- timeline
def ex_frames(ex):
    """(joints, caption, highlight) の列を返す"""
    keys = ex["keys"]
    solved = [solve(k[0]) for k in keys]
    seq = []
    order = list(range(len(keys))) * ex["cycles"]
    prev = None
    for i in order:
        _, trans, hold, cap, hl = keys[i]
        if prev is not None and trans > 0:
            n = int(trans * FPS)
            for f in range(n):
                t = ease((f + 1) / n)
                seq.append((blend(solved[prev], solved[i], t, ex["plant"]), cap, hl))
        for _ in range(int(hold * FPS)):
            seq.append((solved[i], cap, hl))
        prev = i
    return seq


def card(lines, sub=None):
    img = Image.new("RGBA", (W * SS, H * SS), BG + (255,))
    d = ImageDraw.Draw(img)
    y = 70 * SS
    for text, size, col in lines:
        f = font(size)
        d.text(((W * SS - f.getlength(text)) / 2, y), text, font=f, fill=col,
               stroke_width=1 if size >= 30 else 0, stroke_fill=col)
        y += int(size * 1.55) * SS
    return img


def intro_card():
    rows = [("バドミントン上級者のための", 30, MUTED),
            ("可動域UP ストレッチメニュー 8種", 46, INK),
            ("赤く光る部位 ＝ 伸ばす（意識する）場所", 24, RED),
            ("", 10, INK),
            ("1  ニー・トゥ・ウォール ……………… ① 足首（背屈）", 24, INK),
            ("2  90/90 ヒップスイッチ ……………… ② 股関節（内旋・外旋）", 24, INK),
            ("3  ワールドグレイテスト …………… ②③ 股関節屈曲＋胸椎回旋", 24, INK),
            ("4  コサックスクワット ………………… ② 股関節（切り返し）", 24, INK),
            ("5  四つ這い胸椎ローテーション …… ③ 胸椎（回旋）", 24, INK),
            ("6  ウォールエンジェル ………………… ④ 肩甲骨・肩（外旋・挙上）", 24, INK),
            ("7  キャット・リーチ …………………… ④ 肩（屈曲）※練習後", 24, INK),
            ("8  ランジ・ツイスト …………………… ⑤ 下半身→体幹→腕の連動", 24, INK),
            ("所要 約12〜15分 ／ 1〜6・8 は練習前（動的）、7 と各部位の静的30秒は練習後", 20, MUTED)]
    return card(rows)


def outro_card():
    rows = [("続けるコツと注意", 44, INK),
            ("", 8, INK),
            ("● 頻度：週4〜6回。可動域の変化は2〜4週間が目安", 26, INK),
            ("● 練習前：反動をつけない「動的」ストレッチでウォームアップ", 26, INK),
            ("● 練習後・入浴後：硬い部位を30〜60秒「静的」にキープ", 26, INK),
            ("● 呼吸は止めない。伸ばす時にゆっくり吐く", 26, INK),
            ("● 鋭い痛み・しびれが出たら即中止", 26, RED),
            ("  既往のケガがある部位は、医師・理学療法士に相談を", 22, MUTED)]
    return card(rows)


def next_card(i, ex):
    return card([("", 120, INK), (f"NEXT  {i}/{len(EXERCISES)}", 30, RED), (ex["title"], 44, INK),
                 (ex["area_no"] + "  |  " + ex["timing"], 26, MUTED)])


def finalize(img):
    return np.asarray(img.convert("RGB").resize((W, H), Image.LANCZOS))


def main():
    preview = "--preview" in sys.argv
    if preview:
        for i, ex in enumerate(EXERCISES, 1):
            for k, key in enumerate(ex["keys"]):
                img = Image.new("RGBA", (W * SS, H * SS), BG + (255,))
                draw_props(ImageDraw.Draw(img), ex["props"])
                draw_figure(img, solve(key[0]), ex["view"], key[4], 1.0)
                draw_panel(img, ex, i, len(EXERCISES), key[3], 0.5)
                Image.fromarray(finalize(img)).save(f"preview_{i}_{k}.png")
        Image.fromarray(finalize(intro_card())).save("preview_intro.png")
        Image.fromarray(finalize(outro_card())).save("preview_outro.png")
        return

    all_seqs = [ex_frames(ex) for ex in EXERCISES]
    total = sum(len(s) for s in all_seqs)
    writer = imageio.get_writer("badminton_mobility_stretch.mp4", fps=FPS, codec="libx264",
                                quality=8, macro_block_size=16,
                                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    still = finalize(intro_card())
    for _ in range(int(7 * FPS)):
        writer.append_data(still)
    done = 0
    for i, (ex, seq) in enumerate(zip(EXERCISES, all_seqs), 1):
        still = finalize(next_card(i, ex))
        for _ in range(int(1.8 * FPS)):
            writer.append_data(still)
        bg = Image.new("RGBA", (W * SS, H * SS), BG + (255,))
        draw_props(ImageDraw.Draw(bg), ex["props"])
        for f, (j, cap, hl) in enumerate(seq):
            img = bg.copy()
            pulse = 0.5 + 0.5 * math.sin(f / FPS * 2 * math.pi * 0.8)
            draw_figure(img, j, ex["view"], hl, pulse)
            done += 1
            draw_panel(img, ex, i, len(EXERCISES), cap, done / total)
            writer.append_data(finalize(img))
        print(f"exercise {i} done", flush=True)
    still = finalize(outro_card())
    for _ in range(int(7 * FPS)):
        writer.append_data(still)
    writer.close()


if __name__ == "__main__":
    main()
