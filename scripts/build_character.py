"""
Procedural Blender 5.2 builder for the Masked Brawler character (Pass 2 - High Fidelity):
- Squat, broad hunched brawler proportions matching front-3/4 & back-3/4 turnaround art
- Smooth-shaded organic forms (vest, arms, shorts) + crisp faceted hard forms (hockey mask, hair spikes, staples, boots)
- Pitch-black recessed 3D eye sockets, exposed orange brow/temples, swept green spiky hair tuft, painted back scars & cross-stitches
- Clean 21-bone game Armature + smooth analytical skinning (<=4 joints/vertex)
- Three NLA animations at 30 fps (no scale keys, clean additive-free spine chain):
    1. AN_Brute_Attack_Heavy_01 (heavy right-hand haymaker smash, 42f)
    2. AN_Brute_Attack_Combo_02 (left stapled-shoulder hook + right smash, 54f)
    3. AN_Brute_Idle (menacing heavy breathing loop, 60f)
"""

import math
import os
import struct
import json
import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix, Euler, Quaternion

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PUBLIC_DIR = ROOT_DIR
RENDERS_DIR = os.path.join(ROOT_DIR, "renders")
os.makedirs(RENDERS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Procedural Hand-Painted Texture Atlas (1024x1024 RGBA)
# ---------------------------------------------------------------------------
def srgb_hex(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float32) / 255.0


def draw_capsule_mask(xx: np.ndarray, yy: np.ndarray, p0, p1, radius: float, feather: float = 0.0018) -> np.ndarray:
    ax, ay = p0
    bx, by = p1
    abx, aby = bx - ax, by - ay
    apx, apy = xx - ax, yy - ay
    t = np.clip((apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-12), 0.0, 1.0)
    dx = apx - abx * t
    dy = apy - aby * t
    dist = np.sqrt(dx * dx + dy * dy)
    return np.clip((radius - dist) / max(feather, 1e-5), 0.0, 1.0)


def draw_tapered_slash(xx: np.ndarray, yy: np.ndarray, p0, p1, max_radius: float, feather: float = 0.0016) -> np.ndarray:
    ax, ay = p0
    bx, by = p1
    abx, aby = bx - ax, by - ay
    apx, apy = xx - ax, yy - ay
    t = np.clip((apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-12), 0.0, 1.0)
    profile = np.maximum(0.0, np.sin(np.pi * t)) ** 0.72
    dx = apx - abx * t
    dy = apy - aby * t
    dist = np.sqrt(dx * dx + dy * dy)
    return np.clip((max_radius * profile - dist) / max(feather, 1e-5), 0.0, 1.0)


def create_atlas_image(size: int = 1024) -> bpy.types.Image:
    u = (np.arange(size, dtype=np.float32) + 0.5) / size
    v = (np.arange(size, dtype=np.float32) + 0.5) / size
    uu, vv = np.meshgrid(u, v)
    rgb = np.zeros((size, size, 3), dtype=np.float32)

    # --- Quadrant TL: u in [0.0, 0.5], v in [0.5, 1.0] -> VEST (front u=[0..0.25], back u=[0.25..0.50]) ---
    vest_m = (uu <= 0.5) & (vv >= 0.5)
    vt = np.clip((vv - 0.5) / 0.5, 0.0, 1.0)
    c_bot = srgb_hex("#d84c24")
    c_mid = srgb_hex("#f48229")
    c_top = srgb_hex("#ffc238")
    vest_col = np.where(
        vt[..., None] < 0.52,
        c_bot + (c_mid - c_bot) * (vt[..., None] / 0.52),
        c_mid + (c_top - c_mid) * ((vt[..., None] - 0.52) / 0.48),
    )

    # Subtle side shading & warm reddish-orange V-cut armhole border near upper outer edges of vest
    u_front_dist = np.abs(uu - 0.125) / 0.105
    u_back_dist = np.abs(uu - 0.375) / 0.105
    side_d = np.where(uu < 0.25, u_front_dist, u_back_dist)
    vest_col = vest_col * (1.0 - 0.08 * np.clip(side_d, 0.0, 1.0)[..., None] ** 1.8)

    # Warm red-orange armhole collar trim on upper side edges (v > 0.76, side_d > 0.84)
    collar_m = np.clip((side_d - 0.83) / 0.10, 0.0, 1.0) * np.clip((vv - 0.76) / 0.08, 0.0, 1.0)
    vest_col = vest_col * (1.0 - 0.85 * collar_m[..., None]) + srgb_hex("#e64f25") * (0.85 * collar_m[..., None])

    # Front chest fold crease right under the mask chin (around u=0.125, v=0.735)
    crease_y = 0.735 + 0.020 * ((uu - 0.125) / 0.085) ** 2
    crease_m = (uu > 0.045) & (uu < 0.205)
    crease_w = np.clip(1.0 - np.abs(vv - crease_y) / 0.014, 0.0, 1.0) * crease_m
    vest_col = vest_col * (1.0 - 0.24 * crease_w[..., None]) + srgb_hex("#bf3e1b") * (0.24 * crease_w[..., None])

    # Back lower hem double-arch shadow (around u=0.375, v=0.555)
    bcrease = draw_tapered_slash(uu, vv, (0.315, 0.560), (0.375, 0.542), 0.008) + draw_tapered_slash(
        uu, vv, (0.375, 0.542), (0.435, 0.560), 0.008
    )
    bcrease = np.clip(bcrease, 0.0, 1.0)
    vest_col = vest_col * (1.0 - 0.20 * bcrease[..., None])

    # Back Scar 1 (Main diagonal gash: lower-left +X to upper-center-right -X in back view)
    s1_a, s1_b = (0.308, 0.665), (0.396, 0.850)
    scar1_outer = draw_tapered_slash(uu, vv, s1_a, s1_b, 0.014, 0.0028)
    scar1_core = draw_tapered_slash(uu, vv, s1_a, s1_b, 0.0082, 0.0012)
    vest_col = vest_col * (1.0 - 0.50 * scar1_outer[..., None]) + srgb_hex("#d93d26") * (0.50 * scar1_outer[..., None])
    vest_col = vest_col * (1.0 - 0.92 * scar1_core[..., None]) + srgb_hex("#9e1b1f") * (0.92 * scar1_core[..., None])

    d1 = np.array([s1_b[0] - s1_a[0], s1_b[1] - s1_a[1]])
    n1 = np.array([-d1[1], d1[0]])
    n1 = n1 / np.linalg.norm(n1) * 0.022
    for st_t in (0.20, 0.41, 0.62, 0.82):
        c = np.array([s1_a[0] + d1[0] * st_t, s1_a[1] + d1[1] * st_t])
        st_m = draw_capsule_mask(uu, vv, c - n1, c + n1, 0.0040, 0.0010)
        vest_col = vest_col * (1.0 - 0.94 * st_m[..., None]) + srgb_hex("#54271d") * (0.94 * st_m[..., None])

    # Back Scar 2 (Smaller opposing diagonal gash on lower-right of back, cleanly above bottom hem)
    s2_a, s2_b = (0.376, 0.688), (0.436, 0.602)
    scar2_outer = draw_tapered_slash(uu, vv, s2_a, s2_b, 0.011, 0.0024)
    scar2_core = draw_tapered_slash(uu, vv, s2_a, s2_b, 0.0065, 0.0012)
    vest_col = vest_col * (1.0 - 0.45 * scar2_outer[..., None]) + srgb_hex("#d63c26") * (0.45 * scar2_outer[..., None])
    vest_col = vest_col * (1.0 - 0.90 * scar2_core[..., None]) + srgb_hex("#9e1b1f") * (0.90 * scar2_core[..., None])

    d2 = np.array([s2_b[0] - s2_a[0], s2_b[1] - s2_a[1]])
    n2 = np.array([-d2[1], d2[0]])
    n2 = n2 / np.linalg.norm(n2) * 0.018
    for st_t in (0.24, 0.52, 0.80):
        c = np.array([s2_a[0] + d2[0] * st_t, s2_a[1] + d2[1] * st_t])
        st_m = draw_capsule_mask(uu, vv, c - n2, c + n2, 0.0036, 0.0010)
        vest_col = vest_col * (1.0 - 0.94 * st_m[..., None]) + srgb_hex("#54271d") * (0.94 * st_m[..., None])

    rgb[vest_m] = vest_col[vest_m]

    # --- Quadrant TR: u in [0.5, 1.0], v in [0.5, 1.0] -> MASK, EYE BLACK, RED MARKINGS, NOSE ---
    mask_m = (uu > 0.5) & (uu <= 0.75) & (vv >= 0.65)
    mt = np.clip((vv - 0.65) / 0.35, 0.0, 1.0)
    mask_col = srgb_hex("#cfd4de") + (srgb_hex("#f7f6f2") - srgb_hex("#cfd4de")) * (mt[..., None] ** 0.65)
    rgb[mask_m] = mask_col[mask_m]

    rgb[(uu > 0.75) & (uu <= 0.90) & (vv >= 0.80)] = srgb_hex("#07080b")  # Deep pitch black eye sockets
    rgb[(uu > 0.75) & (uu <= 0.90) & (vv >= 0.65) & (vv < 0.80)] = srgb_hex("#b51c20")  # Crimson mask markings
    rgb[(uu > 0.90) & (vv >= 0.80)] = srgb_hex("#918375")  # Nose shadow warm grey
    rgb[(uu > 0.90) & (vv >= 0.65) & (vv < 0.80)] = srgb_hex("#e34b24")
    rgb[(uu > 0.50) & (vv >= 0.50) & (vv < 0.65)] = srgb_hex("#54271d")

    # --- Quadrant BL: u in [0.0, 0.5], v in [0.0, 0.5] -> SKIN / ARMS / HEAD / BOOTS ---
    skin_m = (uu <= 0.35) & (vv < 0.5)
    st = np.clip(vv / 0.5, 0.0, 1.0)
    s_bot = srgb_hex("#da4e24")
    s_mid = srgb_hex("#f37f29")
    s_top = srgb_hex("#ffbe38")
    skin_col = np.where(
        st[..., None] < 0.5,
        s_bot + (s_mid - s_bot) * (st[..., None] / 0.5),
        s_mid + (s_top - s_mid) * ((st[..., None] - 0.5) / 0.5),
    )
    rgb[skin_m] = skin_col[skin_m]

    boot_m = (uu > 0.35) & (uu <= 0.50) & (vv >= 0.25) & (vv < 0.50)
    bt = np.clip((vv - 0.25) / 0.25, 0.0, 1.0)
    rgb[boot_m] = (srgb_hex("#bf3c1b") + (srgb_hex("#e65b28") - srgb_hex("#bf3c1b")) * bt[..., None])[boot_m]

    crease_sw_m = (uu > 0.35) & (uu <= 0.50) & (vv < 0.25)
    rgb[crease_sw_m] = srgb_hex("#993016")

    # --- Quadrant BR: u in [0.5, 1.0], v in [0.0, 0.5] -> SHORTS, GREEN HAIR, METAL STAPLES ---
    sh_m = (uu > 0.50) & (uu <= 0.72) & (vv < 0.50)
    sht = np.clip(vv / 0.50, 0.0, 1.0)
    rgb[sh_m] = (srgb_hex("#182e31") + (srgb_hex("#2d5458") - srgb_hex("#182e31")) * sht[..., None])[sh_m]

    hr_m = (uu > 0.72) & (uu <= 0.86) & (vv < 0.50)
    hrt = np.clip(vv / 0.50, 0.0, 1.0)
    rgb[hr_m] = (srgb_hex("#224f49") + (srgb_hex("#4b9671") - srgb_hex("#224f49")) * (hrt[..., None] ** 0.82))[hr_m]

    mt_m = (uu > 0.86) & (vv < 0.50)
    mtt = np.clip(vv / 0.50, 0.0, 1.0)
    rgb[mt_m] = (srgb_hex("#3a3f4e") + (srgb_hex("#7f8aa1") - srgb_hex("#3a3f4e")) * mtt[..., None])[mt_m]

    rgba = np.ones((size, size, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(rgb, 0.0, 1.0)

    img = bpy.data.images.new("BrawlerAtlas", width=size, height=size, alpha=False)
    img.colorspace_settings.name = "sRGB"
    img.pixels.foreach_set(rgba.ravel())
    img.pack()

    atlas_path = os.path.join(RENDERS_DIR, "brawler_atlas.png")
    img.filepath_raw = atlas_path
    img.file_format = "PNG"
    img.save()
    return img


# ---------------------------------------------------------------------------
# 2. BMesh Geometry Helpers & Part Builders
# ---------------------------------------------------------------------------
class MeshPartBuilder:
    def __init__(self):
        self.bm = bmesh.new()
        self.uv_layer = self.bm.loops.layers.uv.new("UVMap")
        self.vert_weights = []

    def add_vert(self, co, weights: dict):
        v = self.bm.verts.new(co)
        total = sum(weights.values())
        if total <= 1e-8:
            norm_w = {"Pelvis": 1.0}
        else:
            items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:4]
            s = sum(val for _, val in items)
            norm_w = {k: val / s for k, val in items if val > 1e-5}
        self.vert_weights.append(norm_w)
        return v

    def add_face(self, verts, uvs, smooth: bool = True):
        try:
            f = self.bm.faces.new(verts)
        except ValueError:
            return None
        f.smooth = smooth
        for loop, uv in zip(f.loops, uvs):
            loop[self.uv_layer].uv = uv
        return f

    def add_loft(self, rings_xyz, rings_uv, rings_weights, cap_start=False, cap_end=False, closed=True, smooth=True):
        v_rings = []
        for pts, w_list in zip(rings_xyz, rings_weights):
            v_rings.append([self.add_vert(p, w) for p, w in zip(pts, w_list)])

        n_rings = len(v_rings)
        m = len(v_rings[0])
        seg_count = m if closed else m - 1

        for r in range(n_rings - 1):
            for i in range(seg_count):
                i_next = (i + 1) % m
                self.add_face(
                    [v_rings[r][i], v_rings[r][i_next], v_rings[r + 1][i_next], v_rings[r + 1][i]],
                    [rings_uv[r][i], rings_uv[r][i_next], rings_uv[r + 1][i_next], rings_uv[r + 1][i]],
                    smooth=smooth,
                )

        if cap_start:
            c_pt = np.mean(np.array(rings_xyz[0]), axis=0)
            c_uv = np.mean(np.array(rings_uv[0]), axis=0)
            vc = self.add_vert(tuple(c_pt), rings_weights[0][0])
            for i in range(seg_count):
                i_next = (i + 1) % m
                self.add_face([vc, v_rings[0][i_next], v_rings[0][i]], [tuple(c_uv), rings_uv[0][i_next], rings_uv[0][i]], smooth=smooth)

        if cap_end:
            c_pt = np.mean(np.array(rings_xyz[-1]), axis=0)
            c_uv = np.mean(np.array(rings_uv[-1]), axis=0)
            vc = self.add_vert(tuple(c_pt), rings_weights[-1][0])
            for i in range(seg_count):
                i_next = (i + 1) % m
                self.add_face([vc, v_rings[-1][i], v_rings[-1][i_next]], [tuple(c_uv), rings_uv[-1][i], rings_uv[-1][i_next]], smooth=smooth)
        return v_rings


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def spine_weights_for_z(z: float) -> dict:
    # Wide, gentle overlap across Pelvis -> Spine -> Chest so torso twist is butter-smooth!
    w_pelvis = 1.0 - smoothstep(0.40, 0.82, z)
    w_chest = smoothstep(0.75, 1.20, z)
    w_spine = max(0.0, 1.0 - w_pelvis - w_chest)
    return {"Pelvis": w_pelvis, "Spine": w_spine, "Chest": w_chest}


def superellipse_point(theta: float, rx: float, ry_front: float, ry_back: float, power_front: float, power_back: float) -> tuple:
    ct = math.cos(theta)
    st = math.sin(theta)
    pwr = power_front if st < 0.0 else power_back
    ry = ry_front if st < 0.0 else ry_back
    ex = math.copysign(abs(ct) ** (2.0 / pwr), ct) * rx
    ey = math.copysign(abs(st) ** (2.0 / pwr), st) * ry
    return ex, ey


# Squat, broad hunched brawler torso profile:
# Wide round potbelly in front, broad boxy hunched shoulder wall in back!
TORSO_SPECS = [
    # (z,   rx,   y_center, ry_front, ry_back, pwr_f, pwr_b, front_dip, back_cleft)
    (0.40, 0.46, -0.04, 0.36, 0.32, 2.0, 2.3, 0.03, 0.01),
    (0.46, 0.56, -0.06, 0.47, 0.39, 2.0, 2.6, 0.07, 0.025),  # Broad potbelly overhang!
    (0.60, 0.58, -0.08, 0.50, 0.41, 2.05, 2.8, 0.05, 0.015),
    (0.76, 0.56, -0.06, 0.47, 0.42, 2.1, 3.0, 0.01, 0.0),
    (0.92, 0.53, -0.03, 0.41, 0.42, 2.2, 3.2, 0.00, 0.0),    # Chest crease under mask
    (1.09, 0.54, +0.01, 0.38, 0.41, 2.4, 3.4, 0.00, 0.0),
    (1.25, 0.52, +0.04, 0.32, 0.36, 2.7, 3.6, 0.00, 0.0),    # Broad boxy upper vest shoulders!
    (1.38, 0.46, +0.06, 0.23, 0.28, 3.0, 3.8, 0.00, 0.0),
    (1.44, 0.36, +0.07, 0.13, 0.17, 3.1, 3.8, 0.00, 0.0),    # Flat-topped hunch ridge
]


def build_torso(b: MeshPartBuilder):
    m = 28
    rings_xyz, rings_w = [], []
    n_r = len(TORSO_SPECS)

    for z0, rx, yc, ry_f, ry_b, pf, pb_exp, f_dip, b_cleft in TORSO_SPECS:
        pts, ws = [], []
        for i in range(m):
            theta = (2.0 * math.pi * i) / m
            ex, ey = superellipse_point(theta, rx, ry_f, ry_b, pf, pb_exp)
            x = ex
            y = yc + ey
            z = z0

            frontness = max(0.0, -math.sin(theta))
            z -= f_dip * (frontness ** 1.6)

            backness = max(0.0, math.sin(theta))
            if b_cleft > 0.0:
                y -= b_cleft * (backness ** 4.0)

            # Sleeveless V-taper near upper armholes so shoulder straps frame the massive deltoids
            if z0 >= 1.05:
                arm_t = smoothstep(1.04, 1.34, z0)
                x *= 1.0 - 0.10 * arm_t * (abs(math.cos(theta)) ** 2.2)

            pts.append((x, y, z))
            ws.append(spine_weights_for_z(z))

        rings_xyz.append(pts)
        rings_w.append(ws)

    v_rings = [[b.add_vert(p, w) for p, w in zip(pts, ws)] for pts, ws in zip(rings_xyz, rings_w)]

    for r in range(n_r - 1):
        for i in range(m):
            i_next = (i + 1) % m
            is_back = i < (m // 2)
            quad_vs = [v_rings[r][i], v_rings[r][i_next], v_rings[r + 1][i_next], v_rings[r + 1][i]]
            quad_uvs = []
            for rr, ii in [(r, i), (r, i_next), (r + 1, i_next), (r + 1, i)]:
                px = rings_xyz[rr][ii][0]
                rx_r = TORSO_SPECS[rr][1]
                vt_r = 0.52 + 0.46 * (rr / (n_r - 1))
                if not is_back:
                    u_c = 0.125 + 0.100 * (px / max(rx_r, 1e-4))
                else:
                    u_c = 0.375 - 0.100 * (px / max(rx_r, 1e-4))
                quad_uvs.append((u_c, vt_r))
            b.add_face(quad_vs, quad_uvs, smooth=True)

    c_bot = b.add_vert((0.0, -0.04, 0.39), {"Pelvis": 1.0})
    for i in range(m):
        i_next = (i + 1) % m
        b.add_face([c_bot, v_rings[0][i_next], v_rings[0][i]], [(0.125, 0.52)] * 3, smooth=True)

    c_top = b.add_vert((0.0, +0.06, 1.455), {"Chest": 1.0})
    for i in range(m):
        i_next = (i + 1) % m
        b.add_face([c_top, v_rings[-1][i], v_rings[-1][i_next]], [(0.125, 0.98)] * 3, smooth=True)


def build_shoulder_staples(b: MeshPartBuilder):
    """Two chunky dark-steel polygonal C-clamps hooked over the character's Left (+X) shoulder."""
    clamps = [
        # Front-lower clamp on +X shoulder strap
        [
            (+0.33, -0.15, 1.38),
            (+0.36, -0.16, 1.455),
            (+0.48, -0.15, 1.455),
            (+0.55, -0.12, 1.37),
            (+0.50, -0.11, 1.29),
        ],
        # Rear-higher clamp on +X shoulder strap
        [
            (+0.31, +0.02, 1.40),
            (+0.35, +0.03, 1.505),
            (+0.49, +0.03, 1.500),
            (+0.57, +0.02, 1.39),
            (+0.52, +0.01, 1.30),
        ],
    ]
    w_clamp = {"Chest": 0.88, "Clavicle.L": 0.12}
    for path in clamps:
        pts = [Vector(p) for p in path]
        rings_xyz, rings_uv, rings_w = [], [], []
        rad = 0.025
        sides = 6
        for idx, p in enumerate(pts):
            if idx == 0:
                tang = (pts[1] - pts[0]).normalized()
            elif idx == len(pts) - 1:
                tang = (pts[-1] - pts[-2]).normalized()
            else:
                tang = (pts[idx + 1] - pts[idx - 1]).normalized()
            up_ref = Vector((0.0, 1.0, 0.0))
            n1 = tang.cross(up_ref).normalized()
            n2 = n1.cross(tang).normalized()
            ring_p, ring_uv = [], []
            for k in range(sides):
                ang = (2.0 * math.pi * k) / sides + math.pi / 6.0
                off = n1 * (math.cos(ang) * rad) + n2 * (math.sin(ang) * rad * 0.86)
                vp = p + off
                ring_p.append(tuple(vp))
                vt = 0.14 + 0.32 * (0.5 + 0.5 * math.sin(ang))
                ring_uv.append((0.90 + 0.07 * (idx / (len(pts) - 1)), vt))
            rings_xyz.append(ring_p)
            rings_uv.append(ring_uv)
            rings_w.append([w_clamp] * sides)
        b.add_loft(rings_xyz, rings_uv, rings_w, cap_start=True, cap_end=True, closed=True, smooth=False)


def build_head_assembly(b: MeshPartBuilder):
    """
    Head nestled forward against upper chest:
    - Rounded warm-orange cranium & exposed brow/temples above mask (no intersection with eye sockets!)
    - Rounded-hexagonal chunky side ears
    - Swept green spiky hair tuft on crown/sides (peeking cleanly above back hump)
    - Crisp faceted white hockey mask with pitch-black recessed eye cavities, nose ridge & red markings
    """
    w_head = {"Head": 1.0}
    hc = Vector((0.0, -0.34, 1.13))

    # 1. Smooth rounded orange cranium (front y reaches -0.51, safely behind mask eye backings at -0.545!)
    lat_rings = [
        (-0.18, 0.12, 0.11, 0.00),
        (-0.11, 0.19, 0.15, 0.00),
        (+0.00, 0.225, 0.165, 0.00),
        (+0.11, 0.220, 0.180, -0.025),  # Exposed warm-orange forehead rounding forward above mask!
        (+0.18, 0.175, 0.155, -0.020),
        (+0.22, 0.110, 0.105, -0.010),
    ]
    m_head = 18
    r_xyz, r_uv, r_w = [], [], []
    for dz, rx, ry, dy_f in lat_rings:
        vt = 0.22 + 0.25 * ((dz + 0.18) / 0.40)
        pts, uvs = [], []
        for i in range(m_head):
            ang = (2.0 * math.pi * i) / m_head
            pts.append((hc.x + math.cos(ang) * rx, hc.y + dy_f + math.sin(ang) * ry, hc.z + dz))
            uvs.append((0.08 + 0.18 * (0.5 + 0.5 * math.cos(ang)), vt))
        r_xyz.append(pts)
        r_uv.append(uvs)
        r_w.append([w_head] * m_head)
    b.add_loft(r_xyz, r_uv, r_w, cap_start=True, cap_end=True, closed=True, smooth=True)

    # 2. Chunky rounded-polygonal side ears (+X and -X)
    ear_profile = [
        (-0.025, -0.052),
        (+0.030, -0.045),
        (+0.042, +0.005),
        (+0.032, +0.055),
        (-0.020, +0.062),
        (-0.040, +0.010),
    ]
    for side in (-1.0, +1.0):
        ear_rings = [
            [(side * 0.20, hc.y - 0.02 + dy, hc.z + 0.02 + dz) for dy, dz in ear_profile],
            [(side * 0.285, hc.y - 0.03 + dy * 0.92, hc.z + 0.03 + dz * 0.92) for dy, dz in ear_profile],
        ]
        if side < 0:
            ear_rings = [list(reversed(r)) for r in ear_rings]
        ear_uvs = [[(0.12 + 0.08 * (k / 6.0), 0.30) for k in range(6)] for _ in range(2)]
        b.add_loft(ear_rings, ear_uvs, [[w_head] * 6, [w_head] * 6], cap_start=False, cap_end=True, closed=True, smooth=False)

    # 3. Dark teal sideburn/crown hairline + swept green spiky hair tuft (sweeping upward & toward -X right!)
    # Notice in Front reference: orange forehead is visible between mask top (z=1.24) and green hair base (z=1.30),
    # while dark teal sideburns curve down along the temples (x=+-0.21, z=1.22..1.31) behind the forehead!
    cap_rings = [
        (1.27, 0.222, 0.172, -0.34),
        (1.32, 0.195, 0.155, -0.34),
        (1.35, 0.130, 0.115, -0.33),
    ]
    hc_xyz, hc_uv, hc_w = [], [], []
    for r_i, (z_c, rx, ry, yc) in enumerate(cap_rings):
        vt = 0.06 + 0.18 * (r_i / 2.0)
        pts, uvs = [], []
        for i in range(16):
            ang = (2.0 * math.pi * i) / 16
            # Pull front hairline BACK & UP so warm orange forehead shows clearly above the mask!
            frontness = max(0.0, -math.sin(ang))
            y_push_back = +0.055 * (frontness ** 1.6) if r_i == 0 else +0.025 * (frontness ** 1.6)
            z_front_arch = +0.045 * (frontness ** 1.5) if r_i == 0 else +0.015 * (frontness ** 1.5)
            # Sideburn tabs dipping down above ears (|cos(ang)| near 1)
            sideburn_dip = -0.055 * (abs(math.cos(ang)) ** 4.0) if r_i == 0 else 0.0
            pts.append((math.cos(ang) * rx, yc + math.sin(ang) * ry + y_push_back, z_c + z_front_arch + sideburn_dip))
            uvs.append((0.75 + 0.08 * (i / 16.0), vt))
        hc_xyz.append(pts)
        hc_uv.append(uvs)
        hc_w.append([w_head] * 16)
    b.add_loft(hc_xyz, hc_uv, hc_w, cap_start=False, cap_end=True, closed=True, smooth=False)

    # Swept green hair spikes matching reference tuft (sweeping forward-up & toward -X viewer-left in Front, viewer-right in Back!)
    spikes = [
        # Main broad swept front spike pointing up-left (-X)
        (Vector((-0.01, -0.36, 1.32)), Vector((-0.13, -0.44, 1.535)), 0.095, 0.075),
        # Center-tall spike
        (Vector((+0.05, -0.34, 1.33)), Vector((-0.02, -0.39, 1.520)), 0.085, 0.070),
        # Left-side (+X) smaller swept spike
        (Vector((+0.11, -0.33, 1.31)), Vector((+0.09, -0.36, 1.465)), 0.068, 0.060),
        # Back-crown tuft peeking right over the hunched vest wall (z=1.44 -> 1.525)!
        (Vector((-0.02, -0.25, 1.32)), Vector((-0.08, -0.22, 1.525)), 0.090, 0.075),
    ]
    for base_c, tip_c, s_rx, s_ry in spikes:
        mid_c = base_c.lerp(tip_c, 0.54) + Vector((-0.015, -0.01, 0.018))
        sp_rings, sp_uvs, sp_ws = [], [], []
        for c_pt, scale, vt in [(base_c, 1.0, 0.16), (mid_c, 0.60, 0.32), (base_c.lerp(tip_c, 0.95), 0.10, 0.47)]:
            pts, uvs = [], []
            for k in range(5):
                ang = (2.0 * math.pi * k) / 5 - math.pi / 2
                pts.append((c_pt.x + math.cos(ang) * s_rx * scale, c_pt.y + math.sin(ang) * s_ry * scale, c_pt.z))
                uvs.append((0.74 + 0.10 * (k / 5.0), vt))
            sp_rings.append(pts)
            sp_uvs.append(uvs)
            sp_ws.append([w_head] * 5)
        b.add_loft(sp_rings, sp_uvs, sp_ws, cap_start=False, cap_end=True, closed=True, smooth=False)

    # 4. Crisp Faceted White Hockey Mask (positioned cleanly forward of cranium so eye holes are 100% pitch black!)
    def mask_surf(nx: float, nz: float, depth_offset: float = 0.0) -> Vector:
        x = nx * 0.212
        z = 1.075 + nz * 0.172  # Spans z = 0.903 .. 1.247 (exposing warm orange forehead above 1.24!)
        y = -0.595 + 0.115 * (nx * nx) + 0.032 * (nz * nz) - 0.010 * max(0.0, -nz) + depth_offset
        return Vector((x, y, z))

    def mask_uv(nx: float, nz: float) -> tuple:
        return (0.625 + 0.10 * nx, 0.825 + 0.15 * nz)

    uv_black = [(0.78, 0.84), (0.88, 0.84), (0.88, 0.96), (0.78, 0.96)]
    uv_black5 = [(0.78, 0.84), (0.88, 0.84), (0.88, 0.92), (0.83, 0.97), (0.78, 0.92)]

    outer_n = [
        (0.00, -0.96),
        (+0.54, -0.90),
        (+0.90, -0.56),
        (+0.99, -0.04),
        (+0.94, +0.50),
        (+0.60, +0.90),
        (0.00, +0.97),
        (-0.60, +0.90),
        (-0.94, +0.50),
        (-0.99, -0.04),
        (-0.90, -0.56),
        (-0.54, -0.90),
    ]
    pts_dict = {}

    def mv(key, nx, nz, off=0.0):
        if key not in pts_dict:
            p = mask_surf(nx, nz, off)
            pts_dict[key] = (b.add_vert(tuple(p), w_head), mask_uv(nx, nz))
        return pts_dict[key]

    for idx, (nx, nz) in enumerate(outer_n):
        mv(f"O{idx}", nx, nz, 0.0)
        mv(f"B{idx}", nx * 0.95, nz * 0.97, +0.052)

    for idx in range(len(outer_n)):
        nxt = (idx + 1) % len(outer_n)
        v0, uv0 = pts_dict[f"O{idx}"]
        v1, uv1 = pts_dict[f"O{nxt}"]
        vb1, uvb1 = pts_dict[f"B{nxt}"]
        vb0, uvb0 = pts_dict[f"B{idx}"]
        b.add_face([v0, vb0, vb1, v1], [uv0, uvb0, uvb1, uv1], smooth=False)

    mv("C_brow", 0.0, +0.50, -0.004)
    mv("C_bridge", 0.0, +0.20, -0.008)
    mv("C_nose_tip", 0.0, -0.05, -0.025)
    mv("C_nose_L", +0.13, -0.16, -0.004)
    mv("C_nose_R", -0.13, -0.16, -0.004)
    mv("C_mouth", 0.0, -0.50, -0.003)

    # Large, bold fierce black eye sockets matching reference!
    el_coords = [
        ("EL0", +0.15, -0.02),
        ("EL1", +0.74, -0.04),
        ("EL2", +0.80, +0.34),
        ("EL3", +0.44, +0.45),
        ("EL4", +0.13, +0.25),
    ]
    er_coords = [
        ("ER0", -0.15, -0.02),
        ("ER1", -0.74, -0.04),
        ("ER2", -0.80, +0.34),
        ("ER3", -0.44, +0.45),
        ("ER4", -0.13, +0.25),
    ]
    for k, nx, nz in el_coords + er_coords:
        mv(k, nx, nz, 0.0)
        mv(k + "_in", nx * 0.94, nz * 0.97, +0.020)

    def face_keys(keys, custom_uv=None):
        vs = [pts_dict[k][0] for k in keys]
        uvs = custom_uv if custom_uv is not None else [pts_dict[k][1] for k in keys]
        b.add_face(vs, uvs, smooth=False)

    # Recessed black bevel walls + SEALED pitch-black back cap on each eye socket!
    for idx in range(5):
        nxt = (idx + 1) % 5
        face_keys([f"EL{idx}", f"EL{idx}_in", f"EL{nxt}_in", f"EL{nxt}"], uv_black)
        face_keys([f"ER{idx}", f"ER{nxt}", f"ER{nxt}_in", f"ER{idx}_in"], uv_black)
    # Sealed black eye socket caps (CCW looking from -Y -> +Y)
    face_keys(["EL0_in", "EL1_in", "EL2_in", "EL3_in", "EL4_in"], uv_black5)
    face_keys(["ER4_in", "ER3_in", "ER2_in", "ER1_in", "ER0_in"], uv_black5)

    # Mask front topology
    face_keys(["C_brow", "EL3", "O5", "O6"])
    face_keys(["O6", "O7", "ER3", "C_brow"])
    face_keys(["EL3", "EL2", "O4", "O5"])
    face_keys(["ER2", "ER3", "O7", "O8"])
    face_keys(["C_bridge", "EL4", "EL3", "C_brow"])
    face_keys(["C_bridge", "C_brow", "ER3", "ER4"])
    face_keys(["EL1", "O3", "O4", "EL2"])
    face_keys(["O9", "ER1", "ER2", "O8"])

    uv_nose_shadow = [(0.92, 0.84), (0.98, 0.84), (0.95, 0.96)]
    face_keys(["C_nose_tip", "EL0", "EL4", "C_bridge"])
    face_keys(["C_nose_tip", "C_bridge", "ER4", "ER0"])
    face_keys(["C_nose_R", "C_nose_L", "C_nose_tip"], uv_nose_shadow)
    face_keys(["C_nose_L", "EL0", "C_nose_tip"])
    face_keys(["ER0", "C_nose_R", "C_nose_tip"])

    face_keys(["C_nose_L", "O2", "O3", "EL1", "EL0"])
    face_keys(["ER0", "ER1", "O9", "O10", "C_nose_R"])
    face_keys(["C_mouth", "O1", "O2", "C_nose_L"])
    face_keys(["C_mouth", "C_nose_L", "C_nose_R"])
    face_keys(["C_mouth", "C_nose_R", "O10", "O11"])
    face_keys(["O0", "O1", "C_mouth"])
    face_keys(["O11", "O0", "C_mouth"])

    # 5. Crisp 3D Red War-Paint Markings (flush-offset by 2mm, no Freestyle crease clutter)
    uv_red4 = [(0.78, 0.68), (0.88, 0.68), (0.88, 0.78), (0.78, 0.78)]
    uv_red3 = uv_red4[:3]

    chev_pts = [
        (0.00, +0.52),
        (+0.25, +0.78),
        (+0.16, +0.84),
        (0.00, +0.66),
        (-0.16, +0.84),
        (-0.25, +0.78),
    ]
    cv = [b.add_vert(tuple(mask_surf(nx, nz, -0.0025)), w_head) for nx, nz in chev_pts]
    b.add_face([cv[0], cv[1], cv[2], cv[3]], uv_red4, smooth=False)
    b.add_face([cv[0], cv[3], cv[4], cv[5]], uv_red4, smooth=False)

    for side in (+1.0, -1.0):
        tri_n = [
            (side * 0.28, -0.28),
            (side * 0.72, -0.60),
            (side * 0.50, -0.78),
        ]
        if side < 0:
            tri_n = [tri_n[0], tri_n[2], tri_n[1]]
        tv = [b.add_vert(tuple(mask_surf(nx, nz, -0.0025)), w_head) for nx, nz in tri_n]
        b.add_face(tv, uv_red3, smooth=False)


def build_shorts_and_feet(b: MeshPartBuilder):
    """Wide squat ragged slate-teal shorts + wide flat-bottomed burnt-orange boots."""
    pelvis_rings = [
        (0.26, 0.50, -0.04, 0.34),
        (0.38, 0.55, -0.05, 0.40),
        (0.50, 0.52, -0.05, 0.38),
    ]
    m_p = 20
    pr_xyz, pr_uv, pr_w = [], [], []
    for r_i, (z_p, rx, yc, ry) in enumerate(pelvis_rings):
        vt = 0.24 + 0.22 * (r_i / 2.0)
        pts, uvs, ws = [], [], []
        for i in range(m_p):
            ang = (2.0 * math.pi * i) / m_p
            x = math.cos(ang) * rx
            y = yc + math.sin(ang) * ry
            z_adj = z_p - (0.04 * (abs(math.sin(ang)) ** 2.0) if r_i == 0 else 0.0)
            pts.append((x, y, z_adj))
            uvs.append((0.53 + 0.16 * (i / m_p), vt))
            if r_i == 0 and abs(x) > 0.15:
                leg_key = "Thigh.L" if x > 0 else "Thigh.R"
                ws.append({"Pelvis": 0.65, leg_key: 0.35})
            else:
                ws.append({"Pelvis": 1.0})
        pr_xyz.append(pts)
        pr_uv.append(uvs)
        pr_w.append(ws)
    b.add_loft(pr_xyz, pr_uv, pr_w, cap_start=True, cap_end=True, closed=True, smooth=True)

    for side, bone_th, bone_ft in [(+1.0, "Thigh.L", "Foot.L"), (-1.0, "Thigh.R", "Foot.R")]:
        cx, cy = side * 0.34, -0.04
        yaw = side * math.radians(16.0)
        cy_cos, cy_sin = math.cos(yaw), math.sin(yaw)

        def leg_pt(lx, ly, z):
            return (cx + lx * cy_cos - ly * cy_sin, cy + lx * cy_sin + ly * cy_cos, z)

        m_leg = 14
        leg_rings_xyz, leg_rings_uv, leg_rings_w = [], [], []
        for z_base, rad_x, rad_y, jagged_amp, vt, w_th in [
            (0.35, 0.24, 0.23, 0.00, 0.34, 0.45),
            (0.25, 0.255, 0.24, 0.00, 0.20, 0.85),
            (0.165, 0.265, 0.245, 0.045, 0.06, 0.95),
        ]:
            pts, uvs, ws = [], [], []
            for i in range(m_leg):
                ang = (2.0 * math.pi * i) / m_leg
                tooth = -jagged_amp if (i % 2 == 0) else +jagged_amp
                r_scale = 1.03 if (jagged_amp > 0 and i % 2 == 0) else 0.97
                pts.append(leg_pt(math.cos(ang) * rad_x * r_scale, math.sin(ang) * rad_y * r_scale, z_base + tooth))
                uvs.append((0.53 + 0.16 * (i / m_leg), vt))
                ws.append({"Pelvis": 1.0 - w_th, bone_th: w_th})
            leg_rings_xyz.append(pts)
            leg_rings_uv.append(uvs)
            leg_rings_w.append(ws)
        inner_pts = [leg_pt(math.cos((2.0 * math.pi * i) / m_leg) * 0.18, math.sin((2.0 * math.pi * i) / m_leg) * 0.18, 0.20) for i in range(m_leg)]
        leg_rings_xyz.append(inner_pts)
        leg_rings_uv.append([(0.55, 0.03)] * m_leg)
        leg_rings_w.append([{bone_th: 1.0}] * m_leg)
        b.add_loft(leg_rings_xyz, leg_rings_uv, leg_rings_w, cap_start=True, cap_end=True, closed=True, smooth=False)

        # Wide flat-bottomed boot
        fcx, fcy = side * 0.365, -0.06
        fyaw = side * math.radians(20.0)
        fc_cos, fc_sin = math.cos(fyaw), math.sin(fyaw)

        def foot_pt(lx, ly, z):
            return (fcx + lx * fc_cos - ly * fc_sin, fcy + lx * fc_sin + ly * fc_cos, z)

        boot_base_poly = [
            (-0.125, -0.195),
            (+0.125, -0.195),
            (+0.155, -0.06),
            (+0.140, +0.130),
            (+0.085, +0.160),
            (-0.085, +0.160),
            (-0.140, +0.130),
            (-0.155, -0.06),
        ]
        f_rings_xyz, f_rings_uv, f_rings_w = [], [], []
        for z_f, sx, sy_f, sy_b, dy_shift, vt, w_foot in [
            (0.00, 1.00, 1.04, 1.00, -0.020, 0.28, 1.00),
            (0.08, 0.98, 0.96, 0.98, -0.012, 0.36, 0.95),
            (0.16, 0.84, 0.78, 0.88, +0.015, 0.44, 0.75),
            (0.24, 0.76, 0.72, 0.82, +0.020, 0.48, 0.45),
        ]:
            pts, uvs, ws = [], [], []
            for idx_p, (bx, by) in enumerate(boot_base_poly):
                sy = sy_f if by < 0 else sy_b
                pts.append(foot_pt(bx * sx, by * sy + dy_shift, z_f))
                uvs.append((0.37 + 0.11 * (idx_p / len(boot_base_poly)), vt))
                ws.append({bone_ft: w_foot, bone_th: 1.0 - w_foot})
            f_rings_xyz.append(pts)
            f_rings_uv.append(uvs)
            f_rings_w.append(ws)
        b.add_loft(f_rings_xyz, f_rings_uv, f_rings_w, cap_start=True, cap_end=True, closed=True, smooth=False)


def build_arm_and_fist(b: MeshPartBuilder, side: float):
    """Massive curved gorilla-brawler arms + broad heavy fists with distinct inner thumb & 3 chunky curled fingers."""
    suf = "L" if side > 0 else "R"
    b_clav = f"Clavicle.{suf}"
    b_up = f"UpperArm.{suf}"
    b_fore = f"Forearm.{suf}"
    b_hand = f"Hand.{suf}"
    b_thumb = f"Thumb.{suf}"
    b_fing = f"Fingers.{suf}"

    stations = [
        (Vector((side * 0.33, +0.03, 1.24)), 0.145, 0.145, 0.49, {b_clav: 0.75, b_up: 0.25}),
        (Vector((side * 0.48, -0.01, 1.22)), 0.198, 0.192, 0.46, {b_clav: 0.40, b_up: 0.60}),
        (Vector((side * 0.62, -0.04, 1.14)), 0.222, 0.212, 0.41, {b_clav: 0.12, b_up: 0.88}),
        (Vector((side * 0.75, -0.07, 0.98)), 0.215, 0.205, 0.34, {b_up: 0.90, b_fore: 0.10}),
        (Vector((side * 0.84, -0.12, 0.80)), 0.200, 0.192, 0.27, {b_up: 0.48, b_fore: 0.52}),  # Elbow
        (Vector((side * 0.90, -0.19, 0.61)), 0.232, 0.215, 0.19, {b_fore: 0.92, b_up: 0.08}),  # Huge forearm!
        (Vector((side * 0.88, -0.24, 0.44)), 0.215, 0.185, 0.12, {b_fore: 0.58, b_hand: 0.42}),  # Wrist
        (Vector((side * 0.85, -0.26, 0.34)), 0.205, 0.155, 0.07, {b_hand: 0.95, b_fore: 0.05}),  # Broad hand block
        (Vector((side * 0.82, -0.25, 0.27)), 0.185, 0.135, 0.04, {b_hand: 1.0}),                  # Knuckles
    ]

    m_arm = 16
    rings_xyz, rings_uv, rings_w = [], [], []
    for idx, (pos, rx, ry, vt, w_dict) in enumerate(stations):
        if idx == 0:
            tang = (stations[1][0] - stations[0][0]).normalized()
        elif idx == len(stations) - 1:
            tang = (stations[-1][0] - stations[-2][0]).normalized()
        else:
            tang = (stations[idx + 1][0] - stations[idx - 1][0]).normalized()

        fwd = Vector((0.0, -1.0, 0.0))
        lat = tang.cross(fwd).normalized()
        ortho_fwd = lat.cross(tang).normalized()

        pts, uvs = [], []
        for k in range(m_arm):
            ang = (2.0 * math.pi * k) / m_arm
            pwr = 2.05 if idx < 6 else 2.55
            ca, sa = math.cos(ang), math.sin(ang)
            ex = math.copysign(abs(ca) ** (2.0 / pwr), ca) * rx
            ey = math.copysign(abs(sa) ** (2.0 / pwr), sa) * ry
            vp = pos + lat * ex + ortho_fwd * ey
            pts.append(tuple(vp))
            uvs.append((0.04 + 0.26 * (k / m_arm), vt))
        rings_xyz.append(pts)
        rings_uv.append(uvs)
        rings_w.append([w_dict] * m_arm)

    b.add_loft(rings_xyz, rings_uv, rings_w, cap_start=True, cap_end=True, closed=True, smooth=True)

    # Big chunky inner thumb creating iconic brawler hand notch in front view!
    thumb_stations = [
        (Vector((side * 0.72, -0.29, 0.37)), 0.082, 0.075, 0.09, {b_hand: 0.50, b_thumb: 0.50}),
        (Vector((side * 0.65, -0.32, 0.27)), 0.074, 0.068, 0.05, {b_thumb: 0.90, b_hand: 0.10}),
        (Vector((side * 0.68, -0.28, 0.19)), 0.058, 0.054, 0.03, {b_thumb: 1.0}),
    ]
    th_xyz, th_uv, th_w = [], [], []
    for pos, rx, ry, vt, wd in thumb_stations:
        pts, uvs = [], []
        for k in range(6):
            ang = (2.0 * math.pi * k) / 6 + math.pi / 6
            pts.append((pos.x + math.cos(ang) * rx, pos.y + math.sin(ang) * ry, pos.z))
            uvs.append((0.08 + 0.14 * (k / 6.0), vt))
        th_xyz.append(pts)
        th_uv.append(uvs)
        th_w.append([wd] * 6)
    b.add_loft(th_xyz, th_uv, th_w, cap_start=True, cap_end=True, closed=True, smooth=True)

    # 3 Full, Chunky Curled Fingers side-by-side (touching cleanly, curling inward into palm!)
    finger_offsets = [
        (-0.095, +0.025, 0.066),  # Index
        (+0.000, +0.005, 0.070),  # Middle
        (+0.095, -0.020, 0.065),  # Ring/Outer
    ]
    for f_idx, (dy_off, dx_in_off, f_rad) in enumerate(finger_offsets):
        knuckle = Vector((side * (0.85 + dx_in_off), -0.25 - dy_off, 0.28))
        mid_ph = Vector((side * (0.87 + dx_in_off), -0.22 - dy_off, 0.185))
        curl_tip = Vector((side * (0.75 + dx_in_off), -0.13 - dy_off * 0.88, 0.215))
        f_pts_list = [
            (knuckle, f_rad, 0.06, {b_hand: 0.55, b_fing: 0.45}),
            (mid_ph, f_rad * 0.95, 0.04, {b_fing: 0.90, b_hand: 0.10}),
            (curl_tip, f_rad * 0.84, 0.02, {b_fing: 1.0}),
        ]
        fg_xyz, fg_uv, fg_w = [], [], []
        for s_i, (p_c, rr, vt, wd) in enumerate(f_pts_list):
            if s_i == 0:
                tg = (f_pts_list[1][0] - f_pts_list[0][0]).normalized()
            elif s_i == 2:
                tg = (f_pts_list[2][0] - f_pts_list[1][0]).normalized()
            else:
                tg = (f_pts_list[2][0] - f_pts_list[0][0]).normalized()
            n_y = Vector((0.0, 1.0, 0.0))
            n_a = tg.cross(n_y).normalized()
            n_b = n_a.cross(tg).normalized()
            pts, uvs = [], []
            for k in range(6):
                ang = (2.0 * math.pi * k) / 6 + math.pi / 4
                vp = p_c + n_a * (math.cos(ang) * rr) + n_b * (math.sin(ang) * rr * 0.85)
                pts.append(tuple(vp))
                uvs.append((0.08 + 0.12 * (k / 6.0), vt))
            fg_xyz.append(pts)
            fg_uv.append(uvs)
            fg_w.append([wd] * 6)
        b.add_loft(fg_xyz, fg_uv, fg_w, cap_start=True, cap_end=True, closed=True, smooth=True)


# ---------------------------------------------------------------------------
# 3. Build Armature & Skinning
# ---------------------------------------------------------------------------
def create_armature() -> bpy.types.Object:
    arm_data = bpy.data.armatures.new("BrawlerRig")
    arm_obj = bpy.data.objects.new("BrawlerRig", arm_data)
    bpy.context.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones

    def bone(name, head, tail, parent=None, connect=False):
        bn = eb.new(name)
        bn.head = Vector(head)
        bn.tail = Vector(tail)
        if parent is not None:
            bn.parent = eb[parent]
            bn.use_connect = connect
        return bn

    bone("Root", (0.0, 0.0, 0.0), (0.0, 0.0, 0.22))
    bone("Pelvis", (0.0, -0.04, 0.38), (0.0, -0.04, 0.62), "Root")
    bone("Spine", (0.0, -0.04, 0.62), (0.0, -0.01, 0.96), "Pelvis", connect=True)
    bone("Chest", (0.0, -0.01, 0.96), (0.0, +0.04, 1.36), "Spine", connect=True)
    bone("Head", (0.0, -0.26, 1.06), (0.0, -0.35, 1.34), "Chest")

    for side, suf in [(-1.0, "R"), (+1.0, "L")]:
        bone(f"Clavicle.{suf}", (side * 0.18, +0.02, 1.24), (side * 0.48, -0.02, 1.22), "Chest")
        bone(f"UpperArm.{suf}", (side * 0.48, -0.02, 1.22), (side * 0.84, -0.12, 0.80), f"Clavicle.{suf}", connect=True)
        bone(f"Forearm.{suf}", (side * 0.84, -0.12, 0.80), (side * 0.88, -0.24, 0.44), f"UpperArm.{suf}", connect=True)
        bone(f"Hand.{suf}", (side * 0.88, -0.24, 0.44), (side * 0.82, -0.25, 0.26), f"Forearm.{suf}", connect=True)
        bone(f"Thumb.{suf}", (side * 0.72, -0.29, 0.37), (side * 0.67, -0.28, 0.19), f"Hand.{suf}")
        bone(f"Fingers.{suf}", (side * 0.85, -0.25, 0.28), (side * 0.76, -0.14, 0.21), f"Hand.{suf}")

        bone(f"Thigh.{suf}", (side * 0.34, -0.04, 0.36), (side * 0.36, -0.05, 0.14), "Pelvis")
        bone(f"Foot.{suf}", (side * 0.36, -0.05, 0.14), (side * 0.38, -0.22, 0.04), f"Thigh.{suf}", connect=True)

    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    return arm_obj


def create_character_mesh(arm_obj: bpy.types.Object, atlas_img: bpy.types.Image) -> bpy.types.Object:
    builder = MeshPartBuilder()
    build_torso(builder)
    build_shoulder_staples(builder)
    build_head_assembly(builder)
    build_shorts_and_feet(builder)
    build_arm_and_fist(builder, -1.0)
    build_arm_and_fist(builder, +1.0)

    bmesh.ops.recalc_face_normals(builder.bm, faces=builder.bm.faces)

    mesh = bpy.data.meshes.new("MaskedBrawlerMesh")
    builder.bm.to_mesh(mesh)
    builder.bm.free()

    obj = bpy.data.objects.new("MaskedBrawler", mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = arm_obj

    bone_names = [b.name for b in arm_obj.data.bones]
    vgroups = {bn: obj.vertex_groups.new(name=bn) for bn in bone_names}
    for v_idx, w_map in enumerate(builder.vert_weights):
        for bn, w_val in w_map.items():
            if bn in vgroups and w_val > 1e-5:
                vgroups[bn].add([v_idx], float(w_val), "REPLACE")

    mod = obj.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = arm_obj

    mat = bpy.data.materials.new(name="Mat_MaskedBrawler")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.85
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.14
    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.image = atlas_img
    tex_node.interpolation = "Linear"
    nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
    mesh.materials.append(mat)
    return obj


# ---------------------------------------------------------------------------
# 4. Exact Hierarchical World-Target Pose Solver & Combat Animations
# ---------------------------------------------------------------------------
def apply_pose_in_hierarchy(arm_obj: bpy.types.Object, pose_spec: dict):
    """
    Apply rotations sequentially from root to leaves using TRUE evaluated parent pose matrices!
    Supports:
      - `("AIM", (dx, dy, dz), roll_deg)` -> aims `(pb.tail - pb.head)` along exact world vector `(dx, dy, dz)`
      - `(rx_deg, ry_deg, rz_deg)` -> rotates bone in TRUE world axes relative to current parent pose
      - `((rx_deg, ry_deg, rz_deg), (lx, ly, lz))` -> world rotation + world translation delta
    """
    for pb in arm_obj.pose.bones:
        pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        pb.location = Vector((0.0, 0.0, 0.0))
    bpy.context.view_layer.update()

    for pb in arm_obj.pose.bones:
        if pb.name not in pose_spec:
            continue
        val = pose_spec[pb.name]

        if isinstance(val, tuple) and len(val) == 3 and val[0] == "AIM":
            _, dir_tuple, roll_deg = val
            m0 = pb.matrix.to_3x3()
            cur_dir = (pb.tail - pb.head).normalized()
            tgt_dir = Vector(dir_tuple).normalized()
            q_aim = cur_dir.rotation_difference(tgt_dir)
            q_roll = Quaternion(tgt_dir, math.radians(roll_deg))
            w_rot = (q_roll @ q_aim).to_matrix()
            pb.rotation_quaternion = (m0.inverted() @ w_rot @ m0).to_quaternion()
            bpy.context.view_layer.update()
            continue

        if len(val) == 2 and isinstance(val[1], tuple):
            euler_deg, loc_delta = val
        else:
            euler_deg, loc_delta = val, None

        m0 = pb.matrix.to_3x3()
        rx, ry, rz = [math.radians(a) for a in euler_deg]
        w_rot = Euler((rx, ry, rz), "XYZ").to_matrix()
        pb.rotation_quaternion = (m0.inverted() @ w_rot @ m0).to_quaternion()

        if loc_delta is not None:
            rest_m = pb.bone.matrix_local.to_3x3()
            pb.location = rest_m.inverted() @ Vector(loc_delta)
        bpy.context.view_layer.update()


def key_all_bones(arm_obj: bpy.types.Object, frame: int, pose_spec: dict):
    apply_pose_in_hierarchy(arm_obj, pose_spec)
    for pb in arm_obj.pose.bones:
        pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        if pb.name == "Pelvis":
            pb.keyframe_insert(data_path="location", frame=frame)


def build_animations(arm_obj: bpy.types.Object):
    bpy.context.scene.render.fps = 30
    arm_obj.animation_data_create()

    READY = {
        "Pelvis": ((0, 0, 0), (0.0, 0.0, 0.0)),
        "Spine": (2, 0, 0),
        "Chest": (2, 0, 0),
        "Head": (-3, 0, 0),
        "UpperArm.R": (-4, +2, 0),
        "Forearm.R": (-10, -2, +4),
        "UpperArm.L": (-4, -2, 0),
        "Forearm.L": (-10, +2, -4),
    }

    # --- 1. AN_Brute_Attack_Heavy_01 (42 frames @ 30fps = 1.40s) ---
    act_punch = bpy.data.actions.new(name="AN_Brute_Attack_Heavy_01")
    arm_obj.animation_data.action = act_punch

    PUNCH_ANTICIPATION = {
        # Coil torso right (-26 deg total) & sink slightly while mask glares straight ahead!
        "Pelvis": ((-4, -2, -8), (0.02, +0.04, -0.045)),
        "Spine": (-4, +2, -9),
        "Chest": (-5, +3, -9),
        "Head": (+11, -3, +24),
        "Clavicle.R": (+4, +8, -10),
        # Cock Right arm WIDE OUT RIGHT (-X), BACK (+Y), & HIGH (+Z) with forearm loaded like a sledgehammer!
        "UpperArm.R": ("AIM", (-0.82, +0.50, +0.24), -15),
        "Forearm.R": ("AIM", (-0.18, -0.45, +0.87), +12),
        "Hand.R": (+8, +6, 0),
        "Fingers.R": (+32, 0, 0),
        "Thumb.R": (+20, -14, 0),
        # Left stapled arm guards low-front
        "Clavicle.L": (0, -4, +6),
        "UpperArm.L": ("AIM", (+0.48, -0.62, -0.62), 0),
        "Forearm.L": ("AIM", (-0.22, -0.82, +0.32), 0),
        "Thigh.R": (+6, 0, +8),
        "Foot.R": (-6, 0, -8),
        "Thigh.L": (-6, 0, +8),
        "Foot.L": (+6, 0, -8),
    }

    PUNCH_IMPACT = {
        # Explosive torso whip left (+32 deg total) + lunge forward into -Y!
        "Pelvis": ((+6, +2, +10), (-0.02, -0.12, -0.035)),
        "Spine": (+6, -2, +10),
        "Chest": (+8, -3, +12),
        "Head": (-16, +4, -30),  # White hockey mask glares straight ahead OVER the extended arm!
        "Clavicle.R": (-8, -6, +14),
        # FULL STRAIGHT FORWARD SMASH EXTENSION (-Y = -0.97) slightly below eye level so mask stays 100% readable!
        "UpperArm.R": ("AIM", (+0.16, -0.96, -0.21), +28),
        "Forearm.R": ("AIM", (+0.12, -0.99, -0.06), +38),
        "Hand.R": ("AIM", (+0.08, -0.99, -0.08), +38),
        "Fingers.R": (+38, 0, 0),
        "Thumb.R": (+24, -16, 0),
        # Left arm counter-balances behind left hip
        "Clavicle.L": (+6, +6, -10),
        "UpperArm.L": ("AIM", (+0.65, +0.52, -0.55), 0),
        "Forearm.L": ("AIM", (+0.35, -0.25, -0.90), 0),
        "Thigh.R": (-8, 0, -10),
        "Foot.R": (+8, 0, +10),
        "Thigh.L": (+7, 0, -10),
        "Foot.L": (-7, 0, +10),
    }

    punch_keys = [
        (0, READY),
        (
            4,
            {
                "Pelvis": ((-2, -1, -4), (0.01, +0.02, -0.02)),
                "Spine": (-2, +1, -4),
                "Chest": (-3, +1, -5),
                "Head": (+6, -1, +12),
                "UpperArm.R": ("AIM", (-0.80, +0.18, -0.56), -8),
                "Forearm.R": ("AIM", (-0.30, -0.68, +0.42), +6),
                "Fingers.R": (+18, 0, 0),
                "UpperArm.L": ("AIM", (+0.52, -0.45, -0.72), 0),
                "Forearm.L": ("AIM", (-0.05, -0.72, -0.42), 0),
            },
        ),
        (11, PUNCH_ANTICIPATION),
        # f13: Peak coil hang before 3-frame strike snap!
        (
            13,
            {
                **PUNCH_ANTICIPATION,
                "UpperArm.R": ("AIM", (-0.84, +0.54, +0.28), -18),
                "Forearm.R": ("AIM", (-0.14, -0.40, +0.90), +14),
            },
        ),
        # f15: Wide sweeping haymaker arc out to the right-front!
        (
            15,
            {
                "Pelvis": ((+3, +1, +5), (-0.01, -0.06, -0.025)),
                "Spine": (+4, -1, +5),
                "Chest": (+5, -2, +6),
                "Head": (-9, +2, -15),
                "Clavicle.R": (-4, +4, +8),
                "UpperArm.R": ("AIM", (-0.58, -0.80, -0.14), +15),
                "Forearm.R": ("AIM", (+0.10, -0.98, -0.08), +26),
                "Fingers.R": (+36, 0, 0),
                "Thumb.R": (+22, -15, 0),
                "UpperArm.L": ("AIM", (+0.62, +0.25, -0.74), 0),
                "Forearm.L": ("AIM", (+0.25, -0.42, -0.87), 0),
                "Thigh.R": (-5, 0, -5),
                "Foot.R": (+5, 0, +5),
                "Thigh.L": (+4, 0, -5),
                "Foot.L": (-4, 0, +5),
            },
        ),
        # f17: FULL STRAIGHT FORWARD PUNCH IMPACT!
        (17, PUNCH_IMPACT),
        # f20: Hit-stop & slight heavy downward crush overshoot!
        (
            20,
            {
                **PUNCH_IMPACT,
                "Pelvis": ((+7, +2, +11), (-0.025, -0.13, -0.042)),
                "Chest": (+9, -4, +13),
                "Head": (-18, +4, -32),
                "UpperArm.R": ("AIM", (+0.22, -0.94, -0.26), +34),
                "Forearm.R": ("AIM", (+0.16, -0.97, -0.16), +44),
                "Hand.R": ("AIM", (+0.12, -0.97, -0.20), +44),
            },
        ),
        # f27: Heavy follow-through gravity drop
        (
            27,
            {
                "Pelvis": ((+4, +1, +5), (-0.01, -0.06, -0.025)),
                "Spine": (+4, -1, +5),
                "Chest": (+5, -2, +6),
                "Head": (-8, +2, -15),
                "Clavicle.R": (-3, -3, +5),
                "UpperArm.R": ("AIM", (-0.28, -0.76, -0.58), +14),
                "Forearm.R": ("AIM", (+0.06, -0.80, -0.60), +16),
                "Fingers.R": (+20, 0, 0),
                "Thumb.R": (+12, -8, 0),
                "UpperArm.L": ("AIM", (+0.60, +0.18, -0.78), 0),
                "Forearm.L": ("AIM", (+0.18, -0.42, -0.89), 0),
                "Thigh.R": (-4, 0, -5),
                "Foot.R": (+4, 0, +5),
                "Thigh.L": (+4, 0, -5),
                "Foot.L": (-4, 0, +5),
            },
        ),
        # f35: Settle bounce
        (
            35,
            {
                "Pelvis": ((+1, 0, +1), (0.0, -0.015, -0.012)),
                "Spine": (+3, 0, +1),
                "Chest": (+3, 0, +2),
                "Head": (-4, 0, -3),
                "UpperArm.R": (-10, +2, +2),
                "Forearm.R": (-16, -3, +5),
                "UpperArm.L": (-2, -3, -2),
                "Forearm.L": (-12, +3, -5),
            },
        ),
        (42, READY),
    ]
    for f, spec in punch_keys:
        key_all_bones(arm_obj, f, spec)

    # Self-check assertion on impact frame 17: fist must extend far forward (-Y < -1.05m)!
    bpy.context.scene.frame_set(17)
    bpy.context.view_layer.update()
    fist_tip = arm_obj.pose.bones["Hand.R"].tail
    print(f"IMPACT_FIST_POS=({fist_tip.x:.3f}, {fist_tip.y:.3f}, {fist_tip.z:.3f})")
    assert fist_tip.y < -1.05, f"Expected full forward punch extension (y < -1.05), got {fist_tip}"

    track1 = arm_obj.animation_data.nla_tracks.new()
    track1.name = "AN_Brute_Attack_Heavy_01"
    track1.strips.new("AN_Brute_Attack_Heavy_01", 0, act_punch)
    arm_obj.animation_data.action = None

    # --- 2. AN_Brute_Attack_Combo_02 (54 frames @ 30fps = Left Hook + Right Haymaker!) ---
    act_combo = bpy.data.actions.new(name="AN_Brute_Attack_Combo_02")
    arm_obj.animation_data.action = act_combo
    combo_keys = [
        (0, READY),
        (
            7,
            {
                "Pelvis": ((-3, +2, +8), (-0.02, +0.03, -0.035)),
                "Spine": (-3, -2, +8),
                "Chest": (-4, -3, +9),
                "Head": (+9, +3, -22),
                "Clavicle.L": (+4, -8, +10),
                "UpperArm.L": ("AIM", (+0.82, +0.46, +0.22), +15),
                "Forearm.L": ("AIM", (+0.18, -0.45, +0.87), -12),
                "Fingers.L": (+32, 0, 0),
                "Thumb.L": (+20, +14, 0),
                "UpperArm.R": ("AIM", (-0.48, -0.62, -0.62), 0),
                "Forearm.R": ("AIM", (+0.22, -0.82, +0.32), 0),
            },
        ),
        (
            11,
            {
                "Pelvis": ((+6, -2, -9), (+0.02, -0.09, -0.03)),
                "Spine": (+6, +2, -10),
                "Chest": (+7, +3, -11),
                "Head": (-14, -3, +28),
                "Clavicle.L": (-8, +6, -12),
                "UpperArm.L": ("AIM", (-0.16, -0.96, -0.21), -28),
                "Forearm.L": ("AIM", (-0.12, -0.99, -0.06), -38),
                "Hand.L": ("AIM", (-0.08, -0.99, -0.08), -38),
                "Fingers.L": (+38, 0, 0),
                "Thumb.L": (+24, +16, 0),
                "Clavicle.R": (+4, +8, -10),
                "UpperArm.R": ("AIM", (-0.80, +0.46, +0.22), -15),
                "Forearm.R": ("AIM", (-0.18, -0.45, +0.87), +12),
                "Fingers.R": (+30, 0, 0),
            },
        ),
        (19, PUNCH_ANTICIPATION),
        (22, punch_keys[4][1]),
        (24, PUNCH_IMPACT),
        (28, punch_keys[6][1]),
        (36, punch_keys[7][1]),
        (45, punch_keys[8][1]),
        (54, READY),
    ]
    for f, spec in combo_keys:
        key_all_bones(arm_obj, f, spec)

    track2 = arm_obj.animation_data.nla_tracks.new()
    track2.name = "AN_Brute_Attack_Combo_02"
    track2.strips.new("AN_Brute_Attack_Combo_02", 0, act_combo)
    arm_obj.animation_data.action = None

    # --- 3. AN_Brute_Idle (60 frames @ 30fps seamless heavy breathing loop) ---
    act_idle = bpy.data.actions.new(name="AN_Brute_Idle")
    arm_obj.animation_data.action = act_idle
    idle_keys = [
        (0, READY),
        (
            15,
            {
                "Pelvis": ((0, 0, -2), (0.0, 0.0, +0.014)),
                "Spine": (0, 0, -2),
                "Chest": (-1, 0, -2),
                "Head": (-1, +2, +5),
                "UpperArm.R": (-2, +6, 0),
                "Forearm.R": (-18, -4, +6),
                "UpperArm.L": (-10, -6, 0),
                "Forearm.L": (-12, +4, -6),
            },
        ),
        (
            30,
            {
                "Pelvis": ((+2, 0, 0), (0.0, 0.0, -0.018)),
                "Spine": (+4, 0, 0),
                "Chest": (+6, 0, 0),
                "Head": (-7, 0, 0),
                "UpperArm.R": (-10, +3, 0),
                "Forearm.R": (-12, -4, +6),
                "UpperArm.L": (-10, -3, 0),
                "Forearm.L": (-12, +4, -6),
            },
        ),
        (
            45,
            {
                "Pelvis": ((0, 0, +2), (0.0, 0.0, +0.012)),
                "Spine": (+1, 0, +2),
                "Chest": (0, 0, +2),
                "Head": (-2, -2, -5),
                "UpperArm.R": (-10, +5, 0),
                "Forearm.R": (-12, -4, +6),
                "UpperArm.L": (-2, -5, 0),
                "Forearm.L": (-18, +4, -6),
            },
        ),
        (60, READY),
    ]
    for f, spec in idle_keys:
        key_all_bones(arm_obj, f, spec)

    track3 = arm_obj.animation_data.nla_tracks.new()
    track3.name = "AN_Brute_Idle"
    track3.strips.new("AN_Brute_Idle", 0, act_idle)
    arm_obj.animation_data.action = None

    return act_punch, act_combo, act_idle


# ---------------------------------------------------------------------------
# 5. Export GLB + Headless CPU Cycles Verification Renders
# ---------------------------------------------------------------------------
def setup_qa_render_scene():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 28
    scene.cycles.use_denoising = False
    scene.render.film_transparent = False

    world = bpy.data.worlds.new("QAWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.98, 0.98, 0.99, 1.0)
        bg.inputs[1].default_value = 0.85

    scene.render.use_freestyle = True
    scene.render.line_thickness = 1.45
    vl = bpy.context.view_layer
    vl.use_freestyle = True
    ls = vl.freestyle_settings.linesets[0] if len(vl.freestyle_settings.linesets) else vl.freestyle_settings.linesets.new("OutlineSet")
    ls.select_silhouette = True
    ls.select_border = True
    ls.select_crease = True
    vl.freestyle_settings.crease_angle = math.radians(132.0)
    if ls.linestyle is None:
        ls.linestyle = bpy.data.linestyles.new("InkStyle")
    ls.linestyle.color = (0.13, 0.09, 0.10)

    key_data = bpy.data.lights.new("KeySun", type="SUN")
    key_data.energy = 3.4
    key_data.color = (1.0, 0.92, 0.78)
    key_data.angle = math.radians(14.0)
    key_obj = bpy.data.objects.new("KeySun", key_data)
    bpy.context.collection.objects.link(key_obj)
    key_obj.rotation_euler = Euler((math.radians(50), math.radians(-20), math.radians(-30)), "XYZ")

    fill_data = bpy.data.lights.new("BackSun", type="SUN")
    fill_data.energy = 1.8
    fill_data.color = (0.82, 0.85, 1.0)
    fill_data.angle = math.radians(38.0)
    fill_obj = bpy.data.objects.new("BackSun", fill_data)
    bpy.context.collection.objects.link(fill_obj)
    fill_obj.rotation_euler = Euler((math.radians(62), math.radians(16), math.radians(150)), "XYZ")

    cam_data = bpy.data.cameras.new("QACam")
    cam_data.lens = 56.0
    cam_obj = bpy.data.objects.new("QACam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def aim_camera(cam_obj: bpy.types.Object, loc: tuple, target: tuple):
    cam_obj.location = Vector(loc)
    direction = Vector(target) - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def compose_grid_png(image_paths: list, cols: int, out_path: str):
    imgs = [bpy.data.images.load(p) for p in image_paths]
    w, h = imgs[0].size
    rows = math.ceil(len(imgs) / cols)
    canvas = np.ones((rows * h, cols * w, 4), dtype=np.float32)

    for idx, im in enumerate(imgs):
        r_idx = rows - 1 - (idx // cols)
        c_idx = idx % cols
        buf = np.empty(w * h * 4, dtype=np.float32)
        im.pixels.foreach_get(buf)
        buf = buf.reshape((h, w, 4))
        canvas[r_idx * h : (r_idx + 1) * h, c_idx * w : (c_idx + 1) * w, :] = buf
        bpy.data.images.remove(im)

    out_im = bpy.data.images.new("Sheet", width=cols * w, height=rows * h, alpha=False)
    out_im.colorspace_settings.name = "sRGB"
    out_im.pixels.foreach_set(canvas.ravel())
    out_im.filepath_raw = out_path
    out_im.file_format = "PNG"
    out_im.save()
    bpy.data.images.remove(out_im)


def validate_glb(glb_path: str):
    assert os.path.isfile(glb_path), f"Missing GLB: {glb_path}"
    size_bytes = os.path.getsize(glb_path)
    assert size_bytes > 20_000, f"GLB unexpectedly small: {size_bytes}"
    with open(glb_path, "rb") as f:
        magic, ver, _ = struct.unpack("<4sII", f.read(12))
        assert magic == b"glTF" and ver == 2
        clen, ctype = struct.unpack("<I4s", f.read(8))
        assert ctype == b"JSON"
        gltf = json.loads(f.read(clen).decode("utf-8"))

    anim_names = [a.get("name") for a in gltf.get("animations", [])]
    for expected in ("AN_Brute_Attack_Heavy_01", "AN_Brute_Attack_Combo_02", "AN_Brute_Idle"):
        assert expected in anim_names, f"Missing animation {expected} in {anim_names}"
    assert len(gltf.get("skins", [])) == 1
    assert len(gltf.get("images", [])) >= 1
    print(f"GLB_VALIDATED size={size_bytes} anims={anim_names} joints={len(gltf['skins'][0]['joints'])}")


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    atlas = create_atlas_image(1024)
    arm_obj = create_armature()
    create_character_mesh(arm_obj, atlas)
    act_punch, _, _ = build_animations(arm_obj)

    glb_path = os.path.join(PUBLIC_DIR, "masked_brawler.glb")
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format="GLB",
        export_animations=True,
        export_animation_mode="NLA_TRACKS",
        export_nla_strips_merged_animation_name="Animation",
        export_def_bones=False,
        export_yup=True,
        export_apply=False,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )
    validate_glb(glb_path)

    blend_path = os.path.join(ROOT_DIR, "masked_brawler.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    cam_obj = setup_qa_render_scene()
    scene = bpy.context.scene
    scene.render.resolution_x = 680
    scene.render.resolution_y = 540

    arm_obj.animation_data.action = act_punch
    scene.frame_set(0)

    # Match reference turnaround angles (Front-3/4 from slightly right-front +X/-Y so head faces slightly left!)
    # Wait: look at left side of reference image! In the reference Front-3/4 view, the character's chest/mask faces slightly TOWARD VIEWER'S LEFT (-X)!
    # That means the camera is slightly to the character's LEFT-FRONT (+X, -Y), e.g. (+0.85, -3.30, 1.78)!
    front_png = os.path.join(RENDERS_DIR, "turnaround_front.png")
    aim_camera(cam_obj, (+0.82, -3.35, 1.78), (-0.02, -0.04, 0.74))
    scene.render.filepath = front_png
    bpy.ops.render.render(write_still=True)

    back_png = os.path.join(RENDERS_DIR, "turnaround_back.png")
    aim_camera(cam_obj, (-0.82, +3.35, 1.78), (+0.02, 0.0, 0.74))
    scene.render.filepath = back_png
    bpy.ops.render.render(write_still=True)

    turnaround_sheet = os.path.join(RENDERS_DIR, "turnaround_sheet.png")
    compose_grid_png([front_png, back_png], cols=2, out_path=turnaround_sheet)

    # 8-frame Punch Contact Sheet (Front-3/4 combat camera framing full extended fist!)
    scene.render.resolution_x = 460
    scene.render.resolution_y = 390
    aim_camera(cam_obj, (+1.45, -4.15, 1.85), (-0.05, -0.30, 0.78))
    sample_frames = [0, 6, 11, 15, 17, 20, 27, 36]
    frame_paths = []
    for f in sample_frames:
        scene.frame_set(f)
        fp = os.path.join(RENDERS_DIR, f"punch_f{f:02d}.png")
        scene.render.filepath = fp
        bpy.ops.render.render(write_still=True)
        frame_paths.append(fp)

    contact_sheet = os.path.join(RENDERS_DIR, "punch_contact_sheet.png")
    compose_grid_png(frame_paths, cols=4, out_path=contact_sheet)
    print("RENDERS_COMPLETE", turnaround_sheet, contact_sheet)


if __name__ == "__main__":
    main()
