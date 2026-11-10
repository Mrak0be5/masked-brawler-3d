"""Self-check for Masked Brawler 3D asset & web viewer."""

import json
import os
import struct

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
glb_path = os.path.join(ROOT, "masked_brawler.glb")
html_path = os.path.join(ROOT, "index.html")

assert os.path.isfile(glb_path) and os.path.getsize(glb_path) > 100_000
assert os.path.isfile(html_path)

html = open(html_path, "r", encoding="utf-8").read()
for req in ("./masked_brawler.glb", "AN_Brute_Attack_Heavy_01", "AN_Brute_Attack_Combo_02", "AN_Brute_Idle"):
    assert req in html, f"Missing {req}"

with open(glb_path, "rb") as f:
    magic, ver, _ = struct.unpack("<4sII", f.read(12))
    assert magic == b"glTF" and ver == 2
    clen, ctype = struct.unpack("<I4s", f.read(8))
    assert ctype == b"JSON"
    gltf = json.loads(f.read(clen).decode("utf-8"))

anims = {a["name"] for a in gltf.get("animations", [])}
assert {"AN_Brute_Attack_Heavy_01", "AN_Brute_Attack_Combo_02", "AN_Brute_Idle"} <= anims
assert len(gltf["skins"][0]["joints"]) == 21
print("VERIFY_OK", sorted(anims))
