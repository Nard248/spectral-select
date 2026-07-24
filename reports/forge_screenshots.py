"""Populate a SpectraForge demo project in the GUI and grab polished screenshots (offscreen)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from spectraforge.gui.app import ForgeWindow

OUT = Path("/tmp/forge_shots"); OUT.mkdir(exist_ok=True)

app = QApplication([])
win = ForgeWindow()
st = win._state

# 1. MIX chemicals into materials via the panel (this also plots the spectrum)
win._material.compose_material("Collagen+NADH", {"collagen": 1.0, "NADH": 0.3})
win._material.compose_material("EGFP", {"EGFP": 0.8})
win._material.compose_material("FAD", {"FAD": 1.0})

# 2. PAINT them onto layers (draw shapes into the amount maps)
H = W = st.height
l1 = st.add_layer("collagen patch", st.materials["Collagen+NADH"]); l1.amount_map[5:40, 5:40] = 1.0
l2 = st.add_layer("FAD spot", st.materials["FAD"])
yy, xx = np.ogrid[:H, :W]; l2.amount_map[(yy - 45) ** 2 + (xx - 45) ** 2 <= 12 ** 2] = 1.0
l3 = st.add_layer("EGFP stripe", st.materials["EGFP"]); l3.amount_map[50:60, 5:60] = 0.8
st.active_layer = 0

# 3. RENDER and light up the emission-band preview
win._acquire.render_now()          # sets last_render + _after_render() -> preview shows a band
win._refresh_all()
app.processEvents(); app.processEvents()

# --- hero: full workbench (Library tab active) ---
win.resize(1500, 950)
win.grab().save(str(OUT / "shot_full.png"))

# --- mixing: bring the Materials tab forward, grab full again ---
left_tab = win._material.parentWidget()
try:
    # the QTabWidget holding Library/Materials
    tabw = win._material
    while tabw is not None and tabw.__class__.__name__ != "QTabWidget":
        tabw = tabw.parentWidget()
    if tabw is not None:
        tabw.setCurrentWidget(win._material)
    app.processEvents()
except Exception as e:
    print("tab switch warn:", e)
win.grab().save(str(OUT / "shot_full_materials.png"))

# --- individual panels ---
for name, w in {"shot_library": win._library, "shot_material": win._material,
                "shot_layers": win._layers, "shot_canvas": win._canvas,
                "shot_acquire": win._acquire}.items():
    pm = w.grab(); pm.save(str(OUT / f"{name}.png"))
    print(f"{name}: {pm.width()}x{pm.height()}")
print("done")
