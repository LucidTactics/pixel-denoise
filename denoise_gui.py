"""
denoise_gui.py -- interactive front-end for pixel_denoise.

Top section = a few high-level "amount" sliders (0..10) you just turn up:
    Overall strength | White/bright confetti | Coloured confetti | Edge & fringe
Each drives a group of advanced thresholds. Tick "Show advanced" to fine-tune the
raw thresholds underneath (the advanced sliders track the macros, and override
them once you drag one).

Three live panels: ORIGINAL | CLEANED | DETECTION (cyan=removed, magenta=replaced).

Launch:  python pixel_denoise.py gui [image.png]   (or double-click Denoise.vbs)
"""
from __future__ import annotations

import json
import os
import tkinter as tk
from tkinter import filedialog, ttk
from dataclasses import asdict

import numpy as np
from PIL import Image, ImageTk

import pixel_denoise as pd

# Macro sliders: (name, label)
MACROS = [(m, pd.MACRO_LABELS[m]) for m in pd.MACROS]

# Advanced sliders: (attr, label, lo, hi, step). Grouped by channel for clarity.
FEATURE_PARAMS = {"alpha_thresh", "radius"}
ADVANCED = [
    ("second_thresh",      "Isolation thresh",   5.0, 80.0, 1.0),
    ("edge_second",        "Edge isolation",     5.0, 60.0, 1.0),
    ("edge_frac",          "Edge width",         0.0, 0.5,  0.02),
    ("bright_spike",       "Bright spike",       5.0, 60.0, 1.0),
    ("bright_chroma_max",  "Bright max-chroma",  0.0, 40.0, 1.0),
    ("bright_cluster_max", "Bright cluster px",  0,   30,   1),
    ("color_spike",        "Colour spike",       5.0, 50.0, 1.0),
    ("color_cluster_max",  "Colour cluster px",  0,   30,   1),
    ("island_max",         "Island max px",      0,   20,   1),
    ("fringe_thresh",      "Remove vs replace",  0.0, 1.0,  0.05),
    ("alpha_thresh",       "Alpha cutoff",       0,   254,  1),
    ("radius",             "Radius",             1,   2,    1),
]


class App:
    def __init__(self, root: tk.Tk, path: str | None):
        self.root = root
        root.title("Pixel Denoise")
        self.features = None
        self.img = None
        self.path = None
        self.zoom = 3
        self._job = None
        self._rebuild_pending = False
        self._syncing = False   # guards programmatic var.set from firing callbacks
        # Restore last session (macros + advanced overrides); falls back to defaults.
        saved_image = self.load_settings()

        ctrl = ttk.Frame(root, padding=8)
        ctrl.grid(row=0, column=0, sticky="ns")
        self.canvas = tk.Label(root, bg="#202020")
        self.canvas.grid(row=0, column=1, sticky="nsew")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        ttk.Button(ctrl, text="Open image…", command=self.open).pack(fill="x")
        self.fname = ttk.Label(ctrl, text="(no image)", width=30)
        self.fname.pack(fill="x", pady=(2, 8))

        ttk.Label(ctrl, text="QUICK CONTROLS", font=("", 8, "bold")).pack(anchor="w")
        self.macro_vals = {}
        for name, label in MACROS:
            self._make_macro(ctrl, name, label)

        self.show_adv = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Show advanced", variable=self.show_adv,
                        command=self._toggle_adv).pack(anchor="w", pady=(8, 0))
        self.adv_frame = ttk.Frame(ctrl)
        self.adv_vals = {}
        for attr, label, lo, hi, step in ADVANCED:
            self._make_scale(self.adv_frame, label, getattr(self.params, attr), lo, hi, step,
                             lambda v, a=attr, st=step: self.on_advanced(a, st, v),
                             store=(self.adv_vals, attr))

        zrow = ttk.Frame(ctrl)
        self.zrow = zrow
        zrow.pack(fill="x", pady=(8, 1))
        ttk.Label(zrow, text="Zoom", width=15).pack(side="left")
        self.zvar = tk.IntVar(value=self.zoom)
        ttk.Scale(zrow, from_=1, to=12, variable=self.zvar, orient="horizontal",
                  command=lambda _v: self._set_zoom()).pack(side="left", fill="x", expand=True)

        self.stats = ttk.Label(ctrl, text="", width=30)
        self.stats.pack(fill="x", pady=8)
        ttk.Button(ctrl, text="Save cleaned…", command=self.save).pack(fill="x")
        ttk.Button(ctrl, text="Apply to folder…", command=self.batch).pack(fill="x", pady=2)
        brow = ttk.Frame(ctrl)
        brow.pack(fill="x")
        ttk.Button(brow, text="Restore defaults", command=self.restore_defaults).pack(
            side="left", fill="x", expand=True)
        ttk.Button(brow, text="Print params", command=self.print_params).pack(
            side="left", fill="x", expand=True)

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        to_open = path or saved_image
        if to_open and os.path.exists(to_open):
            self.load(to_open)

    # ---- widget helpers ----
    def _make_macro(self, parent, name, label):
        """Macro slider: label (+value) on its own line, a wide track underneath."""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(3, 5))
        top = ttk.Frame(frame)
        top.pack(fill="x")
        ttk.Label(top, text=label).pack(side="left")
        vlabel = ttk.Label(top, text="", width=5, anchor="e")
        vlabel.pack(side="right")
        var = tk.DoubleVar(value=self.macros[name])
        ttk.Scale(frame, from_=0, to=pd.MACRO_MAX, variable=var, orient="horizontal",
                  length=260, command=lambda v, n=name: self.on_macro(n, v)).pack(fill="x")
        self.macro_vals[name] = (var, vlabel, 0.5)
        vlabel.config(text=f"{self.macros[name]:.1f}")

    def _make_scale(self, parent, label, value, lo, hi, step, cb, store):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=16).pack(side="left")
        var = tk.DoubleVar(value=value)
        vlabel = ttk.Label(row, text="", width=5)
        vlabel.pack(side="right")
        ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal",
                  command=lambda v, c=cb: c(v)).pack(side="left", fill="x", expand=True)
        store[0][store[1]] = (var, vlabel, step)
        self._fmt(vlabel, value, step)

    def _fmt(self, lbl, v, step):
        lbl.config(text=str(int(round(v))) if step >= 1 else f"{float(v):.2f}")

    def _toggle_adv(self):
        if self.show_adv.get():
            self.adv_frame.pack(fill="x", before=self.zrow)  # sit above the zoom row
        else:
            self.adv_frame.forget()

    # ---- callbacks ----
    def on_macro(self, name, v):
        if self._syncing:
            return
        v = round(float(v) * 2) / 2
        self.macros[name] = v
        self.macro_vals[name][1].config(text=f"{v:.1f}")
        pd.apply_macro(self.params, name, v)
        self._sync_advanced()          # reflect new thresholds in advanced sliders
        self._debounce(rebuild=False)

    def on_advanced(self, attr, step, v):
        if self._syncing:
            return
        v = int(round(float(v))) if step >= 1 else round(float(v) / step) * step
        setattr(self.params, attr, v)
        self.adv_vals[attr][1].config(text=str(int(v)) if step >= 1 else f"{v:.2f}")
        self._debounce(rebuild=attr in FEATURE_PARAMS)

    def _sync_advanced(self):
        self._syncing = True
        for attr, (var, vlabel, step) in self.adv_vals.items():
            val = getattr(self.params, attr)
            var.set(val)
            self._fmt(vlabel, val, step)
        self._syncing = False

    def _set_zoom(self):
        self.zoom = self.zvar.get()
        self._debounce(rebuild=False)

    def _debounce(self, rebuild):
        if self._job:
            self.root.after_cancel(self._job)
        self._rebuild_pending = self._rebuild_pending or rebuild
        self._job = self.root.after(40, lambda: self.render(self._rebuild_pending))

    # ---- IO ----
    def open(self):
        p = filedialog.askopenfilename(filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if p:
            self.load(p)

    def load(self, path):
        self.path = path
        self.img = Image.open(path).convert("RGBA")
        self.fname.config(text=os.path.basename(path))
        self.render(rebuild=True)

    def save(self):
        if self.img is None:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=os.path.splitext(os.path.basename(self.path))[0] + "_clean.png")
        if p:
            out, *_ = pd.clean(self.features, self.params)
            Image.fromarray(out, "RGBA").save(p)

    def batch(self):
        d = filedialog.askdirectory(title="Folder of PNGs to clean")
        if not d:
            return
        outdir = os.path.join(d, "cleaned")
        os.makedirs(outdir, exist_ok=True)
        n = 0
        for fn in os.listdir(d):
            if fn.lower().endswith(".png"):
                f = pd.compute_features(Image.open(os.path.join(d, fn)), self.params)
                out, *_ = pd.clean(f, self.params)
                Image.fromarray(out, "RGBA").save(os.path.join(outdir, fn))
                n += 1
        self.stats.config(text=f"batch: {n} files -> {outdir}")

    def print_params(self):
        print("macros:", self.macros)
        print("params:", asdict(self.params))

    # ---- persistence ----
    def _settings_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "denoise_settings.json")

    def load_settings(self):
        """Restore macros + params + last image. Returns the saved image path (or
        None). Robust to a missing/old/corrupt file -> falls back to defaults."""
        try:
            d = json.load(open(self._settings_path()))
            m = d.get("macros") or {}
            self.macros = {k: float(m.get(k, pd.DEFAULT_MACROS[k])) for k in pd.MACROS}
            pdict = d.get("params")
            if pdict:                       # merge onto a fresh Params (tolerate schema drift)
                base = asdict(pd.Params())
                base.update({k: v for k, v in pdict.items() if k in base})
                self.params = pd.Params(**base)
            else:
                self.params = pd.params_from_macros(self.macros)
            return d.get("image")
        except Exception:
            self.macros = dict(pd.DEFAULT_MACROS)
            self.params = pd.params_from_macros(self.macros)
            return None

    def save_settings(self):
        try:
            json.dump({"macros": self.macros, "params": asdict(self.params),
                       "image": self.path}, open(self._settings_path(), "w"), indent=2)
        except Exception:
            pass

    def restore_defaults(self):
        self.macros = dict(pd.DEFAULT_MACROS)
        self.params = pd.params_from_macros(self.macros)
        self._syncing = True
        for n, (var, vlabel, _step) in self.macro_vals.items():
            var.set(self.macros[n])
            vlabel.config(text=f"{self.macros[n]:.1f}")
        for attr, (var, vlabel, step) in self.adv_vals.items():
            val = getattr(self.params, attr)
            var.set(val)
            self._fmt(vlabel, val, step)
        self._syncing = False
        self.render(rebuild=True)           # alpha/radius may have changed
        self.save_settings()

    def _on_close(self):
        self.save_settings()
        self.root.destroy()

    # ---- render ----
    def render(self, rebuild):
        self._job = None
        self._rebuild_pending = False
        if self.img is None:
            return
        if rebuild or self.features is None:
            self.features = pd.compute_features(self.img, self.params)
        f = self.features
        out, flag, remove, replace = pd.clean(f, self.params)

        orig = np.dstack([f.rgb, f.alpha]).astype(np.uint8)
        det = orig.copy()
        det[remove] = (0, 220, 220, 255)
        det[replace] = (230, 0, 230, 255)

        panels = [self._panel(orig), self._panel(out), self._panel(det)]
        gap = 8
        h = max(p.height for p in panels)
        w = sum(p.width for p in panels) + gap * (len(panels) - 1)
        strip = Image.new("RGBA", (w, h), (32, 32, 32, 255))
        x = 0
        for pn in panels:
            strip.alpha_composite(pn, (x, 0))
            x += pn.width + gap
        self._tk = ImageTk.PhotoImage(strip)
        self.canvas.config(image=self._tk)
        self.stats.config(text=f"flagged {int(flag.sum())} | removed {int(remove.sum())} "
                                f"| replaced {int(replace.sum())}")
        self.save_settings()   # persist current macros/params for next session

    def _panel(self, rgba):
        z = self.zoom
        im = Image.fromarray(rgba, "RGBA").resize(
            (rgba.shape[1] * z, rgba.shape[0] * z), Image.NEAREST)
        arr = np.asarray(im).astype(np.float64)
        h, w = arr.shape[:2]
        c = max(1, 8 * z)
        check = (((np.arange(w) // c)[None, :] + (np.arange(h) // c)[:, None]) % 2)
        bg = np.where(check[..., None] == 0,
                      np.array([80, 80, 80]), np.array([60, 60, 60])).astype(np.float64)
        a = arr[..., 3:4] / 255.0
        comp = (arr[..., :3] * a + bg * (1 - a)).astype(np.uint8)
        return Image.fromarray(comp, "RGB").convert("RGBA")


def launch(path: str | None = None):
    root = tk.Tk()
    App(root, path)
    root.mainloop()


if __name__ == "__main__":
    import sys
    launch(sys.argv[1] if len(sys.argv) > 1 else None)
