"""
pixel_denoise.py  --  Remove AI-generation "confetti" noise from pixel-art sprites.

AI pixel-art generators (PixelLab etc.) sprinkle three kinds of single-pixel
noise into otherwise-clean sprites:

  1. Chromatic outliers ("confetti")  -- a lone pixel whose HUE is alien to its
     neighbours (a magenta speck in a green canopy). Most eye-catching.
  2. Luminance spikes (salt & pepper)  -- a lone pixel far brighter/darker than
     its neighbours but roughly the right hue.
  3. Edge-fringe specks / floating islands  -- 1-2px blobs on or just outside the
     alpha silhouette, often mis-hued, that the generator "leaked" past the shape.

The detector scores every opaque pixel ENTIRELY RELATIVE TO ITS LOCAL
NEIGHBOURHOOD (median in CIELAB), so the same thresholds work on a green tree, a
red rock, or a blue banner without retuning. It is deliberately selective: a
pixel is only flagged when it is a strong outlier AND corroborated by at least
one "this is isolated noise, not legit detail" signal.

Flagged pixels are then either REMOVED (alpha->0, for fringe specks & islands) or
REPLACED (vector-median of clean neighbours, for interior pixels) based on how
close they sit to the transparent edge.

Design note: every feature map that does not depend on a slider is computed once
in `compute_features`; `apply_decision` only thresholds those maps, so a GUI can
drag sliders and recomposite in milliseconds.

CLI:
  python pixel_denoise.py clean  IN.png [more.png ...] -o OUTDIR   # batch clean
  python pixel_denoise.py overlay IN.png -o overlay.png           # debug overlay
  python pixel_denoise.py calibrate ORIGINAL.png FLAGGED.png      # precision/recall vs hand-flags
  python pixel_denoise.py gui [IN.png]                            # interactive sliders
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image
from scipy import ndimage


# sRGB (0..255 uint8 or 0..1 float) -> CIELAB (D65). Implemented locally so the
# tool depends only on numpy/scipy/Pillow (lighter, reliable PyInstaller builds).
_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (...,3) in [0,1]. Returns (...,3) Lab (L 0..100)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = lin @ _M_RGB2XYZ.T / _WHITE_D65
    d = 6.0 / 29.0
    fx = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d ** 2) + 4.0 / 29.0)
    L = 116.0 * fx[..., 1] - 16.0
    a = 500.0 * (fx[..., 0] - fx[..., 1])
    b = 200.0 * (fx[..., 1] - fx[..., 2])
    return np.stack([L, a, b], axis=-1)

# Bright purple used to paint detections in debug overlays (matches the user's
# hand-flag colour so overlays can be visually diffed against their flags).
FLAG_RGB = (200, 0, 255)


# --------------------------------------------------------------------------- #
#  Parameters (these are the GUI "power level" sliders)                        #
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    """Detection is the OR of independent channels, each targeting one kind of
    noise. The GUI's high-level "amount" sliders just move the thresholds below;
    a threshold set very high (or a cluster size of 0) turns its channel off."""
    alpha_thresh: int = 128      # pixel counts as opaque when alpha > this
    radius: int = 1              # neighbourhood radius (1 -> 3x3, 2 -> 5x5)

    # --- Isolation channel: general lone/paired specks of ANY colour ---
    # second_min = CIELAB dE to the 2nd-nearest opaque neighbour. A lone speck has
    # no real similar neighbour, so its 2nd-nearest is far; a legit edge/highlight
    # runs in a streak, so its 2nd-nearest is close.
    second_thresh: float = 40.0  # interior: flag when 2nd-nearest dE exceeds this
    edge_second: float = 24.0    # near an alpha edge, this lower bar suffices
    edge_frac: float = 0.08      # transp fraction in 5x5 to count as "near edge"

    # --- Bright/dark-confetti channel: scattered luminance specks ---
    # Catches specks the isolation channel misses when they clump (a 2-3px blob
    # has a similar neighbour, so its second_min stays low). A speck is brighter
    # (or darker) than its surroundings; `bright_chroma_max` bounds how saturated
    # it may be (low = near-grey/white only, protecting green leaf-tips; the macro
    # widens it as you turn the channel up, to also catch pale-coloured specks).
    bright_spike: float = 22.0       # L above neighbour-median L to be a bright speck
    dark_spike: float = 55.0         # L below neighbour-median L to be a dark speck (high=off)
    bright_chroma_max: float = 17.0  # max chroma to still count (widens with the macro)
    bright_cluster_max: int = 7      # only small scattered blobs, never the big canopy

    # --- Colour-confetti channel: saturated, isolated, scattered specks ---
    color_spike: float = 18.0    # chroma above neighbour-median chroma
    color_cluster_max: int = 6   # only small scattered saturated blobs

    # --- Islands + remove/replace ---
    island_max: int = 3          # opaque component <= this many px -> floating speck
    fringe_thresh: float = 0.45  # transp fraction in 5x5 above this -> remove (vs replace)


# --------------------------------------------------------------------------- #
#  Low-level helpers                                                            #
# --------------------------------------------------------------------------- #
def _neighbour_stack(arr: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    """Return a (K, H, W) stack of the K neighbours of each pixel (excluding the
    centre). Neighbours that are out-of-bounds or invalid are set to NaN so that
    nan-aware reductions ignore them."""
    h, w = arr.shape
    a = np.where(valid, arr.astype(np.float64), np.nan)
    ap = np.pad(a, radius, mode="constant", constant_values=np.nan)
    out = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            out.append(ap[radius + dy:radius + dy + h, radius + dx:radius + dx + w])
    return np.stack(out, axis=0)


# --------------------------------------------------------------------------- #
#  Feature computation (slider-independent, run once per image)                 #
# --------------------------------------------------------------------------- #
@dataclass
class Features:
    rgb: np.ndarray          # (H,W,3) uint8
    alpha: np.ndarray        # (H,W) uint8  (255 if image had no alpha)
    opaque: np.ndarray       # (H,W) bool   (alpha > alpha_thresh, frozen at load)
    second_min: np.ndarray   # (H,W) float  dE to 2nd-nearest neighbour (isolation)
    chroma_spike: np.ndarray # (H,W) float  chroma - neighbour median chroma
    bright: np.ndarray       # (H,W) float  signed L - neighbour median L (+ = brighter)
    chroma_abs: np.ndarray   # (H,W) float  absolute chroma sqrt(a^2+b^2)
    transp_frac: np.ndarray  # (H,W) float  transparent fraction in 5x5
    island_size: np.ndarray  # (H,W) float  size of this pixel's opaque component


def compute_features(img: Image.Image, p: Params) -> Features:
    img = img.convert("RGBA")
    arr = np.asarray(img)
    rgb = arr[..., :3].astype(np.uint8)
    alpha = arr[..., 3].astype(np.uint8)
    opaque = alpha > p.alpha_thresh

    lab = rgb2lab(rgb.astype(np.float64) / 255.0)  # L 0..100, a/b ~ -128..127
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(A, B)

    r = p.radius
    Ls = _neighbour_stack(L, opaque, r)
    As = _neighbour_stack(A, opaque, r)
    Bs = _neighbour_stack(B, opaque, r)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # All-NaN slices
        Lmed = np.nanmedian(Ls, axis=0)
        Cs = np.hypot(As, Bs)
        Cmed = np.nanmedian(Cs, axis=0)

    # Per-neighbour dE, then the 2nd-smallest = distance to the 2nd-nearest
    # neighbour. This is the isolation signal: high for lone/paired specks, low
    # for any pixel that belongs to a coherent streak of similar colour.
    nb_de = np.sqrt((Ls - L[None]) ** 2 + (As - A[None]) ** 2 + (Bs - B[None]) ** 2)
    nb_de = np.where(np.isnan(nb_de), np.inf, nb_de)
    nb_de.sort(axis=0)
    second_min = nb_de[1] if nb_de.shape[0] > 1 else nb_de[0]
    # A fully isolated island pixel (all neighbours transparent) -> inf; treat as
    # maximally isolated so it is always caught (and later removed as an island).
    second_min = np.where(np.isfinite(second_min), second_min, 999.0)

    chroma_spike = np.nan_to_num(chroma - Cmed, nan=0.0)
    bright = np.nan_to_num(L - Lmed, nan=0.0)  # signed: + brighter than neighbours

    # Transparent fraction in a 5x5 window (out-of-bounds counts as transparent).
    opaque_f = opaque.astype(np.float64)
    transp_frac = 1.0 - ndimage.uniform_filter(opaque_f, size=5, mode="constant", cval=0.0)

    # Size of each pixel's opaque connected component (8-connectivity).
    struct = np.ones((3, 3), dtype=bool)
    lbl, n = ndimage.label(opaque, structure=struct)
    if n > 0:
        sizes = ndimage.sum(np.ones_like(lbl, dtype=np.float64), lbl, index=np.arange(1, n + 1))
        size_map = np.zeros_like(lbl, dtype=np.float64)
        nz = lbl > 0
        size_map[nz] = sizes[lbl[nz] - 1]
    else:
        size_map = np.zeros_like(lbl, dtype=np.float64)

    return Features(rgb, alpha, opaque, second_min, chroma_spike, bright, chroma,
                    transp_frac, size_map)


# --------------------------------------------------------------------------- #
#  Decision (slider-dependent, cheap)                                          #
# --------------------------------------------------------------------------- #
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _small_clusters(mask: np.ndarray, max_size: int) -> np.ndarray:
    """Keep only pixels of `mask` that belong to a connected blob of <= max_size
    pixels. This is what distinguishes scattered noise specks from a large legit
    feature (a sunlit canopy, a highlight streak): same per-pixel test, but the
    big coherent region is excluded by its size."""
    if max_size <= 0 or not mask.any():
        return np.zeros_like(mask)
    lbl, n = ndimage.label(mask, structure=_STRUCT8)
    if n == 0:
        return np.zeros_like(mask)
    sizes = ndimage.sum(np.ones_like(lbl, dtype=np.float64), lbl, index=np.arange(1, n + 1))
    szmap = np.zeros(lbl.shape, dtype=np.float64)
    nz = lbl > 0
    szmap[nz] = sizes[lbl[nz] - 1]
    return mask & (szmap <= max_size)


def detect(f: Features, p: Params) -> np.ndarray:
    """OR of independent noise channels (see Params). Each channel targets one
    kind of artifact and is driven by its own thresholds, so the GUI's high-level
    sliders can dial each up/down without disturbing the others."""
    op = f.opaque

    # Isolation channel: lone/paired specks of any colour. Interior pixels must
    # clear `second_thresh`; near an alpha edge the lower `edge_second` applies.
    near_edge = f.transp_frac >= p.edge_frac
    isolation = (f.second_min > p.second_thresh) | (near_edge & (f.second_min > p.edge_second))

    # Bright/dark channel: scattered blobs much brighter OR darker than their
    # neighbours, within the chroma ceiling, in a small cluster (so the big legit
    # sunlit canopy / shadow mass is never touched).
    lum_spike = (f.bright > p.bright_spike) | (-f.bright > p.dark_spike)
    speck = _small_clusters(
        op & lum_spike & (f.chroma_abs < p.bright_chroma_max),
        p.bright_cluster_max)

    # Colour channel: scattered, more-saturated, somewhat-isolated blobs.
    color = _small_clusters(
        op & (f.chroma_spike > p.color_spike) & (f.second_min > p.edge_second),
        p.color_cluster_max)

    flag = op & (isolation | speck | color)
    # Floating specks: any opaque pixel in a tiny disconnected island.
    flag |= op & (f.island_size <= p.island_max)
    return flag


def classify(flag: np.ndarray, f: Features, p: Params):
    """Split flagged pixels into (remove_mask, replace_mask)."""
    is_island = f.island_size <= p.island_max
    is_fringe = f.transp_frac > p.fringe_thresh
    remove = flag & (is_island | is_fringe)
    replace = flag & ~remove
    return remove, replace


# --------------------------------------------------------------------------- #
#  High-level "amount" macros (0..20). Each macro drives a few advanced params  #
#  so the GUI can offer a handful of simple "turn it up" sliders, with the      #
#  advanced thresholds available underneath. 0 turns a channel off entirely.    #
#                                                                               #
#  The ramp is PIECEWISE: v in [1,10] goes weak -> old-max (so v=10 reproduces  #
#  the previous slider's strongest pass), then v in [10,20] pushes old-max ->   #
#  a NEW stronger endpoint, giving extra headroom beyond what 0..10 could do.   #
# --------------------------------------------------------------------------- #
MACROS = ("strength", "white", "color", "edge")
MACRO_LABELS = {
    "strength": "Overall strength",
    "white": "Bright / dark confetti",
    "color": "Coloured confetti",
    "edge": "Edge & fringe cleanup",
}
MACRO_MAX = 20.0
# Calibrated defaults (recall ~0.95). These sit on the [1,10] half, so they match
# the original tuning exactly; 10..20 is the new extra-strength region.
DEFAULT_MACROS = {"strength": 6.0, "white": 7.0, "color": 7.0, "edge": 8.0}

# Per-param ramp points: (weak @v=1, old-max @v=10, new-strong @v=20).
_MACRO_RAMP = {
    "strength": {"second_thresh": (65.0, 18.0, 10.0)},
    # As "white" climbs it lowers the bright bar, widens the chroma ceiling (so
    # pale-coloured brights get caught too), and engages dark-speck detection.
    "white":    {"bright_spike": (40.0, 12.0, 7.0),
                 "bright_chroma_max": (12.0, 22.0, 34.0),
                 "dark_spike": (90.0, 40.0, 22.0),
                 "bright_cluster_max": (4, 8, 12)},
    "color":    {"color_spike": (34.0, 10.0, 6.0),  "color_cluster_max": (3, 12, 18)},
    "edge":     {"edge_frac": (0.20, 0.04, 0.0), "edge_second": (40.0, 14.0, 8.0),
                 "island_max": (1, 8, 12)},
}
_INT_PARAMS = {"bright_cluster_max", "color_cluster_max", "island_max"}
# Values that switch a channel off when its macro is at 0.
_MACRO_OFF = {
    "strength": {"second_thresh": 999.0},
    "white":    {"bright_spike": 999.0, "dark_spike": 999.0, "bright_cluster_max": 0},
    "color":    {"color_spike": 999.0, "color_cluster_max": 0},
    "edge":     {"edge_frac": 1.0, "edge_second": 999.0, "island_max": 0},
}


def _ramp(v: float, a: float, b: float, c: float) -> float:
    """Piecewise: a->b over v in [1,10], b->c over v in [10,20]."""
    if v <= 10.0:
        return a + (b - a) * (v - 1.0) / 9.0
    return b + (c - b) * (v - 10.0) / 10.0


def apply_macro(p: Params, name: str, v: float) -> None:
    """Mutate `p`'s advanced fields for one macro at amount `v` in [0, MACRO_MAX]."""
    v = max(0.0, min(MACRO_MAX, float(v)))
    if v <= 0:
        for attr, val in _MACRO_OFF[name].items():
            setattr(p, attr, val)
        return
    for attr, (a, b, c) in _MACRO_RAMP[name].items():
        val = _ramp(v, a, b, c)
        setattr(p, attr, int(round(val)) if attr in _INT_PARAMS else val)


def params_from_macros(macros: dict) -> Params:
    """Build a Params from a {macro_name: amount} dict (missing -> default)."""
    p = Params()
    for name in MACROS:
        apply_macro(p, name, macros.get(name, DEFAULT_MACROS[name]))
    return p


# --------------------------------------------------------------------------- #
#  Apply (produce cleaned RGBA)                                                 #
# --------------------------------------------------------------------------- #
def _median_fill_rgb(rgb: np.ndarray, valid: np.ndarray, radius: int) -> np.ndarray:
    """Per-pixel median RGB over `valid` neighbours (NaN where none)."""
    h, w = valid.shape
    out = np.full((h, w, 3), np.nan)
    for c in range(3):
        st = _neighbour_stack(rgb[..., c], valid, radius)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # All-NaN slices
            out[..., c] = np.nanmedian(st, axis=0)
    return out


def clean(f: Features, p: Params):
    """Return (cleaned_rgba uint8, flag, remove, replace)."""
    flag = detect(f, p)
    remove, replace = classify(flag, f, p)

    out_rgb = f.rgb.copy()
    out_alpha = f.alpha.copy()

    # Replacement: median of CLEAN opaque neighbours (exclude flagged pixels so
    # adjacent noise can't fill each other). Grow the radius once if a pixel has
    # no clean neighbours nearby.
    if replace.any():
        clean_valid = f.opaque & ~flag
        fill = _median_fill_rgb(f.rgb, clean_valid, p.radius)
        need = replace & np.isnan(fill[..., 0])
        if need.any():
            fill2 = _median_fill_rgb(f.rgb, clean_valid, p.radius + 1)
            fill = np.where(np.isnan(fill), fill2, fill)
        # Anything still unfillable becomes a removal instead.
        still = replace & np.isnan(fill[..., 0])
        remove = remove | still
        replace = replace & ~still
        ys, xs = np.where(replace)
        out_rgb[ys, xs] = np.clip(np.round(fill[ys, xs]), 0, 255).astype(np.uint8)

    out_alpha[remove] = 0
    out = np.dstack([out_rgb, out_alpha]).astype(np.uint8)
    return out, flag, remove, replace


def overlay(f: Features, p: Params) -> np.ndarray:
    """RGBA image with detected pixels painted in FLAG_RGB for visual diffing."""
    flag = detect(f, p)
    out = np.dstack([f.rgb.copy(), f.alpha.copy()]).astype(np.uint8)
    out[flag, 0], out[flag, 1], out[flag, 2] = FLAG_RGB
    out[flag, 3] = 255
    return out


# --------------------------------------------------------------------------- #
#  Calibration against hand-flagged ground truth                               #
# --------------------------------------------------------------------------- #
def purple_mask(rgb: np.ndarray) -> np.ndarray:
    """Detect the bright-purple hand-flag colour in a flagged image."""
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    return (r > 110) & (b > 110) & (g < 110) & (r - g > 55) & (b - g > 35)


def calibrate(original: Image.Image, flagged: Image.Image, p: Params, tol: int = 1):
    """Compare detector output to hand-flags. `tol` allows a 1px registration
    slack (a detection within `tol` of a true flag counts as a hit)."""
    f = compute_features(original, p)
    pred = detect(f, p)
    flg = np.asarray(flagged.convert("RGB"))
    gt = purple_mask(flg)
    if gt.shape != pred.shape:
        raise SystemExit(f"size mismatch: original {pred.shape} vs flagged {gt.shape}")

    if tol > 0:
        struct = np.ones((2 * tol + 1, 2 * tol + 1), bool)
        gt_d = ndimage.binary_dilation(gt, struct)
        pred_d = ndimage.binary_dilation(pred, struct)
    else:
        gt_d = pred_d = None

    tp = int((pred & (gt if tol == 0 else gt_d)).sum())
    fp = int((pred & ~(gt if tol == 0 else gt_d)).sum())
    matched_gt = int((gt & (pred if tol == 0 else pred_d)).sum())
    fn = int(gt.sum()) - matched_gt
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = matched_gt / gt.sum() if gt.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "gt_pixels": int(gt.sum()), "pred_pixels": int(pred.sum()),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
    }


# --------------------------------------------------------------------------- #
#  CLI                                                                          #
# --------------------------------------------------------------------------- #
def _save(arr: np.ndarray, path: str):
    Image.fromarray(arr, "RGBA").save(path)


def cmd_clean(args):
    p = params_from_macros(DEFAULT_MACROS)
    os.makedirs(args.out, exist_ok=True)
    for src in args.inputs:
        img = Image.open(src)
        f = compute_features(img, p)
        out, flag, remove, replace = clean(f, p)
        base = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(args.out, base + ".png")
        _save(out, dst)
        if args.overlay:
            _save(overlay(f, p), os.path.join(args.out, base + "_overlay.png"))
        print(f"{src}: flagged={int(flag.sum())} removed={int(remove.sum())} "
              f"replaced={int(replace.sum())} -> {dst}")


def cmd_overlay(args):
    p = params_from_macros(DEFAULT_MACROS)
    img = Image.open(args.input)
    f = compute_features(img, p)
    _save(overlay(f, p), args.out)
    print(f"overlay -> {args.out}  (flagged={int(detect(f, p).sum())})")


def cmd_calibrate(args):
    p = params_from_macros(DEFAULT_MACROS)
    res = calibrate(Image.open(args.original), Image.open(args.flagged), p, tol=args.tol)
    print("Params:", asdict(p))
    for k, v in res.items():
        print(f"  {k:12} {v}")


def cmd_gui(args):
    import denoise_gui  # noqa: deferred import; only needed for GUI
    denoise_gui.launch(args.input)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("clean", help="batch-clean PNGs")
    c.add_argument("inputs", nargs="+")
    c.add_argument("-o", "--out", default="cleaned")
    c.add_argument("--overlay", action="store_true", help="also write debug overlays")
    c.set_defaults(func=cmd_clean)

    o = sub.add_parser("overlay", help="write a debug overlay")
    o.add_argument("input")
    o.add_argument("-o", "--out", default="overlay.png")
    o.set_defaults(func=cmd_overlay)

    cal = sub.add_parser("calibrate", help="precision/recall vs hand-flagged image")
    cal.add_argument("original")
    cal.add_argument("flagged")
    cal.add_argument("--tol", type=int, default=1, help="px registration slack")
    cal.set_defaults(func=cmd_calibrate)

    g = sub.add_parser("gui", help="interactive slider GUI")
    g.add_argument("input", nargs="?")
    g.set_defaults(func=cmd_gui)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
