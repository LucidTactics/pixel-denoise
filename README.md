# pixel_denoise — AI sprite confetti/speck remover

Removes the single-pixel noise that AI pixel-art generators (PixelLab etc.)
sprinkle into sprites: bright/dark luminance specks, chromatic confetti, and
fringe/island specks on the alpha edge. These are barely visible up close but
read as eye-catching "sparkle" when the sprite is zoomed out in-game.

## Requirements

Python 3.9+ with `numpy`, `scipy`, `Pillow` (and `tkinter`, bundled with most
Python installs, for the GUI):

```bash
pip install numpy scipy pillow
```

On Windows you can also double-click **Denoise.vbs** to launch the GUI, or run
`make_shortcut.vbs` once to drop a desktop icon (edit the Python path inside if
you don't use Anaconda).

## How it decides (the short version)

Everything is judged **relative to each pixel's local neighbourhood in CIELAB**,
so the same settings work on a green tree, a red rock, or a blue banner.

Detection is the **OR of independent channels**, each targeting one kind of noise.
The GUI exposes one high-level 0–10 "amount" slider per channel; turn it up to
catch more, set it to 0 to switch the channel off.

1. **Isolation** (`Overall strength`) — lone/paired specks of any colour, via
   `second_min` = ΔE to the *2nd-nearest* neighbour. A legit detail (leaf edge,
   highlight, vein) runs in a streak → ≥2 similar neighbours → small `second_min`;
   a speck is lone/paired → its 2nd-nearest is far → large `second_min`. Far more
   selective than raw local contrast, which fires on every legit edge.
2. **Bright / dark confetti** — blobs much **brighter or darker** than their
   neighbours, within a chroma ceiling, restricted to **small scattered clusters**
   so the big legit sunlit canopy / shadow mass is never touched. Catches specks
   the isolation channel misses when they clump (a 2-3px blob has a similar
   neighbour, so its `second_min` stays low). Turning the slider up lowers the
   brightness bar, **widens the chroma ceiling** (so pale-coloured brights get
   caught, not just near-grey/white), and engages dark-speck detection.
3. **Coloured confetti** — more-saturated, somewhat-isolated, small scattered
   clusters (alien magenta/cyan specks on otherwise-uniform art).
4. **Edge & fringe** — near the alpha edge (`edge_frac` wide) a lower isolation
   bar applies, plus any tiny disconnected opaque islands.

Flagged pixels are **removed** (alpha→0) if they sit on the fringe / are islands,
or **replaced** with the vector-median of their *clean* neighbours if interior.

## Usage

```bash
# Interactive: drag sliders, watch original | cleaned | detection update live
python pixel_denoise.py gui [image.png]

# Batch-clean a folder of PNGs
python pixel_denoise.py clean path/to/sprites/*.png -o cleaned --overlay

# Debug overlay (detections painted bright purple)
python pixel_denoise.py overlay sprite.png -o overlay.png

# Measure precision/recall against a hand-flagged image (purple = 184,0,255)
python pixel_denoise.py calibrate examples/original.png examples/flagged.png
```

In the GUI: the four **Quick controls** (0–20) are the everyday knobs; tick
**Show advanced** to fine-tune the raw thresholds (they track the macros and
override them once you drag one). Cyan = pixels removed, magenta = pixels
replaced. Save-cleaned / Apply-to-folder use the current settings; "Print params"
dumps both the macro values and the resolved thresholds to the console.

**Macro scale (0–20):** `0` turns a channel off; `10` matches the strongest pass
the old 0–10 slider could do; `10–20` is extra-strength headroom beyond that.

**Persistence:** slider positions (and the last image) are saved to
`denoise_settings.json` and restored next launch. **Restore defaults** resets all
sliders to the calibrated defaults.

## Tuning

Start with the Quick controls:

- **Bright / dark / pale specks left over** → turn up **Bright / dark confetti**
  (turning it past ~10 also catches pale-coloured and dark specks).
- **Coloured specks left over** → turn up **Coloured confetti**.
- **Generally not catching enough** → turn up **Overall strength** (best for
  isolated specks; some clumped same-colour pairs need the bright channel instead).
- **Ragged/noisy silhouette edge** → turn up **Edge & fringe cleanup**.
- **Eating real detail** → turn the offending channel *down*.

Advanced (under *Show advanced*): each channel's raw thresholds, plus
`fringe_thresh` (transparent-fraction cutoff above which a flagged pixel is
deleted rather than infilled), `alpha_thresh`, and neighbourhood `radius`.

Macro defaults (strength 6, white 7, colour 7, edge 8) reproduce the thresholds
calibrated against a hand-flagged tree (recall ~0.95). They favour recall per
project preference; lower **Overall strength** for a more conservative pass.
