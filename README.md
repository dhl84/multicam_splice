# multicam_splice

Turn several camera angles of **one event** into a fun, flick-between-angles
**FCPXML** you import into Final Cut Pro — fully local, no cloud.

Given clips from multiple cameras (different makes/mics/start times) it:

1. **Probes** every file (`ffprobe`) — duration, frame rate, geometry, embedded
   timecode, audio.
2. **Syncs** the angles onto one reference timeline by **audio
   cross-correlation** of spectral-flux onset functions, scored by
   peak-to-sidelobe ratio (PSR). A high PSR is a confident lock; a low one means
   the clips don't share audio (e.g. they were filmed at non-overlapping times)
   and you should place that angle manually.
3. **Splices** a remix: the spine flicks between angles with deliberately uneven
   shot lengths, keeping the whole covered span — nothing is dropped.
4. **Keeps the audio on one clean source.** Every spliced clip's own audio is
   force-muted (`srcEnable=video` **and** `adjust-volume -96dB`, the latter
   honoured by every FCP version), and a single continuous **audio bed** from the
   cleanest source plays underneath. Where that source doesn't cover a section
   (e.g. a warm-up only one camera filmed), the next-cleanest covering source is
   used there.
5. **Polishes**: cross-dissolves between sections and a fade-to-black (picture +
   sound) at the end.
6. **Outputs** a DTD-validated FCPXML (`File ▸ Import ▸ XML` makes a new project;
   your media and existing projects are untouched).

Independent: only `numpy` plus `ffmpeg`/`ffprobe` on PATH. It does **not** depend
on any other tool.

## Install / run

First copy a config and point `media_dir` at your footage (the example ships
with a placeholder path), then:

```sh
# with uv (no install needed):
uv run --with numpy python -m multicam examples/open_261.json

# or install it:
pip install -e .
python -m multicam path/to/config.json
```

Requires `ffmpeg` + `ffprobe` on PATH, and (for validation) Final Cut Pro's
bundled FCPXML DTD — validation is skipped gracefully if FCP isn't installed.

## Config

A project is one JSON file (see `examples/open_261.json`):

```jsonc
{
  "project_name": "Open 26.1 — warm-up + multicam remix",
  "event_name": "Open 26.1",
  "media_dir": "/path/to/footage",
  "output": "remix.fcpxml",                 // relative -> written into media_dir

  "format": {"width": 3840, "height": 2160, "fps_num": 60000, "fps_den": 1001},
  "reference": "GoPro",                       // timeline 0 = this angle's start

  "angles": {
    "GoPro":  {"files": ["GX010621.MP4"], "timecode": "08:35:20:49"},
    "iPhone": {"files": ["IMG_0709.MOV"]},               // auto-synced
    "DJI":    {"files": ["DJI_0475.MP4", "DJI_0476.MP4"]} // contiguous files
    // add "offset": <seconds> to any angle to skip auto-sync and place it manually
  },

  "segments": [
    {"type": "intro", "angle": "DJI", "audio": "own"},   // plays one angle straight
    {"type": "remix",
     "window": [66.0, 825.0],                 // reference-time span to remix
     "angles": ["GoPro", "iPhone"],
     "audio_bed": "auto",                     // "auto" = cleanest covering source
     "pattern": [3.5, 5.0, 2.5, 4.0, 6.0, 3.0, 4.5, 2.5, 5.5, 3.5],  // shot lengths (s)
     "tail_single_angle": true}               // calm single shot where one angle runs out
  ],

  "transition_seconds": 1.0,                  // cross dissolve between segments (0 = hard cut)
  "bed_fade_in_seconds": 1.0,
  "end_fade_seconds": 2.5                      // fade to black + audio at the end
}
```

### Segment types

- **`intro`** — one angle played straight through (all its files, contiguous),
  with its own audio. Use for footage that doesn't overlap the main event (a
  warm-up, a separate camera that can't be synced).
- **`remix`** — flick between `angles` across `window` (reference-time seconds)
  over one continuous audio bed. An angle is only shown where it actually covers
  that moment; elsewhere a covering angle is used.

### Audio

`audio_bed: "auto"` measures each remix angle (level + clipping via
`volumedetect`) and beds the cleanest one, falling back to the next-cleanest in
any span the first doesn't cover. Force a choice with `"audio_bed": "iPhone"`.

## How sync works

Each angle's files are concatenated and reduced to a **spectral-flux onset
function** (positive frame-to-frame magnitude change summed across bands) — sharp
shared events (impacts, claps, beeps) spike identically on every mic regardless
of tone or gain. The target's flux is cross-correlated against the reference's;
the peak gives the offset, and the **PSR** (peak vs the rest of the correlation)
tells you how much to trust it. The run prints the offset + PSR per angle and
flags low-confidence locks to verify or override with a manual `offset`.

Some mics never lock — e.g. a pocket gimbal whose audio is wind/handling-noise
dominated, or a camera filmed at a different time. Place those as an `intro`
segment or with a manual `offset`.

## Layout

- `multicam/probe.py` — ffprobe wrappers
- `multicam/sync.py` — spectral-flux onset + cross-correlation + PSR
- `multicam/audio.py` — cleanliness scoring + bed pick
- `multicam/timeline.py` — probe → sync → segment/shot layout → transitions/fades
- `multicam/fcpxml.py` — frame-accurate FCPXML emitter + DTD validation
- `multicam/config.py` — JSON config
- `multicam/__main__.py` — `python -m multicam <config.json>`
- `examples/open_261.json` — a worked 3-camera example (GoPro + iPhone synced,
  a DJI warm-up as an intro); set `media_dir` to your own footage to run it

## Notes / caveats

- **All local** — no network at runtime.
- FCPXML is validated against Apple's bundled DTD, but DTD-valid doesn't
  guarantee every semantic edge case — scrub transitions/fades once on import.
- Assumes the angles share one sequence format (raster + frame rate). Mixed
  codecs are fine; mixed frame sizes/rates are not handled.
- Sync decodes each clip's audio once; large files take a minute or two.
