# multicam_splice

Turn several camera angles of **one event** into a fun, flick-between-angles
**FCPXML** you import into Final Cut Pro — fully local, no cloud required.

Given clips from multiple cameras (different makes/mics/start times) it:

1. **Probes** every file (`ffprobe`) — duration, frame rate, geometry, embedded
   timecode, audio — and **validates** each angle against the project format
   before building anything. The whole timeline is timed by frame count at the
   project rate, so variable-frame-rate media (common on iPhones), a
   mismatched frame rate, or a mismatched raster fails early with an actionable
   message (e.g. how to conform it) rather than silently drifting.
2. **Syncs** the angles onto one reference timeline by **audio
   cross-correlation** of spectral-flux onset functions, scored by
   peak-to-sidelobe ratio (PSR). A high PSR is a confident lock; a low one means
   the clips don't share audio (e.g. they were filmed at non-overlapping times)
   and you should place that angle manually.
3. **Splices** a remix: the spine flicks between angles with deliberately uneven
   shot lengths, keeping the whole covered span — nothing is dropped. With
   `cut_on_action` (default on) each cut snaps to the strongest nearby audio
   onset — a rep landing, a beep — so the edit cuts *on* the action; with
   `punch_in` longer shots get a subtle slow push-in for variety.
4. **Keeps the audio on one clean source.** Every spliced clip's own audio is
   force-muted (`srcEnable=video` **and** `adjust-volume -96dB`, the latter
   honoured by every FCP version), and a single continuous **audio bed** from the
   cleanest source plays underneath. Cleanliness is measured over the *actual
   remix window* — across every file an angle contributes there, not just its
   first — so a split recording whose later files clip or get noisy isn't beded
   on the strength of its opening file. Where the chosen source doesn't cover a
   section (e.g. a warm-up only one camera filmed), the next-cleanest covering
   source is used there.
5. **Picks the best-looking shot** by scored ranking (not plain rotation): the
   filters below gate out clearly-bad candidates, then the survivors are ranked
   by a weighted score (activity + athletes-visible content + media/coverage
   headroom) with a penalty that grows the longer one angle has been held — so a
   visibly better angle wins, and variety emerges on near-ties without ever
   overriding clearly superior coverage. The filters that run at setup time:
   - **Video activity** (always on): decodes each angle to a tiny 64×36
     greyscale raster at 2 fps and computes mean absolute frame-difference.
     Angles with near-zero activity (static or empty scene) are excluded from
     rotation unless every available angle is equally quiet.
   - **Vision content classification** (optional, requires `ANTHROPIC_API_KEY`):
     if `coach_description` is set in the config, thumbnail frames are sent to
     Claude Haiku and labelled A / C / E (athletes / coach-only / empty).
     Angles labelled coach-only are deprioritised so the edit stays on the
     action. Results are cached to disk next to the first file as
     `<first-file-stem>_content_<key>.npy`, where `<key>` digests every input
     that affects the labels (file names/sizes/mtimes, ordering, coach
     description, model, sample rate, analysed duration) so any meaningful
     change invalidates the cache and an identical rerun reuses it.
6. **Polishes**: cross-dissolves between sections and a fade-to-black (picture +
   sound) at the end. A centred dissolve needs half its length of spare media on
   *both* sides of the cut; where the incoming section starts at the very first
   frame of its media (no handle to dissolve from) the dissolve is shortened to
   what the footage allows, or replaced by a clean hard cut. This keeps every
   transition importable — FCP rejects a dissolve that reaches past a clip's
   media start with "Encountered an unexpected value."
7. **Outputs** a DTD-validated FCPXML (`File ▸ Import ▸ XML` makes a new project;
   your media and existing projects are untouched).

Core dependencies: `numpy` + `ffmpeg`/`ffprobe` on PATH. The vision
classification step additionally requires the `anthropic` package and a valid
`ANTHROPIC_API_KEY`; without it the content filter is silently skipped (activity
filtering still runs).

## Install / run

First copy a config and point `media_dir` at your footage (the example ships
with a placeholder path), then:

```sh
# with uv (no install needed):
uv run python -m multicam examples/open_261.json

# or install it (adds the `multicam-splice` console script):
pip install -e .
python -m multicam path/to/config.json
multicam-splice path/to/config.json     # same thing, installed entry point
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
  "end_fade_seconds": 2.5,                    // fade to black + audio at the end

  "cut_on_action": true,                      // snap remix cuts to nearby audio onsets
  "cut_snap_seconds": 0.7,                    // how far a cut may slide to reach one
  "punch_in": true,                           // subtle slow push-in on longer shots
  "punch_in_scale": 1.05,                     // how far the push travels (5%)
  "punch_in_min_seconds": 4.0,                // only shots at least this long

  // Optional: describe who to avoid cutting to when athletes are in another angle.
  // Requires ANTHROPIC_API_KEY. Results are cached; first run is slower.
  "coach_description": "olive green hoodie and dark blue/black tracksuit"
}
```

### Segment types

- **`intro`** — one angle played straight through (all its files, contiguous),
  with its own audio. Use for footage that doesn't overlap the main event (a
  warm-up, a separate camera that can't be synced).
- **`remix`** — flick between `angles` across `window` (reference-time seconds)
  over one continuous audio bed. An angle is only shown where it actually covers
  that moment; elsewhere a covering angle is used. A window stretch that *no*
  angle covers is rejected up front with the uncovered span named, rather than
  silently collapsing that time — narrow the window, add a covering angle, or
  split it into a separate segment.

### Audio

`audio_bed: "auto"` measures each remix angle with ffmpeg's `loudnorm` in EBU
R128 analysis mode — integrated loudness (LUFS), true-peak (dBTP), and loudness
range (LRA) — and beds the cleanest one (a healthy level with peak headroom and
stable dynamics), falling back to the next-cleanest in any span the first
doesn't cover. Cleanliness is measured over the material that actually covers
the remix window: for a split recording every file overlapping the window is
measured and combined by overlap duration (true-peak taken worst-case), so a
later file that clips or gets noisy can't be masked by a clean opening file.
Force a choice with `"audio_bed": "iPhone"`. (True-peak catches inter-sample
clipping that a raw 0-dBFS sample count misses.)

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

## How shot selection works

Within a remix segment the cut points come from the `pattern`'s deliberately
uneven step lengths. With `cut_on_action` on (the default), each cut point then
slides up to `cut_snap_seconds` to the strongest audio onset nearby — a rep
landing, a beep, a shout — so cuts land *on* the action the way an editor would
place them, instead of on a metronome. The onset function is the same spectral
flux already computed for sync, and a cut only snaps to a clearly strong event
(mean + 4σ of the flux) that keeps at least 1.5 s of shot on both sides;
otherwise it stays at its pattern position.

With `punch_in` on, every other shot longer than `punch_in_min_seconds` (and
the calm single-angle tail) gets a slow push-in from 100% to `punch_in_scale`
across the shot — gentle variety from the same camera positions, not an effect
you consciously notice. Shots that straddle a file boundary are skipped.

After the cut points are fixed, each shot's angle is chosen by gating the
covering angles through two filters and then **ranking** the survivors:

1. **Activity filter** (`_MIN_ACTIVITY = 2.0`): computes mean absolute
   frame-difference (64×36 px, 2 fps) per angle over the shot window. Angles
   below the floor are excluded; if *all* fall below (e.g. a rest between
   rounds) the filter is bypassed. An angle with no activity data for the window
   (out of analysed range) is treated as "no opinion" — never excluded on
   missing data, and never on a stale cached sample.

2. **Content filter** (when `coach_description` is set and classification data
   exists): requires that the majority of sampled frames (0.5 fps, 320×180 px)
   for an angle were labelled "A" (athletes visible) by the vision model. Angles
   whose frames are mostly coach-only or empty are excluded; if every candidate
   angle fails the content test the filter is bypassed. An unknown reading
   (no data, or a window past the analysed range) is kept, never dropped, and
   never reuses the last sample.

3. **Ranking**: the survivors are scored by normalised activity, athletes-visible
   fraction, and coverage headroom, minus a repeat penalty that grows with how
   long the current angle has already been on screen (saturating at
   `_REPEAT_FULL_S`). The penalty is capped below the activity weight, so it
   breaks near-ties for variety but never flips away from a visibly better shot.
   Missing activity/content reads score at the neutral midpoint of their range
   — they can't beat a clearly-better angle or be beaten by a clearly-weaker
   one purely on data they don't have.
   The scoring function is a single weighted sum, so new heuristics slot in as
   extra terms. Activity and content analysis windows are sized to the remix
   coverage each angle is actually used for (capped at its footage length), not
   a fixed first-600-seconds slice, so late sections of a long event rank on
   real data.

Classification results are saved next to the first media file as
`<stem>_content_<key>.npy` and loaded automatically on subsequent runs. The
`<key>` digests the inputs (files, ordering, coach description, model, sample
rate, analysed duration), so changing any of them produces a fresh cache rather
than reusing a stale one; to force a regenerate, delete the cache files and
re-run with `ANTHROPIC_API_KEY` set.

The helper script `make_contact_sheets.py` can pre-generate labelled contact
sheets (20 frames per JPEG) from any angle's files for manual inspection or
correction:

```sh
python make_contact_sheets.py /path/to/DJI_0479.MP4 /path/to/DJI_0480.MP4 \
    --out /tmp/sheets --prefix dji --fps 0.5 --max-seconds 500
```

## Layout

- `multicam/probe.py` — ffprobe wrappers
- `multicam/sync.py` — spectral-flux onset + cross-correlation + PSR; video activity
- `multicam/audio.py` — cleanliness scoring + bed pick
- `multicam/classify.py` — vision-based frame classification via Claude Haiku; cache I/O
- `multicam/timeline.py` — probe → sync → activity/content scoring → segment/shot layout → transitions/fades
- `multicam/fcpxml.py` — frame-accurate FCPXML emitter + DTD validation
- `multicam/config.py` — JSON config (`coach_description` field drives vision filter)
- `multicam/__main__.py` — entry point for `python -m multicam <config.json>`
  and the installed `multicam-splice <config.json>` console script
- `tests/` — unit tests (`python -m unittest discover -s tests`)
- `make_contact_sheets.py` — helper: extract labelled thumbnail contact sheets for cache inspection
- `examples/open_261.json` — a worked 3-camera example (GoPro + iPhone synced,
  a DJI warm-up as an intro); set `media_dir` to your own footage to run it

## Notes / caveats

- **Mostly local** — audio sync, activity scoring, and FCPXML generation all run
  offline. Vision classification calls the Anthropic API when a cache doesn't
  exist; set `ANTHROPIC_API_KEY` before first run if using `coach_description`.
- FCPXML is validated against Apple's bundled DTD, but DTD-valid doesn't
  guarantee every semantic edge case — scrub transitions/fades once on import.
- All angles must share one sequence format (raster + frame rate) and be
  constant-frame-rate at that rate. Mixed codecs are fine; mismatched
  rasters/rates and VFR media are rejected up front with an actionable error
  (conform them first). This is enforced, not assumed.
- Sync decodes each clip's audio once (streamed in chunks, so memory stays flat
  regardless of recording length); large files take a minute or two.
- Activity and classification analyse exactly the remix coverage each angle is
  used for (capped at its footage length), so late sections of long events rank
  on real samples — there is no fixed 600 s cap.
- Embedded timecode in media files is read and respected; if the first clip of
  an angle has a non-zero start timecode and none is specified in the config, it
  is read from the file automatically.
- **Frame-granular coverage.** Coverage checks and shot tiling operate at whole
  frames. Sub-frame gaps or overlaps at an angle hand-off are intentionally
  absorbed into the adjacent clip — they vanish under frame rounding at emit
  time, so the spine stays gap-free. A shot that spans a hand-off is split at
  the boundary and continued by another covering angle (no time is dropped).
- Asset audio metadata mirrors the probe: channel count and sample rate come
  from `ffprobe`, sources with no audio are marked `hasAudio="0"`, and when a
  field is genuinely unknown it is omitted rather than guessed (never a
  fabricated stereo/48 kHz).
