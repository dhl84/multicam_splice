# Feedback for Claude: review and fix `multicam_splice`

This repo is a Python tool that builds a multicam FCPXML remix from footage shot on devices like iPhones, GoPros, and DJI Osmo Pocket cameras. The editing goals are:

- keep one clean continuous audio source under the edit
- auto-sync angles by audio where possible
- pick the best available shot for each scene
- avoid angles that are obscured, empty, static, or coach-only when better options exist
- emit valid, importable FCPXML for Final Cut Pro

Please review the current implementation and fix the issues below.

## Scope

Focus on correctness, efficiency, and edit quality. Do not do cosmetic refactors unless they directly support one of the issues below.

## Current repo status

- Static compile check passes: `python3 -m py_compile multicam/*.py make_contact_sheets.py`
- There are no automated tests in the repo right now

## Priority issues to fix

### 1) Frame/timebase correctness is not enforced

This is the highest-risk correctness issue.

The code currently probes source media and then assumes source frame counts can be interpreted using the project FPS. That is only safe if every input is already CFR and matches the project rate. In practice, iPhone footage is a common source of VFR or mismatched-rate media.

Likely affected areas:

- `multicam/probe.py`
- `multicam/timeline.py`

Examples:

- `multicam/probe.py` uses `r_frame_rate`
- `multicam/timeline.py` uses `inf.n_frames` and later converts based on project FPS
- coverage checks, file-boundary splitting, tail logic, and clip timing all depend on this

What to do:

- validate every input angle against the project format before timeline construction
- explicitly reject or clearly flag:
  - variable frame rate inputs if unsupported
  - mismatched FPS inputs
  - mismatched raster sizes if unsupported
- use the most correct probe fields available for playback timing, not just `r_frame_rate`
- make error messages actionable

Acceptance criteria:

- mixed/VFR footage does not silently produce drift
- unsupported media fails early with a clear error
- supported media produces consistent timing across sync, coverage, and emitted XML

### 2) `spectral_flux()` is too memory-heavy for long footage

This is the main efficiency issue.

Current behavior decodes full audio to memory, concatenates it, then builds a large frame-index matrix and full STFT-like buffer. This will scale badly for long recordings and multiple angles.

Likely affected file:

- `multicam/sync.py`

What to do:

- rewrite flux extraction to work in a streaming or chunked way
- avoid materializing one giant frame-index array
- avoid loading all audio for all files into RAM at once
- preserve current sync behavior and PSR-based confidence reporting

Acceptance criteria:

- long recordings use materially less RAM
- sync results remain stable relative to the current algorithm
- reference flux reuse still works for cut-on-action

### 3) Activity/content sampling is capped at 600 seconds and becomes stale later

This is a correctness issue for shot selection.

Current setup computes video activity and content classification from the start of each remix angle, capped at 600 seconds. Later windows beyond that region clamp to the last available sample, which means late sections of a long event can inherit stale “activity” or “athletes visible” values.

Likely affected areas:

- `multicam/timeline.py`
- `multicam/sync.py` (`video_activity`)
- `multicam/classify.py`

What to do:

- stop silently reusing the last sample for out-of-range windows
- either:
  - compute enough samples for the actual windows used, or
  - make out-of-range lookups neutral rather than stale
- ideally make analysis windowing depend on actual segment coverage instead of a hardcoded first-600-seconds cap

Acceptance criteria:

- shot selection late in long edits is based on valid data, not stale cached tail samples
- behavior is deterministic and documented

### 4) “Best shot” selection is currently rotation-with-filters, not actual ranking

Current logic rotates through the angle list, then filters out only obviously bad candidates. If multiple candidates pass, it tends to choose the next angle in rotation rather than the genuinely best-looking option.

This is important because the product goal says it should pick the “best” shot and avoid obscured angles.

Likely affected file:

- `multicam/timeline.py`

What to do:

- replace or augment the current pick logic with a scored ranking model
- keep some variety, but not at the expense of picking a clearly worse shot
- incorporate at least:
  - activity
  - content classification
  - coverage safety
  - a penalty for repeating the same angle too long if you still want variety

Nice to have:

- an extensible scoring function so future heuristics can be added cleanly

Acceptance criteria:

- when one angle is visibly better and another is merely “acceptable,” the better angle wins
- variety is preserved without overriding clearly superior coverage

### 5) Audio bed cleanliness is measured only from the first file of each angle

This is an effectiveness issue.

For split recordings, later files may have different noise, clipping, or level characteristics, but current ranking only measures `paths[0]`.

Likely affected file:

- `multicam/timeline.py`
- possibly `multicam/audio.py`

What to do:

- base bed selection on the actual material that covers the remix window
- do not assume the first file is representative
- preserve current “single continuous audio bed” behavior

Acceptance criteria:

- bed ranking reflects the real audio used in the remix span
- split files do not bias the choice incorrectly

### 6) Content-classification cache invalidation is too weak

Current cache filename is derived only from the first file stem. That makes stale cache reuse likely if any of these change:

- source files
- file ordering
- coach description
- model
- FPS
- max analyzed duration

Likely affected file:

- `multicam/classify.py`

What to do:

- strengthen the cache key
- include enough inputs to make stale reuse unlikely
- keep the cache local and deterministic

Acceptance criteria:

- meaningful input changes invalidate the cache automatically
- identical reruns still reuse cached results

### 7) Installed console script entry point looks incorrect

The package script target points at `multicam.__main__:main`, but `main()` currently expects an `argv` argument.

Likely affected files:

- `pyproject.toml`
- `multicam/__main__.py`

What to do:

- make the installed console script work correctly
- keep `python -m multicam <config.json>` working too

Acceptance criteria:

- both invocation forms work:
  - `python -m multicam <config.json>`
  - `multicam-splice <config.json>`

## Lower-priority issues

### 8) FCPXML asset audio metadata is too optimistic

Assets appear to be emitted as if they all have audio and stereo channels, regardless of probe results.

Likely affected file:

- `multicam/fcpxml.py`

What to do:

- emit asset audio metadata that matches actual probe data where practical

### 9) Add tests for the risky logic

There are currently no automated tests. Add focused tests around the logic most likely to regress.

Suggested coverage:

- timebase/format validation
- sync offset calculations
- cut-on-action snapping boundaries
- shot selection with competing angle quality inputs
- late-window behavior after analysis limits
- cache invalidation behavior
- CLI entry points

## Constraints

- Keep the implementation local/offline where it already is local/offline
- Do not remove the optional vision-classification workflow; improve its reliability
- Preserve the single-audio-bed design
- Preserve DTD-valid and importable FCPXML output

## Requested deliverables

Please provide:

1. A summary of the root causes you found
2. The code changes
3. Any tests added
4. Any behavior changes or tradeoffs
5. Any remaining limitations

## Strong suggestion on implementation order

Fix in this order:

1. frame/timebase validation
2. console script correctness
3. stale 600-second analysis behavior
4. spectral flux memory usage
5. shot ranking improvements
6. audio bed ranking across actual coverage
7. cache invalidation improvements

## Relevant files

- `README.md`
- `pyproject.toml`
- `multicam/__main__.py`
- `multicam/probe.py`
- `multicam/sync.py`
- `multicam/audio.py`
- `multicam/classify.py`
- `multicam/timeline.py`
- `multicam/fcpxml.py`
- `make_contact_sheets.py`

