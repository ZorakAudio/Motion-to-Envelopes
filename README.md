# M2E — Motion to Envelopes

**M2E** extracts motion from video and converts it into **time-aligned control envelopes**.

Those envelopes can be exported as:
- **MIDI CC** (DAW-agnostic, real-time capable)
- **CSV** (high-resolution automation for DAWs or custom tooling)

M2E is not about notes, beats, or musical sequencing.  
It is about **motion → envelopes → sound**.

If something moves on screen, M2E lets that motion *cause* sound — not just decorate it.

---

## What problem M2E solves

Sound design for animation usually means:
- scrubbing picture
- guessing where motion “feels” intense
- hand-drawing automation
- fixing drift and over-reaction later

M2E replaces that with **measured motion envelopes** derived directly from the video:
- speed
- acceleration
- jerk (sudden change)
- directionality
- entropy / texture
- impacts and state transitions

These envelopes are **time-accurate to picture** and can drive:
- foley intensity
- granular density
- transient layers
- reverb and space
- modulation depth
- spatial motion

M2E does not decide *what* the sound should be.  
It gives you envelopes that let motion decide *when and how much*.

---

## What M2E is (and is not)

**M2E is:**
- a motion analysis tool
- an envelope generator
- a bridge between animation and sound
- conservative and confidence-aware

**M2E is not:**
- automatic sound design
- a note-based MIDI sequencer
- a “magic” optical-flow black box
- a hallucination engine

When motion is ambiguous, M2E prefers restraint over invention.

---

## Core idea (important)

M2E exports **state**, not events.

- MIDI CC is just a **carrier**
- CSV is just a **carrier**
- the invariant is the **envelope over time**

This is why M2E works across:
- DAWs
- engines
- workflows
- future formats you haven’t invented yet

---

## Output formats

### MIDI CC (default)
- DAW-agnostic
- Real-time capable
- Scrub-friendly (seek-safe mode)
- Portable

> Note: MIDI contains **CC only** (no notes).

### CSV
- Full-resolution envelopes
- Ideal for:
  - REAPER automation
  - analysis
  - custom pipelines
  - research

---

## Timing model (critical)

M2E uses a **wall-clock MIDI mapping**:

- **Tempo = 60 BPM**
- **1 beat = 1 second**
- **PPQ = 960**

This ensures:
- envelopes align to video time
- no drift
- no musical assumptions

If your DAW asks to import a tempo map:
- **Decline**, unless you explicitly want the project set to 60 BPM.

The tempo map only defines how MIDI ticks map to seconds.  
It does **not** affect audio or video unless you tell the DAW to use it.

---

## When to use M2E

Use M2E when:
- motion matters
- timing matters
- you want sound to feel *caused* by movement
- you want repeatable, inspectable behavior

Do not use M2E when:
- motion is irrelevant
- everything should be purely musical
- hand-drawn automation is the goal

---

## Learn more (recommended)

- **Quickstart** — see below for a 5-minute workflow
- **TECHNICAL.md** — deep dive into:
  - camera motion
  - occlusion
  - smooth shading failures
  - IG-LoG / structural confidence
  - HG vs Farnebäck modes
  - temporal shaping
  - psychoacoustics and salience

If you want to understand *why* something behaves the way it does, start there.

---

## Philosophy (one sentence)

M2E turns motion into envelopes, not decisions — so your sound feels *caused*, not decorated.


## Quickstart (ground-truthed from `Local Motion Tracker.py`)

Run:

```bash
python "Local Motion Tracker.py" path/to/video.mp4
````

Then press **F1** for the in-app hotkey list (Info overlay was removed; F1 is now the entry point).

---

### Basic navigation

* **Space / Enter**: play / pause
* **Left / Right**: step ±1 frame
* **Up / Down**: step ±10 frames
* **Shift+Left / Shift+Right**: ±10 frames (as shown in help)

Scrubbing:

* **Ctrl + Mouse Wheel**: fast scrub (big jumps)
* **Shift + Mouse Wheel**: scene jump (next/prev scene)

Fullscreen:

* **F11**: fullscreen toggle

---

### Scenes (segment your video)

Scenes are the unit of export. You typically make one scene per “motion beat.”

* **N**: new scene (blocked if it would overlap an existing scene; selects existing instead)
* **A**: set scene start to current frame
* **D**: set scene end to current frame (also used as end)
* **E**: end scene (sets end; if samples exist it exports immediately; otherwise it marks export pending)
* **Shift+E**: reopen (clear) scene end
* **X**: split current scene (duplicates over ROI)

---

### ROIs (what you track locally)

* **U**: add ROI (enters “Add ROI” mode)
* **R**: repick ROI geometry (enters “Repick ROI” mode)
* **[ / ]**: select previous/next ROI
* **Double-click ROI**: rename
* **Shift+Backspace**: delete active ROI (intentional hard-delete)

Mouse editing modifiers (geometry helpers):

* **Shift + Drag**: draw/adjust ROI **bound** (secondary constraint box)
* **Ctrl + Drag**: set ROI **anchor** (stabilizes tracking)

---

### Export

* **P**: export current scene
* **S**: export **all scenes** (auto-ends any open sampled scenes first)
* **Esc**: quit (also auto-ends & exports open scenes that have samples)

Export writes:

* `*.csv`
* `*.mid`
* `*.midi_map.json`

---

## “Modes” you’ll actually touch (and what they do)

### 1) Tracking mode (FB vs HG vs HYBRID)

* **Ctrl + J**: cycle **active ROI** motion mode: `fb → hg → hybrid`

High-level:

* **fb**: dense Farnebäck (great when structure is rich)
* **hg**: structure-assisted fallback (better when smooth shading / low texture / fast motion hurts FB)
* **hybrid**: tries to get the best of both

---

### 2) Camera-motion compensation (CMAT profiles)

* **M**: cycle CMAT profile (applies to active ROI via `r.cmat_mode = global/off`)

Profiles (as implemented):

* **off** → disables camera compensation (`mode=off`)
* **moderate** → global camera compensation, medium history cap (`max_h=180`)
* **chaotic** → global camera compensation tuned for shake (`max_h=120`, also forces `tau_ms=0`, `alpha=1.0`)
* **smooth** → global camera compensation, longer history (`max_h=240`)

Projection:

* **Shift+M**: cycle CMAT projection: `full ↔ orth`

---

### 3) Impact detection mode (per ROI)

* **I**: cycle ROI impact mode: `hybrid → fast → smooth`

Practical meaning:

* **fast**: more eager/peaky (more “events”)
* **smooth**: more conservative/stable (fewer spikes)
* **hybrid**: default blend behavior

If you get too many hits: go **smooth**. If you miss hits: try **fast**.

---

### 4) Axis-of-Interest (AoI) and axis modes

Axis logic exists as a selectable mode:

* **T**: cycle `axis_mode = off → cos → cone`

(Your AoI wheel behavior is handled by the wheel filter; Ctrl+Wheel is reserved for scrub and Shift+Wheel is reserved for scene jump, per the help.)

---

### 5) Z / 2.5D-ish options

* **Y**: cycle `vz_mode = curv → div → hybrid` (active ROI)
* **Z**: toggle `z_scale_mode = off ↔ vz` (active ROI)

---

## Preflight: “is my clip even compatible?”

Run the standalone IG-LoG visualizer:

```bash
python tools/iglog_standalone.py path/to/video.mp4 --debug 3
```

If the HYBRID structure view is weak/flickery in your intended ROI, optical flow will struggle. (This is the biggest “why is it failing?” root cause.)

---

## FAQ (short, non-handwavy)

### “Why don’t I see anything when I import the MIDI?”

The MIDI is **CC only** (no notes). You must view CC lanes or map CCs to parameters.

### “Why doesn’t the CC jump to the correct value when I scrub?”

Many DAWs don’t chase CC state on seek. Use the seek-safe export mode (what you’ve implemented) or convert CC→automation in your DAW.

### “Why does camera shake ruin everything?”

Because optical flow measures apparent pixel motion; shake moves the whole frame. Use **M** to enable a CMAT profile and/or use a more robust motion mode (`Ctrl+J`).

---

## More detail

For the full “why” (camera motion, occlusion, smooth shading, IG-LoG/SCM, temporal shaping, psychoacoustics), read:

* **TECHNICAL.md**

