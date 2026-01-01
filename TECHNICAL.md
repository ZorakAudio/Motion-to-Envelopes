## Preflight: Is my clip trackable?

Before you run full motion tracking, M2E includes a **preflight tool** to estimate whether your footage is *structurally compatible* with optical flow.

The tool is:

* `iglog_standalone.py`

It visualizes **trackable image structure** (edges, curvature, mass) using IG-LoG–based maps, so you can see **what the tracker will actually lock onto**.

If the structure is stable and “attached” to the motion you care about, tracking usually behaves well.
If it isn’t, parameter tweaking won’t fully fix it.

---

### Run the preflight tool

```bash
python tools/iglog_standalone.py path/to/video.mp4 --debug 3
```

---

## Technical notes: Camera motion and optical flow (why issues happen)

Optical flow does not measure “object motion.”
It measures **apparent pixel motion between frames**.

When the **camera moves**, every pixel moves—including the background—so the meaning of “motion” changes.

---

### What camera motion does to optical flow

Camera motion adds **global motion** across the frame:

* pans / tilts / zooms → coherent motion across large areas
* handheld shake → high-frequency jitter everywhere
* dolly motion → parallax (near and far move differently)

To optical flow, this often looks like:

* “everything is moving”
* global motion dominating local object motion

If untreated, camera motion can:

* drown out local motion (limbs, face details, etc.)
* bias flow vectors toward camera direction
* inflate speed/energy metrics everywhere
* create false “impacts” from shake, cuts, or drift

This is not a bug. It’s how optical flow works.

---

### How M2E handles camera motion

M2E separates **global motion** from **local motion** using two systems:

#### 1) Global Motion Tracker

Estimates frame-wide motion patterns:

* camera velocity
* camera acceleration
* directionality
* global entropy / shake

Use it when:

* camera motion is the thing you want to drive sound
* you want cinematic drift, shake, or movement “feel”
* local motion is weak or ambiguous

#### 2) Local Motion Tracker (ROI-based)

Tracks motion inside defined **regions of interest (ROIs)**.

To keep camera motion from overwhelming local flow, M2E uses:

* structural cues (IG-LoG / hybrid maps)
* confidence gating
* fallback logic when flow becomes unreliable

Local tracking works best when:

* the ROI contains structure that moves independently of the camera
* that structure stays visible and stable across frames

---

### Common failure modes involving camera motion

#### Global motion overwhelms local motion

**Symptoms**

* envelopes react strongly when the subject barely moves
* direction aligns with camera pans
* intensity spikes during cuts or shake

**Cause**

* camera motion magnitude > local motion magnitude

**Mitigation**

* use Global Motion intentionally
* reduce reliance on local metrics during heavy camera motion
* gate local motion using confidence/entropy

---

#### Parallax breaks coherence

**Symptoms**

* flow vectors diverge across depth
* envelopes fluctuate unpredictably
* structure “tears” across frames

**Cause**

* camera translation + depth variation (foreground/background move differently)

**Mitigation**

* tighten ROIs to depth-consistent regions
* avoid ROIs spanning near + far objects
* prefer local tracking over global in these cases

---

#### Camera shake creates false texture / impacts

**Symptoms**

* entropy rises everywhere
* impact detectors trigger without visible impacts
* jitter dominates envelopes

**Cause**

* high-frequency camera shake produces strong residuals

**Mitigation**

* detect shake via global motion metrics
* downweight or smooth local metrics during shake
* optionally treat shake as its own modulation source

---

### Why M2E doesn’t “just remove camera motion”

Perfect camera-motion removal generally requires:

* depth estimation, and/or
* scene reconstruction, and/or
* explicit camera metadata

Those are outside what optical flow alone can guarantee.

So M2E:

* exposes camera motion explicitly
* lets you decide how to use or ignore it
* avoids “silent correction” that can lie

Result: the system stays **predictable** and **inspectable**.

---

### Practical guidance

Use **Global Motion** when:

* camera movement is expressive and intentional
* you want cinematic modulation
* local motion is weak/noisy

Use **Local Motion** when:

* a specific object/body part matters
* camera motion is minimal or slow
* structure is clearly attached to the target motion

Use **both** when:

* camera motion sets macro feel
* local motion adds micro detail

---

### Key takeaway

Camera motion is not an edge case. It’s often the dominant signal.

M2E doesn’t hide it.
It gives you tools to **see it**, **separate it**, and **use it deliberately**—so envelopes feel intentional instead of mysterious.

---

## Technical notes: Occlusion (why motion “jumps” or disappears)

Occlusion happens when the thing you care about becomes **partially or fully hidden** by something else.

For optical flow, occlusion is brutal: the pixels you want to track **stop existing** for a while. When they return, there’s no guarantee they match the same surface.

---

### What occlusion does to optical flow

From flow’s point of view:

* pixels vanish (no correspondence)
* new pixels appear (ambiguous correspondence)
* edges swap “ownership” between objects
* motion vectors can reset or snap

Common results:

* sudden drops to near-zero motion
* spikes when structure reappears
* vectors snapping to the occluder
* envelopes that pop or flicker

Expected behavior in raw flow.

---

### Common occlusion scenarios in animation

* hair passing in front of face/body
* arms crossing the torso
* clothing folds covering limbs
* foreground objects near camera
* stylized motion blur hiding structure

Occluders often have **stronger edges** than the target surface, so flow tracks them unless you constrain it.

---

### How M2E handles occlusion

M2E does not assume continuous visibility. It combines:

* structural confidence (IG-LoG / hybrid maps)
* temporal consistency checks
* confidence gating + fallback logic
* optional user-defined occlusion masks

This allows M2E to:

* downweight motion when structure disappears
* avoid inventing motion without evidence
* resume when structure becomes reliable again

M2E does **not** hallucinate motion through occlusion. When the signal disappears, the envelope may:

* flatten
* decay
* enter a fallback mode
* briefly reflect the occluder

Intentional behavior.

---

### Failure modes caused by occlusion

#### Occluder hijacks motion

**Symptoms**

* envelopes follow hair/arm instead of target
* direction flips unexpectedly
* intensity increases during occlusion

**Cause**

* occluder has stronger structure; flow reassigns ownership

**Mitigation**

* tighten ROIs to exclude occluders
* use occlusion masks when available
* rely on global motion during heavy occlusion

---

#### Motion collapses to zero

**Symptoms**

* abrupt envelope drop
* motion “disappears” while animation continues

**Cause**

* target is fully hidden or blurred; no reliable structure remains

**Mitigation**

* treat as valid absence of signal
* apply downstream smoothing/decay
* combine with entropy/global motion for continuity

---

#### Reappearance spikes

**Symptoms**

* sharp spikes when target returns
* false impact detection

**Cause**

* structure returns with large pixel displacement

**Mitigation**

* gate impact detection with confidence
* smooth/clip extreme deltas
* interpret as visibility changes, not physical impacts

---

### Why M2E doesn’t “fix” occlusion

True occlusion handling requires:

* depth estimation
* object identity tracking
* surface continuity models

Optical flow alone can’t guarantee these. M2E chooses:

* no hallucination
* explicit uncertainty
* user-controlled smoothing/mapping

Result: envelopes stay **honest**.

---

### Practical guidance

* expect occlusion to destabilize envelopes
* prefer **local motion** when visibility is good
* fall back to **global motion** during heavy occlusion
* use `iglog_standalone.py` to spot problematic regions
* design mappings that tolerate brief signal loss

**Key takeaway:** optical flow can only track what it can see. When visibility changes, envelopes change too.

---

## Technical notes: Structural visibility & smooth shading (the silent failure mode)

Optical flow’s main requirement is **visible structure**.

Not motion. Not speed.
**Structure.**

If a region lacks stable, trackable structure, optical flow has nothing reliable to follow—even if the object is moving dramatically.

---

### What “structure” means here

Trackable structure includes:

* edges
* corners
* curvature
* persistent gradients
* stable contrast across frames

Structure does **not** mean:

* shading alone
* soft gradients
* uniform surfaces
* textureless skin/fabric

A surface can move a lot and still be **untrackable**.

---

### Why smooth shading breaks optical flow

Smooth shading tends to produce:

* large near-uniform regions
* low spatial gradients
* gradients that change with lighting more than motion

To flow, this creates ambiguous correspondence, so:

* vectors jitter or collapse
* envelopes fluctuate randomly
* motion reads weaker than it looks

Common in:

* stylized animation
* soft body parts
* glossy/subsurface surfaces
* large smooth anatomy with minimal edge detail

---

### Why this feels counterintuitive

Humans infer motion using context and object understanding.
Optical flow does not. It only sees **local pixel differences**.

---

### How M2E responds to low structure

M2E uses IG-LoG / hybrid structural maps to estimate **trustworthiness**.

When structure is weak:

* confidence drops
* metrics get gated/downweighted
* fallback logic may engage
* envelopes may flatten or decay

This avoids:

* hallucinated motion
* high-energy envelopes from noise
* locking onto random gradients

It can feel “underpowered,” but it’s more correct.

---

### Failure patterns from smooth shading

#### Motion fades or disappears

**Symptoms:** envelopes near zero
**Cause:** insufficient contrast/structure
**Mitigation:** include edges/folds/silhouette boundaries in ROI; rely more on global motion/entropy

#### Jitter and flicker

**Symptoms:** rapid fluctuation; direction flips
**Cause:** flow chasing noise
**Mitigation:** more downstream smoothing; lower speed/jerk sensitivity; favor aggregate measures

#### Structure “teleports”

**Symptoms:** spikes when edges enter/leave ROI; false impacts
**Cause:** abrupt visibility changes
**Mitigation:** gate impacts with confidence; treat as visibility events; use decay envelopes

---

### Using `iglog_standalone.py` to diagnose structure

Inspect the **HYBRID** structural map.

Good structure:

* clear continuous contours
* stable curvature attached to the target
* minimal flicker

Poor structure:

* mostly dark/uniform
* flickering/crawling response
* strong response from occluders instead of the target

If HYBRID looks weak/unstable, flow will struggle regardless of tuning.

---

### Key takeaway

Optical flow follows structure, not intent.

Weak structure → ambiguous motion → conservative envelopes.
M2E surfaces this so you can design around it instead of fighting invisible math.

---

## Technical notes: Entropy vs Speed vs Jerk (psychoacoustic use)

“Motional intensity” isn’t one thing. M2E exports multiple envelopes because different motion features map to different perceptual cues:

* **Speed** — how much motion is happening (macro energy)
* **Jerk** — how suddenly motion changes (events/impacts)
* **Entropy** — how alive/irregular motion remains over time (anti-habituation)

Treating them as interchangeable produces flat or fatiguing results.

---

### Speed (macro energy)

**Definition:** velocity magnitude

**Good sonic targets**

* overall intensity / “effort”
* foley layer density
* mild saturation amount
* careful spectral tilt / brightness
* small room/ER send changes

**Common failure**

* speed-only mappings get monotonous (high speed can still feel dead if nothing changes)

---

### Jerk (transients and meaningful change)

**Definition:** derivative of acceleration (change-of-change) → perceptual *surprise*

Spikes at:

* direction flips
* collisions
* abrupt stops/starts
* snap poses

**Good sonic targets**

* transient/impact layers
* one-shot triggering
* brief noise bursts (5–30 ms)
* transient shaper attack (bursty, not continuous)

**Common failure**

* over-driving jerk produces clicky, nervous mixes—especially when jitter/camera shake leaks in

---

### Entropy (aliveness / texture / anti-habituation)

**Definition:** structured irregularity/residual variation over time (often dominated by “what’s changing that isn’t just speed”)

**Good sonic targets**

* subtle noise bed level
* micro modulation depth (tiny)
* granular jitter (small depth)
* slight shimmer in tails
* stereo micro-wobble

**Common failure**

* entropy can be inflated by junk (shake, compression, occluders, low-structure jitter). If it rises everywhere, you’re measuring noise, not life.

---

## Using them together (recommended patterns)

### Pattern A: Energy + Events + Life

* Speed → continuous intensity (gain/density)
* Jerk → transient events (impacts)
* Entropy → residual texture floor

### Pattern B: Impact-safe wetness

* Entropy sets a low wetness floor
* Speed scales it upward
* Jerk sharpens briefly at contacts

### Pattern C: Camera-aware gating

When global shake is high:

* reduce local jerk triggers
* cap entropy depth
* keep only large, clear events

---

## Impacts: what to drive with what

Typical lanes (mode/config dependent):

* `impact_score01` — continuous event intensity
* `impact_in01` / `impact_out01` — directional state lanes
* `impact_*_spk01` — 1-frame spike lanes (true triggers)

**Best practice**

* use spike lanes for triggers
* use score for intensity scaling
* use in/out to bias which timbre family dominates

Speed is energy. Jerk/impact lanes are events.

---

## Psychoacoustic do/don’t rules

### Do

* speed → slow parameters (density, macro EQ tilt, gentle gain staging)
* jerk → short parameters (attacks, bursts, triggers)
* entropy → small continuous modulation (life/shimmer/micro-jitter)
* gate event behavior with confidence when structure is weak

### Don’t

* don’t map jerk to continuous loudness (pumps and fatigues)
* don’t map entropy to huge moves (seasick)
* don’t rely on speed alone for “aliveness”
* don’t treat camera-shake entropy as content entropy

---

## Quick mapping cheat-sheet (safe starts)

* **Speed:** gain 0–6 dB, mild density, small saturation drive
* **Jerk:** thresholded trigger probability, short bursts only, 5–30 ms filtered noise hits
* **Entropy:** subtle noise bed, tiny modulation depth, persistent wetness floor

**Key takeaway**

* Speed = macro energy
* Jerk = events/impacts
* Entropy = ongoing life

Use all three and the result stays energetic, punchy, and alive without turning into chaos.

---

# Motion representation internals

## IG-LoG, GSCM, HG vs Farnebäck (FB), and why multiple modes exist

M2E exposes multiple motion-analysis modes because **no single optical-flow representation survives every animation style**.

Different clips fail for different reasons:

* smooth shading
* occlusion
* camera motion
* stylization
* compression artifacts
* low texture

Everything below exists to answer one question:

> **Is this motion signal trustworthy?**

---

## First principles: optical flow needs structure

Optical flow depends on correspondence, which requires:

* gradients
* curvature
* edges
* persistent local contrast

When structure collapses, flow becomes ambiguous or hallucinatory.

---

## IG-LoG (Integrated Gaussian Laplacian of Gaussian)

### What it is

A multi-scale structural operator derived from LoG and integrated across scales for robustness.

Conceptually:

* Gaussian blur → reduces noise
* Laplacian → measures curvature (second derivative)
* integration → stabilizes response across scales

It highlights:

* ridges
* folds
* edges
* curvature transitions

These are the features optical flow can actually anchor to.

### Why it matters in M2E

IG-LoG is not motion. It is **structural confidence**.

M2E uses it to:

* detect trackable structure
* downweight flow when structure collapses
* distinguish real motion from noise-driven flow

Smooth shading fails because low curvature → weak IG-LoG → low confidence.

### Failure modes

IG-LoG can mislead when:

* lighting flicker changes curvature response
* occluders have stronger curvature than the target
* outlines dominate internal surfaces

So M2E uses IG-LoG as a **gate**, not a motion source.

---

## GSCM (Gaussian Structural Confidence Map)

### What it is

A derived confidence field built from IG-LoG and related cues.

It answers:

> “How structurally reliable is this region right now?”

It encodes:

* response magnitude
* spatial consistency
* temporal stability

### What it’s used for

GSCM drives:

* confidence gating of flow vectors
* fallback logic when flow degenerates
* entropy interpretation (life vs junk)
* impact validity checks

Rules of thumb:

* high flow + low GSCM = suspicious
* low flow + high GSCM = meaningful stillness
* changing GSCM = visibility change, not motion

---

## Farnebäck optical flow (FB)

### What it is

A dense polynomial-expansion flow method.

**Strengths**

* fast
* dense (per-pixel)
* smooth and stable on textured input

**Weaknesses**

* struggles with smooth shading, large displacement, occlusion, low texture, stylized animation

**Failure patterns**

* “melting” on smooth surfaces
* hallucinated motion from noise
* occluder lock-on
* underestimates fast motion
* direction flips near edges

M2E never assumes FB is correct by default.

---

## HG (Hybrid / Hierarchical / fallback family)

HG refers to M2E’s non-FB fallback/hybrid family, not a single algorithm.

HG prioritizes stability when FB becomes unreliable by:

* anchoring to structure
* aggregating motion over regions
* favoring consistency over density
* trading per-pixel detail for robustness

FB asks: “What’s moving where?”
HG asks: “Is something meaningful moving at all?”

---

### Tradeoffs: HG vs FB

| Aspect              | FB     | HG         |
| ------------------- | ------ | ---------- |
| Detail              | High   | Low–Medium |
| Stability           | Medium | High       |
| Smooth shading      | Poor   | Better     |
| Occlusion tolerance | Low    | Medium     |
| Event detection     | Weak   | Strong     |
| Perceptual feel     | Smooth | Punchy     |

---

## Hybrid modes (FB + HG + structure)

Most configs combine:

* FB where confidence is high
* HG fallback where confidence drops
* IG-LoG/GSCM to arbitrate

Goal:

* avoid full collapse when FB fails
* avoid overreaction to noise
* avoid losing motion during occlusion

Hybrid behavior is intentionally conservative.

---

## Why multiple modes are exposed

Different styles break different assumptions:

* anime smooth shading → FB fails
* outline-heavy styles → IG-LoG can overfire
* heavy camera motion → local flow lies
* hair/occlusion → HG needed
* compression artifacts → entropy polluted

No universal automatic choice exists, so M2E exposes modes for transparency and control.

---

## Key takeaway

M2E stays usable by separating:

* structure vs motion
* confidence vs magnitude
* continuous flow (FB) vs event-robust fallback (HG)

IG-LoG: what can be trusted
GSCM: how much it can be trusted
FB/HG: different manifestations of motion under different assumptions

---

## Temporal smoothing, decay, and why envelopes shouldn’t be symmetric

Temporal smoothing isn’t cosmetic. It decides **how motion becomes perception**.

A symmetric envelope (attack = release) looks neat, but it’s perceptually wrong for motion-driven sound.

---

## Motion is not time-symmetric

Real motion tends to:

* start abruptly (contacts/impulses)
* decay gradually (inertia/friction)
* rarely switch off cleanly

Symmetric smoothing tends to:

* delay attacks
* smear impacts
* make motion feel “pumped” or mechanical

---

## What smoothing is doing in M2E

After flow estimation + confidence gating + structural validation, smoothing shapes valid-but-noisy motion into usable envelopes.

It’s not “hiding bad data.” It’s making good data perceptually useful.

---

## Why symmetric smoothing fails

Symmetric filters (Gaussian blur, symmetric EMA, standard low-pass) often:

* delay onsets
* smear events forward/back in time
* erase sharp transitions
* flatten rhythm

Result: impacts lose punch; envelopes detach from picture.

---

## Asymmetric envelopes: the perceptual model

M2E favors:

* fast (or instant) attack
* slower, context-dependent decay

This matches:

* physical inertia
* auditory masking
* human onset sensitivity

The brain is far more sensitive to onsets than offsets.

---

## Decay should often be non-linear

Linear decay feels artificial. Natural systems decay exponentially/asymptotically.

Non-linear decay:

* keeps low-level motion alive
* avoids hard zeros
* reduces perceptual fatigue

---

## Entropy and decay

Entropy guides persistence:

* low entropy → faster decay feels fine
* high entropy → slower decay preserves “aliveness”

Entropy often modulates decay time or sets a wetness floor.

---

## Key takeaway

Perception is asymmetric. Motion is asymmetric.
Envelopes should be asymmetric too:

* fast attacks preserve causality
* slow, context-aware decay preserves life

---

## Impact detection internals & directional state machines

Impacts aren’t “high speed.” They’re **sudden, directional changes in motion state**.

Triggering on speed alone creates false positives (shake, occlusion spikes, loops).

---

## Core idea: impacts are changes in motion intent

Impacts correlate with:

* jerk (change in acceleration)
* directional reversal
* confidence-qualified spikes
* state transitions (not raw thresholds)

Impact logic runs on validated descriptors, not raw flow.

---

## Directional state machines

Instead of “is motion big enough?”, M2E asks:

> “Did motion transition between meaningful directional states?”

Example states:

* inward
* outward
* neutral/lateral
* stationary

Transitions that matter:

* outward → inward (contact/compression)
* inward → outward (release/rebound)
* moving → stationary (hard stop)
* stationary → moving (initiation/strike)

Noise flips and low-confidence wobbles should not trigger.

---

## Impact score vs impact spike

M2E separates intensity from timing:

* **impact_score (continuous):** how strong the transition is
* **impact_spike (discrete):** one-frame trigger emitted only on valid transitions with confidence + debounce

This prevents retrigger spam and machine-gun impacts.

---

## Key takeaway

Impacts are not peaks. They’re **state transitions**.

State-based detection (with confidence + context) produces impacts that feel intentional instead of mechanical.

---

## Design patterns: Mapping envelopes to sound (practical + psychoacoustic)

M2E gives envelopes. Your job: map them in a way that preserves causality, avoids habituation, avoids pumping, and stays mix-safe.

---

## The three-layer model

1. **Body (macro energy)** — Speed
2. **Contacts (events)** — Jerk / impact spikes
3. **Life (micro variation)** — Entropy

Use all three, each in its lane.

---

## Reliable mapping patterns

### Pattern A: Energy bus (Speed → intensity without pumping)

Targets: density, mild saturation, slight EQ tilt, small gain (0–6 dB).
Guardrail: don’t drive master gain; clamp and smooth.

### Pattern B: Impact punctuation (Spikes → triggers)

Spikes trigger one-shots/bursts; score scales intensity/selection.
Guardrail: don’t drive sustained parameters from spikes.

### Pattern C: Wetness persistence (Entropy → floor + micro life)

Entropy sets a non-zero floor and small modulation depth.
Guardrail: entropy is micro spice—keep depth tiny.

### Pattern D: Directional timbre switching (In/Out state)

Crossfade compressive vs release timbres.
Guardrail: crossfade > hard switch.

### Pattern E: Stereo embodiment (Direction X/Y)

Small pan/width/ER bias.
Guardrail: narrow range; move early cues more than tails.

### Pattern F: Macro/micro separation (Global vs Local)

Global → space/mood. Local → foley/contacts.
Guardrail: camera shake should not become local impacts.

### Pattern G: Confidence gating

Low confidence reduces modulation depth.
Guardrail: attenuate depth, don’t hard-mute everything.

---

## Anti-patterns

* speed → big gain swings (pumping)
* entropy → huge cutoff/pitch moves (seasick)
* camera shake → impact triggers (machine-gun hits)
* overfitting one clip (fails elsewhere)

---

## Recommended starter mapping

* Speed → foley density + mild saturation
* Impact spikes → short transient one-shots
* Impact score → intensity/sample weight
* Entropy → filtered noise bed + micro modulation
* Direction → small stereo bias + early reflections

---

## Final takeaway

M2E works when mappings align with perception:

* energy (speed)
* events (jerk/impacts)
* life (entropy)
* confidence-aware restraint
* asymmetric time behavior

That turns motion-driven sound from a gimmick into a performance.
