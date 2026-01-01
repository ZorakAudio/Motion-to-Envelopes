# PLAN — M2E (Motion to Envelopes)

This document outlines **near-term planned work** for M2E.  
These items are not speculative “nice ideas” — they are concrete extensions of the current architecture.

The guiding principle remains unchanged:

> **Motion produces envelopes.  
> Envelopes should be usable everywhere.**

---

## Immediate roadmap

### 1) OSC Server + OSC Client Receiver (JUCE VST3)

#### What this adds
An **OSC-based real-time transport layer** between M2E and audio software.

Instead of exporting envelopes to disk (MIDI / CSV), M2E will be able to:
- stream envelopes live via OSC
- send high-resolution, low-latency control data
- address parameters directly by name or path

A companion **OSC Receiver plugin (VST3, built with JUCE)** will receive and expose these signals inside any DAW that supports VST3.

---

#### Why this matters (vs current workflow)

Current workflow:
- analyze video
- export envelopes
- import into DAW
- route envelopes
- iterate offline

With OSC:
- analyze video → **sound responds immediately**
- no export/import loop
- no CC quantization
- no DAW-specific MIDI limitations
- true continuous control (not 7-bit CC unless desired)

This shifts M2E from:
> “offline envelope generator”

to:
> **live motion-driven control system**

---

#### What OSC enables technically

- Higher resolution than MIDI CC
- Named signals instead of numbered CCs
- Multiple channels without CC bookkeeping
- Real-time experimentation and auditioning
- Live tweaking of mappings while video plays

The OSC layer does **not** replace MIDI/CSV — it complements them:
- MIDI/CSV remain ideal for archival, automation, and DAW-native workflows
- OSC is ideal for exploration, performance, and rapid iteration

---

#### How users would use it

Example workflows:
- Play video → hear sound react immediately
- Adjust mapping depth while watching motion
- Drive:
  - synthesis parameters
  - granular engines
  - spatial processors
  - modulation depth
- Commit results later via MIDI/automation if desired

For non-DAW environments:
- drive game engines
- lighting systems
- haptics
- interactive installations

---

#### Why JUCE VST3 specifically
- Cross-platform (Windows/macOS/Linux)
- Works in most modern DAWs
- Allows clean parameter exposure
- Future-proof for CLAP or AU if desired later

This keeps the OSC path **DAW-agnostic**.

---

### 2) Granular Multi-Sampler (user-friendly procedural generator)

#### What this adds
A **purpose-built granular multi-sampler** designed specifically to be driven by M2E envelopes.

This is not “yet another granular synth.”  
It is a **motion-reactive sound generator** that makes envelope mapping immediately audible and useful.

---

#### Why this is needed

Right now, users must:
- bring their own sampler/synth
- manually design mappings
- understand granular synthesis deeply to get good results

This is powerful, but it raises the entry barrier.

The granular multi-sampler would:
- provide a **safe, expressive default sound engine**
- demonstrate best practices for envelope-driven sound
- let users hear results without building a full sound-design stack first

---

#### Core ideas

- **Multi-sample pool**
  - load folders of samples (foley, textures, noise, impacts)
  - sampler chooses grains intelligently

- **Envelope-driven behavior**
  - Speed → grain density / energy
  - Entropy → jitter / texture variation
  - Impacts → reseeding / transient emphasis
  - Direction → spectral or spatial bias

- **Provably smooth granular process**
  - Gaussian windows
  - overlap-add
  - no zipper noise
  - no clicks

- **User-friendly defaults**
  - minimal knobs
  - safe parameter ranges
  - impossible to “blow up” the mix accidentally

---

#### What it provides over existing tools

Compared to generic granular synths:
- no need to understand granular theory first
- mappings are motion-aware by design
- envelopes “just work” without deep setup
- tuned for *movement*, not musical pitch grids

Compared to static samplers:
- sound breathes with motion
- repeated animation loops don’t sound looped
- entropy prevents habituation
- impacts feel physical instead of triggered

---

#### How users would use it

Example workflows:
- Load samples → press play → motion drives sound immediately
- Use as:
  - a foley texture generator
  - a motion-reactive ambience layer
  - a procedural detail bed under traditional sound design
- Export or render once satisfied

Advanced users can:
- bypass it entirely
- or replace parts of it with their own tools

It is meant to **lower friction**, not constrain creativity.

---

## Relationship to current M2E

Neither planned feature invalidates existing workflows.

- MIDI CC export remains essential
- CSV export remains essential
- FluxBridge remains essential

These additions:
- **expand where envelopes can go**
- **reduce setup friction**
- **make motion immediately audible**

They turn M2E from:
> “a powerful but technical tool”

into:
> **a system that demonstrates its value instantly**

---

## Scope discipline

These features are planned because they:
- align directly with M2E’s envelope-centric philosophy
- reuse existing motion descriptors
- do not require changing core tracking logic
- do not force aesthetic decisions on users

Anything that:
- hallucinates motion
- hides uncertainty
- replaces envelopes with “magic behavior”

is explicitly out of scope.

---

## Final note

M2E’s strength is not novelty.  
It is **honesty + leverage**.

These planned additions aim to:
- broaden reach
- speed up iteration
- keep the core model intact

Everything still begins — and ends — with envelopes.
