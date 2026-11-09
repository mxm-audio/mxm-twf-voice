# AGENTS.md — mxm-twf-voice

DOX rail for this repository. The reasoning and the owner's rulings are in
[NOTES.md](NOTES.md). The workspace's `AGENTS.md` and mxm-kit's `docs/plugin-conventions.md`
hold the rules every MXM repository keeps.

# Purpose

The TWF voice (Targeted Wavetable Filtering), lifted out of MXM Model Drums' DSP crate with its
history on 2026-11-09, so that the drums and `mxm-twf-synth`'s keyboard instruments play one voice
(ops' `plans/plan-mxm-twf.md` D2 and ruling 10; its name and home the owner's). **Private**, a
repository of its own, GPL-3.0-or-later as the drums are. Framework-free: no GUI or plugin types,
one runtime dependency (the kit's `mxm-rng`).

The technique: Model Drums' `docs/drum-synthesis.md`. What a keyboard needs of it, each off for
every drum: the plan's §2.

# Ownership

- `src/lib.rs`: the voice, its sections (body, second oscillator, partials, click, noise bodies,
  the tables, the animated filter with a snare's wires, a cymbal's metal and its rattle),
  `SynthKickSpec`, the general `Controls` and `SynthKick`. Its names are the drums' (it began as a
  synthesized kick); its module documentation says what each section is and where its law comes
  from.
- Root: `Cargo.toml`, `Cargo.lock`, `LICENSE`, `NOTICE.md`, `TRADEMARKS.md`, `README.md`,
  `CONTRIBUTING.md`, `.github/`.

# Local Contracts

- **No consumer pays for another's features, in sound or in CPU** (the owner, 2026-11-09: "the
  shared code must not come at the cost of more CPU for any consumer of the code"). A feature
  added for one instrument (a keyboard's hold, release, partners or larger tables; a drum's wires
  or metal) is off for the others at its default, and then:
  - **every other consumer renders bit-identical**: Model Drums' fingerprint of every patch
    (*Verification*), before and after;
  - **costs every other consumer no more CPU**: each consumer's cost examples before and after
    (Model Drums' `model_drums_synth_kick_cost` and `model_drums_engine_cost`; mxm-twf-synth's
    once it has one), release, subnormals flushed, the median of three runs; a change that costs
    one more is not kept.
- **The mix is the sum of the sections rendered alone**; every section is off, and not run, at
  level 0.
- **`Controls::apply` is a pure function**; a strike at full velocity under neutral controls plays
  the patch exactly.
- **Realtime shape**: tables and metal are built off the audio thread and shared (`Arc`); a
  strike copies no patch onto the stack (`Controls::apply_to`, the voice's own working copies: a
  patch with metal is about 11 kB and a voice 55 kB, and a host's audio thread may be small);
  `strike` and `render` allocate nothing.
- **Samples are `f32`, slow falls `f64`** (the owner, 2026-10-25): filters, table reads, noise,
  sines and sums in single precision; the patch, the strike's arithmetic, phases and tables'
  places, and every slow fall (envelope stages, the metal's level curves, the rattle's sum) in
  double, since a fall's step a sample is too near 1 for single.
- **Speed work leaves every render bit-identical**: a sum is never split or reassociated.
- **Every constant is labelled** for what it is: *published* (cited), *derived* (derived in its
  doc comment) or *chosen*.

# Work Guidance

- A change for a keyboard instrument goes through ops' `plans/plan-mxm-twf.md` (§2) with the
  owner's word; a change for a drum through Model Drums' own plans.

# Verification

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

And for any change to the voice, in every consumer, before and after (built against this checkout
with `python ws.py link mxm-twf-voice`, undone before any push):

```bash
cargo test -p mxm-model-drums-dsp --release --test voice_fingerprint -- --ignored   # Windows' bits
cargo run -p mxm-model-drums-dsp --release --example model_drums_synth_kick_cost
cargo run -p mxm-model-drums-dsp --release --example model_drums_engine_cost
```

# Child DOX Index

No child AGENTS.md files.
