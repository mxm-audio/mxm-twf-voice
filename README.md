# mxm-twf-voice

The TWF voice: **Targeted Wavetable Filtering**, the synthesized voice of
[MXM Model Drums](https://github.com/mxm-audio/mxm-model-drums), shared with
[mxm-twf-synth](https://github.com/mxm-audio/mxm-twf-synth), whose keyboard instruments (struck and
plucked: a marimba, a bass, a guitar, a piano) play the same voice.

A sound is built from three things:

- **groups of overtones** that live and die together, each group one looping wavetable of set
  overtones under one envelope, a glide and a moving low-pass;
- **a ringing noise**, third-octave bands whose gains follow a compact law;
- **a contact**, the strike's thump and tick.

A cymbal's ring is a fourth: bands of partials over a noise floor (`Metal`). Every part is a voice
of its own, off and costing nothing at level 0. What it holds is numbers a person may edit: each
overtone's ratio, level and phase, each group's envelope a law. Never a recording's per-partial
envelopes.

The technique is described in Model Drums' `docs/drum-synthesis.md`.

## Use

```toml
[dependencies]
mxm-twf-voice = { git = "https://github.com/mxm-audio/mxm-twf-voice", branch = "main" }
```

```rust
use mxm_twf_voice::{SynthKick, SynthKickSpec};

let mut voice = SynthKick::new(SynthKickSpec::default(), 48_000.0);
voice.strike(1.0);
let mut out = vec![0.0f32; 48_000];
voice.render(&mut out);
```

The names are the voice's own from Model Drums (`SynthKick`, `SynthKickSpec`, `Controls`): it
began as a synthesized kick.

## Licence

GPL-3.0-or-later (`LICENSE`). The MXM name and logo are not covered by it: see `TRADEMARKS.md`.
Contributions go through the CLA: see `CONTRIBUTING.md`.
