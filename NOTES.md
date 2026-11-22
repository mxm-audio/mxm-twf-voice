# NOTES.md — mxm-twf-voice

The reasoning behind [AGENTS.md](AGENTS.md).

## Why a library of its own (2026-11-09)

ops' `plans/plan-mxm-twf.md` D2 and ruling 10. The keyboard synth is a new synth of its own (the
owner: "This should ba new synths"), `mxm-twf-synth` ("mxm-twf-synth is the name"), and "It is ok
to make some crates for shared code if there is any." The drums' TWF voice is that code. Its name,
`mxm-twf-voice`, and its home, a repository of its own rather than mxm-kit (MIT), are the owner's
choices, asked for; it went public with the drums on 2026-11-22, once they were out to the beta
testers (the owner: "If they are out in beta, their repos should also be published"). "But the sharec code must not come at the cost of more CPU for
any consumer of the code": every change is measured in every consumer (AGENTS.md).

## How it was lifted

`crates/mxm-model-drums-dsp/src/synth_kick.rs` and its history (17 commits since the drums' own
repository began, 2026-10-06; before that it is in the archived monorepo) taken with `git
filter-branch` from a clone of mxm-model-drums at `81e43b3`, the file moved to `src/lib.rs`
unchanged. The drums keep `synth_kick` as a module that re-exports this crate, so none of their
code changed. Fingerprint before the lift, every patch the drums ship struck at 1, 0.5 and 0.15 and
once under moved controls, 2 s at 48 kHz each, FNV-1a over the samples' bits: the 57 synthesized
patches' combined `e77aabdc2a3dcbfb`, repeated run to run, and after it the same.

## The drums' voice alone (2026-11-10)

The keyboard's first additions came here, each off for the drums and checked against them as the
contract asked: `play_as`, `park_quiet`, then `set_pitch` and `set_lowpass`, every one a method the
drums never call, every check a fingerprint run and six alternated cost rounds. The owner: "If you
are spending a lot of time making sure you dont change the drums, perhaps make a split in the
code?", then, asked, "Synth gets its own copy", named `mxm-twf-synth-voice`. The keyboard's work
still ahead (hold, a damper's release, partners, larger tables) is inside the sample loop, where
the checks would bind on every change. mxm-twf-synth took this crate at `2be28a5` with
`set_pitch` and `set_lowpass` into a crate of its own; the three keyboard commits here are reverted,
so the code is again what the drums lock (`e1389ae`), and the two no longer follow each other.
