# NOTES.md — mxm-twf-voice

The reasoning behind [AGENTS.md](AGENTS.md).

## Why a library of its own (2026-11-09)

ops' `plans/plan-mxm-twf.md` D2 and ruling 10. The keyboard synth is a new synth of its own (the
owner: "This should ba new synths"), `mxm-twf-synth` ("mxm-twf-synth is the name"), and "It is ok
to make some crates for shared code if there is any." The drums' TWF voice is that code. Its name,
`mxm-twf-voice`, and its home, a private repository of its own rather than mxm-kit (public, MIT),
are the owner's choices, asked for. "But the sharec code must not come at the cost of more CPU for
any consumer of the code": every change is measured in every consumer (AGENTS.md).

## How it was lifted

`crates/mxm-model-drums-dsp/src/synth_kick.rs` and its history (17 commits since the drums' own
repository began, 2026-10-06; before that it is in the archived monorepo) taken with `git
filter-branch` from a clone of mxm-model-drums at `81e43b3`, the file moved to `src/lib.rs`
unchanged. The drums keep `synth_kick` as a module that re-exports this crate, so none of their
code changed. Fingerprint before the lift, every patch the drums ship struck at 1, 0.5 and 0.15 and
once under moved controls, 2 s at 48 kHz each, FNV-1a over the samples' bits: the 57 synthesized
patches' combined `e77aabdc2a3dcbfb`, repeated run to run, and after it the same.

## `play_as`, the keyboard's first addition (2026-11-09)

A strike reads everything from the patch but the tables' periods and their softer readings, which
depend on the overtones and the grid, not on the pitch. So a keyboard's 32 voices can play any
anchor at any key without building on the audio thread: each is a clone of an anchor's voice, and
`play_as` hands it another anchor's periods (shared `Arc`s) and the patch moved to the key. Checked
against the drums as the contract asks: all 57 patches' fingerprints identical before and after;
their cost examples alternated over six rounds, three in each order, the total within 0.3 % and
every case the same in each order (the one-slot case read 0.9 % first in a round and 1.0 %
second, whichever binary it was).

## `park_quiet`, the keyboard's second (2026-11-10)

A bar's strike lines die within a second, but each was worked out every sample until the whole
note parked, 100 dB under its own peak, 13 to 16 s for a bass bar: three fifths of mxm-twf-synth's
cost, 32 notes at 29.9 % of one core (the owner: "this is using extreme amounts of cpu").
`park_quiet(level)` stops each part under the level asked, by setting its level to 0, which every
section already reads as off; the keyboard's engine asks before each block, at -140 dB of full
scale over the note's gain: 32 notes 9.7 %, a note every 0.1 s 18.3 % against 38.3 %. `tick` is
unchanged and the drums never ask. Checked as the contract asks, against the voice the drums lock
(`e1389ae`): all 57 patches' fingerprints identical; their cost examples alternated over six
rounds, three in each order, the total -0.13 %, every case within its print's resolution (idle
read 0.3 or 0.4 in either binary, three of six at 0.4 before and four after).
