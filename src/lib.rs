//! A synthesized kick: separate sections — a body, a second oscillator, partials, a click and a noise
//! body — mixed, beside the physical kick of `drum.rs`.
//!
//! **Why it is here.** The instrument's goal is realistic drums; physical modelling is a tool, not
//! the goal (the owner, 2026-09-29: "The goal is realistic drums. Not physical modelled in itself.
//! That is just the tool"). This voice answers how close ordinary synthesis gets to a recorded kick
//! (`plans/plan-mxm-model-drums-synth-kick.md`), and it becomes a drum synth afterwards (the owner:
//! "having things separate is the most logical way to do that"). So every section is a voice of its
//! own: its own pitch, envelope, drive and velocity response, rendered alone or in the mix. The
//! sections share only the strike (its moment and velocity) and the recording chain's low cut on the
//! mix, so each is fitted against the part of the sound it makes, holding the others.
//!
//! **The sections, each exactly off at level 0 (not run, so it costs nothing and changes nothing):**
//! - *Body*: an oscillator, a sine shaped toward a triangle, whose pitch drops as a
//!   current-controlled oscillator's does, linear in hertz: `f = f_rest·(1 + k·e^(−t/τ))`
//!   (mxm-drum-machine's `docs/drum-model-fitting.md` §6; an exponential sweep in semitones does
//!   not fit a drum machine's kick). Its level rises linearly over its attack — nothing starts at
//!   its peak (§6) — and falls in two stages, a first with its own T60 and a share `tail` falling
//!   with a second.
//!   Its own drive, `tanh(d·x)/d`, after the strike's velocity, so a harder strike saturates more.
//! - *Second oscillator*: a sine with its own rest pitch, drop, attack and T60.
//! - *Partials*: up to thirty-two sines, the drum's lines, each with its level, its ratio to the body's
//!   rest pitch, its T60 (in one stage or two), its start phase and, if it builds up after the strike,
//!   its own attack; they share a drop and an attack. A drum's overtones are its head's
//!   modes, not harmonics (an ideal membrane's lie 1.59, 2.14, 2.30… times its fundamental), so they
//!   are sines at any ratio, not a waveform's; set as ratios, one tune moves the whole drum, as
//!   tightening a head does. The body's shape toward a triangle adds only odd harmonics.
//! - *Click*: white noise through a band-pass, falling with its own T60.
//! - *Noise*: white noise through its own band-pass, rising over its attack and falling with its T60.
//!   Either falls in two stages as the body does, a share `tail` with a second T60: a drum's ring
//!   dies fast after the strike and then hangs on (Virtuosity's kick, its ring near 560 Hz).
//! - *Noise 2*: a second band of noise with its own envelope. A drum's ringing is not one band: the
//!   Virtuosity kick's rings narrow near 560 Hz and broad over 240–680 Hz, with different lives.
//! - *Tables* (the owner's, 2026-09-30): wavetables of a drum's overtones, each overtone at a whole
//!   multiple of a common step so one period loops, played with a glide through a two-stage envelope
//!   and a moving low-pass — a subtractive synth's oscillator and filter, with a drum's overtones.
//! - *Animated filter* (the owner's, 2026-09-30: "an animated parametric filter"): white noise
//!   through third-octave bands whose levels follow a law — a broad band with its own spectrum, rise,
//!   two stages and a tilt that darkens it, and up to four rings, each a resonance whose centre moves
//!   (a drum's ring sinks as it dies) with its own rise and two stages. Fitted to what a recorded
//!   kick's lines leave, 32 numbers came closer than the same bands' measured curves (390 numbers):
//!   Virtuosity's hard hit, 45 audible differences against another hit, 49 for the curves.
//! - *Wires*, a snare's, in the animated filter's bands: a broad band of its own heard from the wires'
//!   first return on (see `Wires`).
//!
//! **Filters (published).** The band-passes and the low cut are the trapezoidal state-variable filter
//! (mxm-kit's `docs/filters/02-topologies.md` §2.2: Simper's linear trapezoidal SVF, Zavalishin's
//! TPT SVF, *The Art of VA Filter Design* ch. 4), prewarped with `g = tan(π·fc/fs)`; the band-pass
//! is its band output times `1/Q`, unity at its centre, and the low cut its high-pass output at
//! `Q = 1/√2`, the second-order Butterworth.
//!
//! **Velocity (chosen law).** Every section's level follows the strike's velocity `v`, and each of a
//! section's routes moves one of its parameters by `route` octaves per unit of velocity about
//! `v = ½`: `p(v) = p·2^(route·(v − ½))`. A section's own values are the middle strike's.

use std::f64::consts::{FRAC_1_SQRT_2, PI, TAU};
use std::sync::Arc;

use mxm_rng::{SplitMix64, Xorshift64Star};

/// A section below this, and the mix below it, parks the voice at exact zeros.
const PARK_LEVEL: f64 = 1e-7;
/// The velocity every route is centred on (**chosen**: a patch is the middle strike).
const ROUTE_CENTRE: f64 = 0.5;
/// Partials the voice holds: a recorded kick's lines run to twenty-odd (Virtuosity's hard hit, 22).
pub const PARTIALS: usize = 32;

/// `p(v) = p·2^(route·(v − ½))` (see *Velocity*).
fn routed(value: f64, route: f64, velocity: f64) -> f64 {
    value * 2f64.powf(route * (velocity - ROUTE_CENTRE))
}

/// The per-sample factor that takes a level 60 dB down in `t60` seconds.
fn fall(t60: f64, sample_rate: f64) -> f64 {
    10f64.powf(-3.0 / (t60.max(1e-4) * sample_rate))
}

/// The body: an oscillator with a pitch drop, a two-stage envelope and its own drive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Body {
    /// Level against the mix (0 off).
    pub level: f64,
    /// Rest pitch, drop depth `k` and time `τ`, shape (0 sine, 1 triangle), start phase (radians).
    pub hz: f64,
    pub drop: f64,
    pub drop_s: f64,
    pub shape: f64,
    pub phase: f64,
    /// Linear attack; the first stage's T60; the share falling with the second stage, its T60.
    pub attack_s: f64,
    pub decay_s: f64,
    pub tail: f64,
    pub tail_s: f64,
    /// Drive after the velocity: 0 off.
    pub drive: f64,
    /// Routes: the drop's depth and the decays (both stages).
    pub vel_drop: f64,
    pub vel_decay: f64,
}

/// A sine with its own rest pitch, drop, attack and T60: the second oscillator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tone {
    pub level: f64,
    pub hz: f64,
    pub drop: f64,
    pub drop_s: f64,
    pub attack_s: f64,
    pub t60_s: f64,
    /// Route: its level.
    pub vel_level: f64,
}

/// One partial, a line of the drum: level, rest frequency as a ratio of the body's rest pitch, T60;
/// the share `tail` falling with a second T60 `tail_s`; its start phase (radians, of a sine); its own
/// linear attack (0: the section's), over which its level holds before it falls — a line that builds
/// up after the strike, as a drum's ring does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Partial {
    pub level: f64,
    pub ratio: f64,
    pub t60_s: f64,
    pub tail: f64,
    pub tail_s: f64,
    pub phase: f64,
    pub attack_s: f64,
}

impl Default for Partial {
    /// Off: level 0, one stage, from phase 0 with the section's attack.
    fn default() -> Self {
        Self {
            level: 0.0,
            ratio: 1.59,
            t60_s: 0.4,
            tail: 0.0,
            tail_s: 1.0,
            phase: 0.0,
            attack_s: 0.0,
        }
    }
}

/// Partials sharing a drop and an attack.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Partials {
    pub partials: [Partial; PARTIALS],
    pub drop: f64,
    pub drop_s: f64,
    pub attack_s: f64,
    /// Route: their level.
    pub vel_level: f64,
}

/// Band-passed noise: a click (no attack) or a noise body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Noise {
    pub level: f64,
    pub hz: f64,
    pub q: f64,
    pub attack_s: f64,
    pub t60_s: f64,
    /// The share falling with the second stage's T60, `tail_s` (0: one stage).
    pub tail: f64,
    pub tail_s: f64,
    /// Routes: its level and its centre.
    pub vel_level: f64,
    pub vel_hz: f64,
}

/// Rings the animated filter holds.
pub const RINGS: usize = 4;
/// The law's level for "off", dB: a broad band or a ring at or below it is not there.
pub const BANK_OFF_DB: f64 = -200.0;
/// Bands the animated filter can hold: thirds of an octave from 20 Hz, thirty up to 20 kHz.
pub const MAX_BANDS: usize = 32;
/// The animated filter's gains are worked out from its law every this many samples through the strike,
/// eased between; after `BANK_STRIKE_S`, when the ring moves slowly, every `BANK_BLOCK_LATE` (a
/// quarter of its cost was the law, most of it spent on a ring that barely changes in 5 ms).
const BANK_BLOCK: u64 = 32;
const BANK_BLOCK_LATE: u64 = 256;
const BANK_STRIKE_S: f64 = 0.05;
/// A band whose law is this far below the strike's loudest band (−100 dB) is not run: late in the ring
/// most of the upper bands are. Against the strike, not full scale: a soft hit is quieter throughout,
/// yet brought up to level its quiet bands are heard (an absolute floor cost the soft layer 5
/// differences).
const BANK_SILENT: f64 = 1e-5;
/// Each band is two band-passes in cascade at this Q, which meet −3 dB a sixth of an octave either
/// side of the centre: a third-octave band (|H|² = 10^−0.15 per stage at r − 1/r = 0.2316).
const BANK_Q: f64 = 2.78;

/// The animated filter's broad band: its spectrum at the strike `level_db + slope_db·x + curve_db·x²`
/// (dB, `x` octaves from 500 Hz); a linear rise; two stages, as the body's; and a tilt of `tilt_db`
/// per octave it reaches with time constant `tilt_s` — a drum darkens as it rings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Broad {
    pub level_db: f64,
    pub slope_db: f64,
    pub curve_db: f64,
    pub rise_s: f64,
    pub t60_s: f64,
    pub tail: f64,
    pub tail_s: f64,
    pub tilt_db: f64,
    pub tilt_s: f64,
}

/// One of the animated filter's rings: a resonance of width `q` whose centre moves from `from_hz` to
/// `to_hz` with time constant `move_s` (a drum's ring sinks as it dies), rising linearly, falling in
/// two stages.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ring {
    pub level_db: f64,
    pub from_hz: f64,
    pub to_hz: f64,
    pub move_s: f64,
    pub q: f64,
    pub rise_s: f64,
    pub t60_s: f64,
    pub tail: f64,
    pub tail_s: f64,
}

/// The strike's own sound, the beater on the head: in each third-octave band a level (dB, the law's
/// units; `BANK_OFF_DB` none) and a T60, all rising together. A band's own fall, because a kick's
/// strike is a low thump and a high tick of different lengths (Virtuosity's hard hit: 25–50 Hz holds
/// to 35 ms, the tick above 5 kHz is over in 2) — one shape for all bands missed the thump by 9 dB.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contact {
    pub level_db: [f64; MAX_BANDS],
    pub t60_s: [f64; MAX_BANDS],
    pub rise_s: f64,
}

/// A snare's wires: a broad band of their own, `band`, its time counted from `delay_s`, nothing before.
///
/// The wires leave the snare-side head at the strike and slap it again when they return; the head
/// radiates the slaps, a band the strike does not have (`research:physical-modelling/snare-wires.md`
/// §2.3, §2.8; the owner, 2026-10-08: "not internal noise from the drum skin? That the wires cause?").
/// Virtuosity's snare, wires on against off at matched layers: 20–39 dB more above 4 kHz, from 10 ms
/// after a soft strike and 13–15 ms after a hard one, falling twice as fast after the soft one, while
/// the head's lowest mode loses 20 dB more by 45 ms. The band does not pulse with the head: its 4–12
/// kHz level folded over the head's cycle varies 1.4–2.3 dB with the wires and 2.3–3.0 without. So a
/// law carries it, and it costs nothing a sample beyond the bands it keeps sounding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wires {
    pub band: Broad,
    pub delay_s: f64,
}

/// The animated filter: white noise through third-octave bands whose levels follow a law, a broad
/// band, the strike's contact, the rings and a snare's wires, `level` (linear) on top of it, 0 off.
/// The contact is the strike's alone, so the ring's law is not spent on it (fitted from the strike
/// together, the rings went to +33 dB bursts dying in 3 ms instead of Virtuosity's ring near 560 Hz).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bank {
    pub level: f64,
    pub broad: Broad,
    pub contact: Contact,
    pub rings: [Ring; RINGS],
    pub wires: Wires,
    /// Route: its level.
    pub vel_level: f64,
}

impl Default for Bank {
    /// Off.
    fn default() -> Self {
        let off = Broad {
            level_db: BANK_OFF_DB,
            slope_db: 0.0,
            curve_db: 0.0,
            rise_s: 0.0,
            t60_s: 0.3,
            tail: 0.0,
            tail_s: 1.0,
            tilt_db: 0.0,
            tilt_s: 0.05,
        };
        Self {
            level: 0.0,
            broad: off,
            contact: Contact {
                level_db: [BANK_OFF_DB; MAX_BANDS],
                t60_s: [0.01; MAX_BANDS],
                rise_s: 0.0,
            },
            rings: [Ring {
                level_db: BANK_OFF_DB,
                from_hz: 500.0,
                to_hz: 500.0,
                move_s: 0.1,
                q: 10.0,
                rise_s: 0.0,
                t60_s: 0.3,
                tail: 0.0,
                tail_s: 1.0,
            }; RINGS],
            wires: Wires {
                band: off,
                delay_s: 0.01,
            },
            vel_level: 0.0,
        }
    }
}

/// A broad band's level at `x` octaves from 500 Hz, `t` s into its course, dB.
fn broad_db(b: &Broad, x: f64, t: f64) -> f64 {
    b.level_db
        + b.slope_db * x
        + b.curve_db * x * x
        + rise_db(t, b.rise_s)
        + stages_db(t, b.t60_s, b.tail, b.tail_s)
        + b.tilt_db * x * (1.0 - (-t / b.tilt_s.max(1e-4)).exp())
}

/// Two stages in power, dB: `(1 − a)·10^(−6t/T1) + a·10^(−6t/T2)`, each falling 60 dB in its T60.
fn stages_db(t: f64, t60: f64, tail: f64, tail_s: f64) -> f64 {
    let a = tail.clamp(0.0, 1.0);
    let power = (1.0 - a) * 10f64.powf(-6.0 * t / t60.max(1e-4))
        + a * 10f64.powf(-6.0 * t / tail_s.max(1e-4));
    10.0 * (power + 1e-30).log10()
}

/// A linear rise in amplitude over `rise_s`, dB.
fn rise_db(t: f64, rise_s: f64) -> f64 {
    if rise_s <= 0.0 {
        0.0
    } else {
        20.0 * (t / rise_s).clamp(1e-6, 1.0).log10()
    }
}

impl Bank {
    /// The law: the level of the band centred on `centre_hz`, dB of its RMS, `t` s after the strike.
    #[must_use]
    pub fn level_db(&self, centre_hz: f64, t: f64) -> f64 {
        let t = t.max(0.0);
        let x = (centre_hz / 500.0).log2();
        let mut power = 0.0;
        // The contact at this band: its nearest band's level and T60 (the bands are the law's own).
        let k = ((centre_hz / 20.0).log2() * 3.0)
            .floor()
            .clamp(0.0, (MAX_BANDS - 1) as f64) as usize;
        let c = &self.contact;
        if c.level_db[k] > BANK_OFF_DB {
            let db = c.level_db[k] + rise_db(t, c.rise_s) - 60.0 * t / c.t60_s[k].max(1e-4);
            power += 10f64.powf(db / 10.0);
        }
        if self.broad.level_db > BANK_OFF_DB {
            power += 10f64.powf(broad_db(&self.broad, x, t) / 10.0);
        }
        for r in &self.rings {
            if r.level_db <= BANK_OFF_DB {
                continue;
            }
            let centre = r.to_hz * (r.from_hz / r.to_hz).powf((-t / r.move_s.max(1e-4)).exp());
            let ratio = centre_hz / centre;
            let shape = -10.0 * (1.0 + r.q * r.q * (ratio - 1.0 / ratio).powi(2)).log10();
            let db =
                r.level_db + shape + rise_db(t, r.rise_s) + stages_db(t, r.t60_s, r.tail, r.tail_s);
            power += 10f64.powf(db / 10.0);
        }
        let w = &self.wires;
        if w.band.level_db > BANK_OFF_DB && t >= w.delay_s {
            power += 10f64.powf(broad_db(&w.band, x, t - w.delay_s) / 10.0);
        }
        if power > 0.0 {
            10.0 * power.log10()
        } else {
            BANK_OFF_DB
        }
    }

    fn rises(&self) -> f64 {
        let wires = if self.wires.band.level_db > BANK_OFF_DB {
            self.wires.delay_s + self.wires.band.rise_s
        } else {
            0.0
        };
        self.rings
            .iter()
            .map(|r| r.rise_s)
            .fold(self.broad.rise_s.max(self.contact.rise_s), f64::max)
            .max(wires)
    }
}

/// The animated filter's bands at a sample rate, (low, high) Hz: thirds of an octave from 20 Hz up to
/// 20 kHz or 0.45 of the rate, the listener's noise bands.
#[must_use]
pub fn bank_bands(sample_rate: f64) -> Vec<(f64, f64)> {
    let top = 20_000.0f64.min(0.45 * sample_rate);
    let step = 2f64.powf(1.0 / 3.0);
    let mut out = Vec::new();
    let mut lo = 20.0;
    while lo < top && out.len() < MAX_BANDS {
        let hi = (lo * step).min(top);
        if hi / lo > 1.05 {
            out.push((lo, hi));
        }
        lo *= step;
    }
    out
}

/// One band's filter: two band-passes in cascade, unity at the centre.
#[derive(Debug, Clone, Copy, Default)]
struct BandFilter {
    a: Svf,
    b: Svf,
}

impl BandFilter {
    fn new(centre_hz: f64, sample_rate: f64) -> Self {
        Self {
            a: Svf::new(centre_hz, BANK_Q, sample_rate),
            b: Svf::new(centre_hz, BANK_Q, sample_rate),
        }
    }

    fn tick(&mut self, x: f64) -> f64 {
        self.b.band(self.a.band(x))
    }

    fn clear(&mut self) {
        self.a.clear();
        self.b.clear();
    }
}

/// A recording read in the animated filter's terms: each band's level, dB of its RMS, around each of
/// `at` (samples), from half-way to the previous point to half-way to the next — what the law is
/// fitted to. Read through the band's filter twice, as the voice's band noise is heard through it once
/// more.
#[must_use]
pub fn bank_levels(x: &[f64], sample_rate: f64, at: &[usize]) -> Vec<Vec<f64>> {
    let n = x.len();
    bank_bands(sample_rate)
        .iter()
        .map(|&(lo, hi)| {
            let centre = (lo * hi).sqrt();
            let (mut f, mut g) = (
                BandFilter::new(centre, sample_rate),
                BandFilter::new(centre, sample_rate),
            );
            let y: Vec<f64> = x.iter().map(|&v| g.tick(f.tick(v))).collect();
            (0..at.len())
                .map(|k| {
                    let a = if k == 0 {
                        at[0]
                    } else {
                        (at[k - 1] + at[k]) / 2
                    };
                    let b = if k + 1 == at.len() {
                        at[k] + 1
                    } else {
                        (at[k] + at[k + 1]) / 2
                    };
                    let a = a.min(n);
                    let b = b.max(a + 1).min(n);
                    if b <= a {
                        return BANK_OFF_DB;
                    }
                    let power = y[a..b].iter().map(|v| v * v).sum::<f64>() / (b - a) as f64;
                    10.0 * (power + 1e-30).log10()
                })
                .collect()
        })
        .collect()
}

/// One band running: its own noise and filter, the level a gain of 1 reads, its gain eased toward
/// the law's. Each band has its own noise: bands filtering one noise summed near opposite phase where
/// they overlap (a two-stage band-pass turns from +180° to −180° across its band), cut notches
/// between the centres and played the law 12 dB low (measured, `the_animated_filter_plays_its_law`).
#[derive(Debug, Clone, Copy)]
struct BandRun {
    filter: BandFilter,
    white: Xorshift64Star,
    centre: f64,
    /// What a gain of 1 reads, as `bank_levels` reads, every band playing (see `BankRun::new`).
    unit: f64,
    gain: f64,
    step: f64,
    /// Run this block: its gain, or where it is going, above silence.
    active: bool,
}

impl Default for BandRun {
    fn default() -> Self {
        Self {
            filter: BandFilter::default(),
            white: white(1),
            centre: 0.0,
            unit: 0.0,
            gain: 0.0,
            step: 0.0,
            active: false,
        }
    }
}

/// The animated filter running.
#[derive(Debug, Clone, Copy)]
struct BankRun {
    bands: [BandRun; MAX_BANDS],
    count: usize,
    level: f64,
    /// The age, samples, the law is worked out again at.
    next_plan: u64,
    /// The strike's loudest band, as the law's first plan read it: what silence is measured against.
    loudest: f64,
}

impl BankRun {
    fn new(sample_rate: f64) -> Self {
        let mut bands = [BandRun::default(); MAX_BANDS];
        let layout = bank_bands(sample_rate);
        for (k, (band, &(lo, hi))) in bands.iter_mut().zip(&layout).enumerate() {
            let centre = (lo * hi).sqrt();
            *band = BandRun {
                filter: BandFilter::new(centre, sample_rate),
                white: white(0x00DD_BA11 + k as u64),
                centre,
                unit: 1.0,
                gain: 0.0,
                step: 0.0,
                active: false,
            };
        }
        // What a gain of 1 reads as `bank_levels` reads it — every band playing at 1 with its own
        // noise, the sum read through each band's filter twice — so a smooth spectrum plays at its
        // law: a band alone, it read 1.5 dB under its neighbours' leakage (measured,
        // `the_animated_filter_plays_its_law`). Half a second after 50 ms settling. **Measured**: the
        // cascades' overlap has no short closed form.
        let settle = (0.05 * sample_rate) as usize;
        let count = (0.5 * sample_rate) as usize;
        let mut sum = vec![0.0; settle + count];
        for band in &mut bands[..layout.len()] {
            for v in &mut sum {
                *v += band.filter.tick(band.white.bipolar_f64());
            }
            band.filter.clear();
        }
        for band in &mut bands[..layout.len()] {
            let mut f = BandFilter::new(band.centre, sample_rate);
            let mut g = BandFilter::new(band.centre, sample_rate);
            let mut power = 0.0;
            for (i, &v) in sum.iter().enumerate() {
                let y = g.tick(f.tick(v));
                if i >= settle {
                    power += y * y;
                }
            }
            band.unit = (power / count as f64).sqrt().max(1e-12);
        }
        Self {
            bands,
            count: layout.len(),
            level: 0.0,
            next_plan: 0,
            loudest: 0.0,
        }
    }

    fn start(&mut self, spec: &Bank, velocity: f64, seed: u64) {
        self.level = routed(spec.level, spec.vel_level, velocity) * velocity;
        self.next_plan = 0;
        self.loudest = 0.0;
        for (k, band) in self.bands[..self.count].iter_mut().enumerate() {
            band.white =
                white(seed.wrapping_add((k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            band.filter.clear();
            band.gain = 0.0;
            band.step = 0.0;
        }
    }

    /// The gains the law asks for `BANK_BLOCK` samples on from `t` s after the strike, eased toward.
    fn plan(&mut self, spec: &Bank, age: u64, sample_rate: f64) {
        let t = age as f64 / sample_rate;
        let span = if t < BANK_STRIKE_S {
            BANK_BLOCK
        } else {
            BANK_BLOCK_LATE
        };
        self.next_plan = age + span;
        let t_next = t + span as f64 / sample_rate;
        if age == 0 {
            self.loudest = self.bands[..self.count]
                .iter()
                .map(|b| 10f64.powf(spec.level_db(b.centre, t_next) / 20.0))
                .fold(0.0, f64::max);
        }
        let floor = BANK_SILENT * self.loudest;
        for band in &mut self.bands[..self.count] {
            let target = 10f64.powf(spec.level_db(band.centre, t_next) / 20.0) / band.unit;
            band.step = (target - band.gain) / span as f64;
            band.active = target.max(band.gain) * band.unit > floor;
            if !band.active {
                band.gain = target;
                band.step = 0.0;
            }
        }
    }

    fn tick(&mut self) -> f64 {
        let mut out = 0.0;
        for band in self.bands[..self.count].iter_mut().filter(|b| b.active) {
            let n = band.white.bipolar_f64();
            out += band.gain * band.filter.tick(n);
            band.gain += band.step;
        }
        self.level * out
    }

    fn loudness(&self) -> f64 {
        self.level
            * self.bands[..self.count]
                .iter()
                .map(|b| b.gain * b.unit)
                .fold(0.0, f64::max)
    }
}

/// Overtones a table holds.
pub const OVERTONES: usize = 8;
/// Tables the voice holds. The Ringing kick plays three — a body, a second skin and a long ring —
/// and leaves the fourth off; built from their recordings, the kick and both snares start closer in
/// four (`plans/plan-synth-snare.md`, revision 10). An off table has no period and costs a test a
/// sample.
pub const TABLES: usize = 4;
/// Samples in one period of a table, at every rate: at the finest step the highest overtone still
/// has tens of samples a cycle for the linear read.
const TABLE_LEN: usize = 1 << 17;

/// One of a table's overtones: its ratio to the table's pitch, its level (linear, 0 off) and the phase
/// it starts the period at (radians, of a sine). Overtones all at phase 0 rise together into one sharp
/// first swing (Virtuosity's hard hit: its first 3 ms 6 dB loud, its peak at 2.5 ms against 16.7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Overtone {
    pub ratio: f64,
    pub level: f64,
    pub phase: f64,
}

/// The owner's wavetable (2026-09-30: "a waveform/long sound for the drum, with set overtones. That
/// can then be filtered like a subtractive synth"). One period holds every overtone, each rounded to
/// a whole multiple of a common `step` of the pitch, so the period repeats and the phases meet again
/// (the owner: "1/2 + 1/4 = 2/4 + 1/4"); a step of 1/64 holds a 83 Hz drum's overtones within about
/// 10 cents in 0.77 s, 1/256 within a few in 3 s. Each overtone starts at its own phase; the attack
/// keeps a table that does not start at zero from clicking. Played at its pitch with a glide `f = hz·(1 + drop·e^(−t/τ))`, through a two-stage
/// envelope and a low-pass whose corner moves from `lp_from_hz` to `lp_to_hz` with time constant
/// `lp_s` — a drum's higher overtones die first. The overtones are editable; the table is built when
/// the voice is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Table {
    pub level: f64,
    pub hz: f64,
    pub overtones: [Overtone; OVERTONES],
    pub step: f64,
    pub drop: f64,
    pub drop_s: f64,
    /// A second, fast stage of the glide: the beater's instant of high tension, a kick's chirp
    /// (Virtuosity's hard hit: its first cycle about 130 Hz, 7.5 ms, then 88 Hz; 0 off).
    pub drop2: f64,
    pub drop2_s: f64,
    pub attack_s: f64,
    /// The level at the strike, as a share of the level the attack rises to: 0 rises from silence;
    /// above it the table swells over its attack, as a snare's resonant head does, driven through
    /// the shell's air (Virtuosity's wires-off snare: its 335 Hz line 3.4 dB up over 49 ms, read as
    /// falling from the strike 5–10 dB loud there).
    pub swell: f64,
    pub t60_s: f64,
    pub tail: f64,
    pub tail_s: f64,
    pub lp_from_hz: f64,
    pub lp_to_hz: f64,
    pub lp_s: f64,
    pub q: f64,
    /// Route: its level.
    pub vel_level: f64,
}

impl Default for Table {
    /// Off; on, a plain sine at 100 Hz through an open low-pass.
    fn default() -> Self {
        let mut overtones = [Overtone {
            ratio: 1.0,
            level: 0.0,
            phase: 0.0,
        }; OVERTONES];
        overtones[0].level = 1.0;
        Self {
            level: 0.0,
            hz: 100.0,
            overtones,
            step: 1.0 / 64.0,
            drop: 0.0,
            drop_s: 0.05,
            drop2: 0.0,
            drop2_s: 0.004,
            attack_s: 0.001,
            swell: 0.0,
            t60_s: 0.5,
            tail: 0.0,
            tail_s: 1.0,
            lp_from_hz: 20_000.0,
            lp_to_hz: 20_000.0,
            lp_s: 0.1,
            q: FRAC_1_SQRT_2,
            vel_level: 0.0,
        }
    }
}

/// A table's one period: each overtone at its whole multiple of the step, from phase 0.
fn table_period(t: &Table) -> Vec<f32> {
    let step = t.step.clamp(1e-4, 1.0);
    let mut out = vec![0.0f32; TABLE_LEN];
    for o in t.overtones.iter().filter(|o| o.level > 0.0) {
        let m = (o.ratio / step).round().max(1.0);
        for (i, v) in out.iter_mut().enumerate() {
            *v += (o.level * (TAU * m * i as f64 / TABLE_LEN as f64 + o.phase).sin()) as f32;
        }
    }
    out
}

/// A table running. Its period is shared: a voice cloned, or rebuilt at another rate, plays the same
/// tables without building them again (half a megabyte each). Beside it, its period at each softer
/// reading of the metal whose lines move from the hardest's balance (`table_readings`), empty where
/// they do not; a strike plays the two either side of its velocity, their lines the same and in
/// phase, so their levels blend straight.
#[derive(Debug, Clone)]
struct TableRun {
    period: Arc<[f32]>,
    readings: Vec<Arc<[f32]>>,
    /// The periods playing, 0 the hardest's and `i + 1` reading `i`'s, and their gains.
    playing: [usize; 2],
    mix: [f64; 2],
    /// The period's own frequency, `hz·step`.
    base: f64,
    phase: f64,
    level: f64,
    drop_env: f64,
    drop_fall: f64,
    drop2_env: f64,
    drop2_fall: f64,
    first: f64,
    first_fall: f64,
    second: f64,
    second_fall: f64,
    lp: Svf,
}

impl TableRun {
    fn new(spec: &Table, period: Arc<[f32]>, readings: Vec<Arc<[f32]>>, sample_rate: f64) -> Self {
        Self {
            period,
            readings,
            playing: [0; 2],
            mix: [1.0, 0.0],
            base: spec.hz * spec.step.clamp(1e-4, 1.0),
            phase: 0.0,
            level: 0.0,
            drop_env: 0.0,
            drop_fall: 0.0,
            drop2_env: 0.0,
            drop2_fall: 0.0,
            first: 0.0,
            first_fall: 0.0,
            second: 0.0,
            second_fall: 0.0,
            lp: Svf::new(spec.lp_from_hz, spec.q, sample_rate),
        }
    }

    fn start(&mut self, spec: &Table, velocity: f64, sample_rate: f64) {
        self.level = if self.period.is_empty() {
            0.0
        } else {
            routed(spec.level, spec.vel_level, velocity) * velocity
        };
        self.phase = 0.0;
        self.playing = [0; 2];
        self.mix = [1.0, 0.0];
        // The pitch as struck: a velocity's tune moves it.
        self.base = spec.hz * spec.step.clamp(1e-4, 1.0);
        self.drop_env = 1.0;
        self.drop_fall = (-1.0 / (spec.drop_s.max(1e-4) * sample_rate)).exp();
        self.drop2_env = 1.0;
        self.drop2_fall = (-1.0 / (spec.drop2_s.max(1e-4) * sample_rate)).exp();
        self.first = 1.0 - spec.tail.clamp(0.0, 1.0);
        self.first_fall = fall(spec.t60_s, sample_rate);
        self.second = spec.tail.clamp(0.0, 1.0);
        self.second_fall = fall(spec.tail_s, sample_rate);
        self.lp = Svf::new(spec.lp_from_hz, spec.q, sample_rate);
    }

    fn tick(&mut self, spec: &Table, age: u64, sample_rate: f64) -> f64 {
        if self.level <= 0.0 {
            return 0.0;
        }
        if age.is_multiple_of(BANK_BLOCK) {
            // The low-pass's corner moves from `lp_from_hz` to `lp_to_hz`.
            let t = age as f64 / sample_rate;
            let to = spec.lp_to_hz.max(1.0);
            let corner =
                to * (spec.lp_from_hz.max(1.0) / to).powf((-t / spec.lp_s.max(1e-4)).exp());
            self.lp.retune(corner, spec.q, sample_rate);
        }
        let at = self.phase * TABLE_LEN as f64;
        let i = at as usize % TABLE_LEN;
        let frac = at - at.floor();
        let read = |period: &[f32]| {
            let (a, b) = (f64::from(period[i]), f64::from(period[(i + 1) % TABLE_LEN]));
            a + frac * (b - a)
        };
        let wave = if self.mix[1] > 0.0 {
            self.mix[0] * read(self.set(self.playing[0]))
                + self.mix[1] * read(self.set(self.playing[1]))
        } else {
            read(self.set(self.playing[0]))
        };
        let pitch = 1.0 + spec.drop * self.drop_env + spec.drop2 * self.drop2_env;
        self.phase = (self.phase + self.base * pitch / sample_rate).fract();
        self.drop_env *= self.drop_fall;
        self.drop2_env *= self.drop2_fall;
        let swell = spec.swell.clamp(0.0, 1.0);
        let env = (swell + (1.0 - swell) * rise(age, spec.attack_s * sample_rate))
            * (self.first + self.second);
        self.first *= self.first_fall;
        self.second *= self.second_fall;
        self.lp.low(self.level * env * wave)
    }

    fn loudness(&self) -> f64 {
        self.level * (self.first + self.second)
    }

    /// A set's period: 0 the hardest's, `i + 1` reading `i`'s, the hardest's where it has none.
    fn set(&self, set: usize) -> &[f32] {
        match set.checked_sub(1).and_then(|i| self.readings.get(i)) {
            Some(r) if !r.is_empty() => r,
            _ => &self.period,
        }
    }

    /// The periods a strike plays, as `Metal::tones_at`'s sets and how far along from the first.
    fn choose(&mut self, [first, second]: [usize; 2], w: f64) {
        let own = |set: usize| {
            set.checked_sub(1)
                .and_then(|i| self.readings.get(i))
                .is_some_and(|r| !r.is_empty())
        };
        let (first, second) = (
            if own(first) { first } else { 0 },
            if own(second) { second } else { 0 },
        );
        if first == second || w <= 0.0 {
            self.playing = [first; 2];
            self.mix = [1.0, 0.0];
        } else {
            self.playing = [first, second];
            self.mix = [1.0 - w.min(1.0), w.min(1.0)];
        }
    }
}

/// Table `n`'s period at each softer reading of the metal whose lines move from the hardest's
/// balance (`MetalTone::tables_db`), empty where they do not.
fn table_readings(spec: &SynthKickSpec, n: usize) -> Vec<Arc<[f32]>> {
    let t = &spec.tables[n];
    if spec.metal.level <= 0.0 || t.level <= 0.0 {
        return Vec::new();
    }
    spec.metal
        .softer
        .iter()
        .map(|s| match s.tone {
            Some(tone) if s.velocity > 0.0 && tone.tables_db[n].iter().any(|&db| db != 0.0) => {
                let mut moved = *t;
                for (o, db) in moved.overtones.iter_mut().zip(&tone.tables_db[n]) {
                    o.level *= 10f64.powf(db / 20.0);
                }
                Arc::from(table_period(&moved))
            }
            _ => Arc::from(Vec::new()),
        })
        .collect()
}

/// A cymbal's ring in fourteen bands (`Metal`): what the model-drums plan measured on Big Rusty's
/// cymbals (revisions 13–15) — noise alone is too flat, partials alone too peaky, and what holds is
/// many partials over a floor, every partial following its band's level, each band's level a curve
/// over the whole ring. Measured on its crash against a second strike (flatness 1.2 dB off), the
/// cymbal's own 905 partials read 2.7 dB off and as many scattered at random in their bands 2.8, so
/// the bands scatter theirs, by seed; filtered noise read 7.9 and FM 10.3.
/// From 50 Hz: a crash's strike holds as much below 150 Hz in its first 30 ms as above it.
pub const METAL_BANDS: usize = 14;
/// The points each band's level curve passes through.
pub const METAL_KNOTS: usize = 10;
/// The points' times, s: close through the strike, wider as the ring slows.
pub const METAL_KNOT_S: [f64; METAL_KNOTS] =
    [0.0, 0.002, 0.007, 0.02, 0.06, 0.17, 0.5, 1.5, 4.0, 8.0];
/// The lines read exactly rather than scattered: a sparse band's, whose few scattered partials beat
/// at random rates, and the strongest low ones.
pub const METAL_EXACT: usize = 128;
/// An exact line trimmed this far down or further (`MetalTone::exact_trim_db`) is taken out of its
/// band whole.
pub const METAL_EXACT_GONE_DB: f64 = -120.0;
/// A band table's period, s: its partials sit on multiples of its inverse (rounded at most a quarter
/// of a hertz), so the table loops without a seam however many it holds. Each band has its own,
/// between 1.6 and 2.4 s, so the bands' repeats never fall together (the owner, 2026-10-10: the
/// loops staggered, "like this").
#[must_use]
pub fn metal_period_s(band: usize) -> f64 {
    // The golden ratio's multiples, folded: the periods spread evenly, none twice.
    let golden = 0.618_033_988_749_894_9;
    1.6 + 0.8 * (band as f64 * golden).fract()
}
/// A band table's length: 54 613 to 81 920 samples a second as its period goes, room for partials
/// up to 27 kHz.
const METAL_LEN: usize = 1 << 17;
/// After its last point a band falls at least this fast, dB a second, so every ring ends.
const METAL_TAIL_DB_S: f64 = 6.0;
/// A band's table starts where its first 20 ms, and the 60 after, sound at its mean level. A band
/// of a few close lines beats, and where its loop started set the crash's 417–638 Hz band 4 dB
/// under the recording through its first 80 ms, or 3 over.
const CALM_S: [f64; 2] = [0.02, 0.08];
/// Of this many places evenly round the table, the calmest this many.
const CALM_TRIED: usize = 512;
const CALM_KEPT: usize = 16;
/// The tick's low cut, Hz: a fourth-order Butterworth high-pass, whose two sections' Q are
/// `TICK_CUT_Q`. A Gaussian's spectrum is flat to DC, and the crash's 50–117 Hz stood 20–33 dB over
/// the recording's through the strike (the owner, 2026-10-11: "too low", "too dark"); the cut takes
/// 52, 38, 23 and 9 dB of it from the four lowest bands and nothing above 417 Hz. A Ricker wavelet
/// instead, its peak between samples, lost the attack to the ring (82 established differences on
/// the crash's hardest layer, the cut Gaussian 77).
pub const TICK_CUT_HZ: f64 = 300.0;
pub const TICK_CUT_Q: [f64; 2] = [0.541_196_100_146_197, 1.306_562_964_876_376_6];
/// The cut rings this long after the spike, s, and is then not run.
const TICK_RING_S: f64 = 0.02;

/// A sizzle cymbal's rattle (the owner, 2026-10-19: "high frequency sizzle that goes in and out",
/// "sizzle or metallic rattle"): its rivets kicked loose again and again as the cymbal swings. Each
/// kick lifts the top bands at once, by `jump_db` on average (`rattle_depth`; each kick drawn
/// evenly between half and one and a half times it, as the takes' kicks differ), and the lift falls
/// back exponentially over `relax_s`; the kicks come at random, `rate_hz` a second on average (a
/// Poisson process, drawn anew each strike: a seed's one draw would kick every hit alike, and a
/// rare rattle's long first wait would leave every hit of it without). The lift is taken about its mean, so the bands' curves still set
/// their level. Read on Big Rusty's sizzle crash and ride, 4–16 kHz in 5 ms frames about a 1 s running
/// median (2026-10-19): 0.3–6 kicks a second of 3–5 dB, the level scattering 0.6–2.3 dB, where a
/// model without it scatters 0.3–0.5 and never kicks; the swings move together from 3.5 kHz up, and
/// not below 2 kHz, where the cymbal's own modes beat apart. Off at a rate of 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rattle {
    /// Kicks a second, on average.
    pub rate_hz: f64,
    /// A kick's mean lift where the depth is 1 (`RATTLE_UNIT_HZ`), dB.
    pub jump_db: f64,
    /// Its time constant falling back, s.
    pub relax_s: f64,
}

impl Rattle {
    pub const OFF: Self = Self {
        rate_hz: 0.0,
        jump_db: 0.0,
        relax_s: 0.0,
    };

    /// Whether it sounds.
    #[must_use]
    pub fn is_on(&self) -> bool {
        self.rate_hz > 0.0 && self.jump_db > 0.0 && self.relax_s > 0.0
    }

    /// `w` of the way from `self` to `to`: the rate and the lift straight, the time in octaves.
    #[must_use]
    pub fn toward(&self, to: &Self, w: f64) -> Self {
        let relax = if self.relax_s > 0.0 && to.relax_s > 0.0 {
            self.relax_s * (to.relax_s / self.relax_s).powf(w)
        } else {
            self.relax_s + w * (to.relax_s - self.relax_s)
        };
        Self {
            rate_hz: self.rate_hz + w * (to.rate_hz - self.rate_hz),
            jump_db: self.jump_db + w * (to.jump_db - self.jump_db),
            relax_s: relax,
        }
    }
}

impl Default for Rattle {
    fn default() -> Self {
        Self::OFF
    }
}

/// How deep the rattle swings a band against `Rattle::jump_db`, by the band's centre: nothing up to
/// `RATTLE_FROM_HZ`, 1 at `RATTLE_UNIT_HZ`, and on up by octaves (0.4 at 4.3 kHz, 1.3 at 15 kHz on the
/// default bands). Read on the same takes: the 4–8 kHz swing about 0.7 of the 8–16 kHz one, moving
/// with it (correlation 0.4–0.8), the 16–22 kHz about 1.5–2 times; 2–4 kHz hardly with it (0.1–0.45).
const RATTLE_FROM_HZ: f64 = 2_600.0;
const RATTLE_UNIT_HZ: f64 = 10_000.0;
/// A kick's rise, s: the takes' kicks rise over 5–15 ms; at once, a band's lift is a click.
const RATTLE_RISE_S: f64 = 0.003;

/// The rattle's depth in band `b` of `m` (`RATTLE_FROM_HZ`).
fn rattle_depth(m: &Metal, b: usize) -> f64 {
    let (lo, hi) = m.edges(b);
    let centre = (lo * hi).sqrt();
    if centre <= RATTLE_FROM_HZ {
        return 0.0;
    }
    (centre / RATTLE_FROM_HZ).log2() / (RATTLE_UNIT_HZ / RATTLE_FROM_HZ).log2()
}

/// One band of the metal: its scattered partials, the noise between them, and its level over time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetalBand {
    /// How many partials are scattered in the band (rounded); none, and the band is its floor.
    pub partials: f64,
    /// Their levels' spread, dB: a log-normal's standard deviation around their mean.
    pub spread_db: f64,
    /// The share of the band's power that is noise, between the partials, 0–1.
    pub floor: f64,
    /// The band's level at each point of `METAL_KNOT_S`, dB.
    pub curve_db: [f64; METAL_KNOTS],
    /// The points' times taken this many times: 1 as read, the controls' decay moves it.
    pub stretch: f64,
}

impl Default for MetalBand {
    fn default() -> Self {
        Self {
            partials: 0.0,
            spread_db: 6.0,
            floor: 0.0,
            curve_db: [-120.0; METAL_KNOTS],
            stretch: 1.0,
        }
    }
}

/// The bands' levels over time as a softer strike reads them: how the balance between the bands
/// moves with the strike, which the velocity's controls, one tilt over all the bands, could not
/// follow (the owner, 2026-10-11: the soft hits "off pitch/resonance", "more muddy"; the open
/// hi-hat's soft 273–638 Hz rang 14–20 dB over its take's). Its tables' likewise: stepped by the
/// controls, the open hi-hat's soft 654 and 507 Hz rang as long as its 645, which alone rings in the
/// take, and beat with it 9 times a second ("the model looks more on and off").
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetalSofter {
    /// The strike's velocity, 0–1; 0, none.
    pub velocity: f64,
    /// Each band's `MetalBand::curve_db` at it, its overall level the hardest's.
    pub curve_db: [[f64; METAL_KNOTS]; METAL_BANDS],
    /// The tick's peak at it, as `Metal::strike`.
    pub strike: f64,
    /// Each table's level at it against the hardest's, dB, the overall level the hardest's; and
    /// its T60s', a factor.
    pub tables_db: [f64; TABLES],
    pub tables_decay: [f64; TABLES],
    /// Its own tone (`MetalTone`). A softer strike rings its own lines clean where the hardest's
    /// are buried in noise: the crash's soft 2182, 2986 and 3366 Hz stood 38–45 dB over their floor
    /// and 14–36 dB over the hardest's tables there (the owner, 2026-10-12: "you are constantly
    /// missing the pitch part", "the balance on the tone"). None, the hardest's tables play it.
    /// Held by reference: a patch is copied at every strike, and four kilobytes more in each copy
    /// overflowed a debug build's two-megabyte stack through the plugin's `process`.
    pub tone: Option<&'static MetalTone>,
    /// Its rattle (`Rattle`): a soft hit kicks the rivets less often.
    pub rattle: Rattle,
}

/// A reading's tone, as the hardest's is in `MetalBand` and `Metal`: each band's scattered
/// partials, their spread and the noise between, and the exact lines.
#[derive(Debug, Clone, PartialEq)]
pub struct MetalTone {
    pub partials: [f64; METAL_BANDS],
    pub spread_db: [f64; METAL_BANDS],
    pub floor: [f64; METAL_BANDS],
    pub exact_hz: [f64; METAL_EXACT],
    pub exact_db: [f64; METAL_EXACT],
    /// Each exact line moved after its band is scaled to its curve, dB; 0 as read. `exact_db` is
    /// read into the band's power, which its curve then sets, so moving one line there moves its
    /// band's others the other way: the undamped kick's 64–104 Hz band, one line turned down,
    /// rang the rest up, and dropped, it handed the snare's soft 61–75 Hz to its 52 and 66 Hz,
    /// 16–20 dB over the take (2026-10-13). Trimmed, a line moves alone.
    pub exact_trim_db: [f64; METAL_EXACT],
    /// Each table's lines at it against the table's own level (`MetalSofter::tables_db`), dB, as
    /// `Table::overtones`; 0, the hardest's balance. A table read whole follows its loudest line:
    /// the 14-inch tom's soft table rose 15.8 dB with its 96 Hz and carried its 405 Hz 24 dB over
    /// the take's (2026-10-12).
    pub tables_db: [[f64; OVERTONES]; TABLES],
}

impl Default for MetalTone {
    fn default() -> Self {
        Self {
            partials: [0.0; METAL_BANDS],
            spread_db: [6.0; METAL_BANDS],
            floor: [0.0; METAL_BANDS],
            exact_hz: [0.0; METAL_EXACT],
            exact_db: [0.0; METAL_EXACT],
            exact_trim_db: [0.0; METAL_EXACT],
            tables_db: [[0.0; OVERTONES]; TABLES],
        }
    }
}

impl Default for MetalSofter {
    fn default() -> Self {
        Self {
            velocity: 0.0,
            curve_db: [[-120.0; METAL_KNOTS]; METAL_BANDS],
            strike: 0.0,
            tables_db: [0.0; TABLES],
            tables_decay: [1.0; TABLES],
            tone: None,
            rattle: Rattle::OFF,
        }
    }
}

/// A cymbal's ring: fourteen bands between `lo_hz` and `hi_hz`, geometric, each a table of its
/// partials and the noise between them under its own level curve. Off at level 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metal {
    /// Linear; 0 off.
    pub level: f64,
    pub lo_hz: f64,
    pub hi_hz: f64,
    pub bands: [MetalBand; METAL_BANDS],
    /// The exact lines, Hz (0 none), each in the band it falls in; their levels, dB against that
    /// band's scattered partials' mean.
    pub exact_hz: [f64; METAL_EXACT],
    pub exact_db: [f64; METAL_EXACT],
    /// The hardest's exact lines' trims (`MetalTone::exact_trim_db`); none, as read. Held by
    /// reference, as the softer tones are.
    pub exact_trim_db: Option<&'static [f64; METAL_EXACT]>,
    /// How fast the tables play: 1 as read, the controls' tune moves it.
    pub rate: f64,
    /// Where the scattered partials fall.
    pub seed: u64,
    /// The stick's tick: a Gaussian pulse at the strike less its lows (`TICK_CUT_HZ`), its peak
    /// (signed) and its width σ, s. A crash's strike is one spike 0.07 ms in, 22–27 dB over the ring
    /// that follows; the bands, a few milliseconds apart, cannot make it, and the listener reads the
    /// attack against it: without it the crash read 86 established differences against 78. The peak
    /// is the one read, the cut's loss scaled back.
    pub strike: f64,
    pub strike_s: f64,
    /// The softest strike's reading, then the middle's; the hardest's is the bands' own, at 1.
    pub softer: [MetalSofter; 2],
    /// The hardest's rattle (`Rattle`); a softer reading has its own.
    pub rattle: Rattle,
}

impl Default for Metal {
    fn default() -> Self {
        Self {
            level: 0.0,
            lo_hz: 50.0,
            hi_hz: 19_000.0,
            bands: [MetalBand::default(); METAL_BANDS],
            exact_hz: [0.0; METAL_EXACT],
            exact_db: [0.0; METAL_EXACT],
            exact_trim_db: None,
            rate: 1.0,
            seed: 1,
            strike: 0.0,
            strike_s: 3e-5,
            softer: [MetalSofter::default(); 2],
            rattle: Rattle::OFF,
        }
    }
}

impl Metal {
    /// Band `b`'s edges, Hz, as read (before `rate`).
    #[must_use]
    pub fn edges(&self, b: usize) -> (f64, f64) {
        let ratio = (self.hi_hz / self.lo_hz.max(1.0)).max(1.0);
        let at = |k: usize| self.lo_hz * ratio.powf(k as f64 / METAL_BANDS as f64);
        (at(b), at(b + 1))
    }

    /// Whether softer strikes were read (`softer`).
    #[must_use]
    pub fn reads_softer(&self) -> bool {
        self.softer.iter().any(|s| s.velocity > 0.0)
    }

    /// The metal struck at `v`: its curves and tick between the readings either side of it, in dB
    /// and in amplitude straight along the velocity, the softest's below it and the hardest's, at
    /// 1, above the rest.
    #[must_use]
    pub fn at_velocity(&self, v: f64) -> Self {
        let Some((a, b, w)) = self.readings_at(v) else {
            return *self;
        };
        let mut m = *self;
        for (band, (ca, cb)) in m.bands.iter_mut().zip(a.curve_db.iter().zip(&b.curve_db)) {
            for (k, db) in band.curve_db.iter_mut().enumerate() {
                *db = ca[k] + w * (cb[k] - ca[k]);
            }
        }
        m.strike = a.strike + w * (b.strike - a.strike);
        m.rattle = a.rattle.toward(&b.rattle, w);
        m
    }

    /// Each table's level (dB against the hardest's) and T60s' factor struck at `v`, between the
    /// readings either side of it as `at_velocity`'s: the level in dB and the factor in octaves
    /// straight along the velocity.
    #[must_use]
    pub fn tables_at(&self, v: f64) -> [(f64, f64); TABLES] {
        let Some((a, b, w)) = self.readings_at(v) else {
            return [(0.0, 1.0); TABLES];
        };
        std::array::from_fn(|n| {
            let db = a.tables_db[n] + w * (b.tables_db[n] - a.tables_db[n]);
            let (da, db_) = (a.tables_decay[n].max(1e-9), b.tables_decay[n].max(1e-9));
            (db, da * (db_ / da).powf(w))
        })
    }

    /// The band tables a strike at `v` plays, as `metal_tables`' sets (0 the hardest's, 1 and 2 the
    /// softest's and the middle's own tones), and how far along from the first: the readings
    /// either side of it as `at_velocity`'s, a reading without its own tone playing the
    /// hardest's.
    #[must_use]
    pub fn tones_at(&self, v: f64) -> ([usize; 2], f64) {
        let set = |i: usize| match self.softer.get(i) {
            Some(s) if s.tone.is_some() => i + 1,
            _ => 0,
        };
        match self.anchors_at(v) {
            Some((a, b, w)) if set(a) != set(b) => ([set(a), set(b)], w),
            Some((a, _, _)) => ([set(a); 2], 0.0),
            None => ([0; 2], 0.0),
        }
    }

    /// The readings either side of `v` and how far along from the first, the hardest's at 1;
    /// `None` without a softer one.
    fn readings_at(&self, v: f64) -> Option<(MetalSofter, MetalSofter, f64)> {
        let (a, b, w) = self.anchors_at(v)?;
        Some((self.reading(a), self.reading(b), w))
    }

    /// Reading `i`: the softest's and the middle's, then the hardest's (2).
    fn reading(&self, i: usize) -> MetalSofter {
        self.softer.get(i).copied().unwrap_or(MetalSofter {
            velocity: 1.0,
            curve_db: self.bands.map(|b| b.curve_db),
            strike: self.strike,
            rattle: self.rattle,
            ..MetalSofter::default()
        })
    }

    /// `readings_at`'s, as the readings' indices (`reading`).
    fn anchors_at(&self, v: f64) -> Option<(usize, usize, f64)> {
        let velocity = |i: usize| if i < 2 { self.softer[i].velocity } else { 1.0 };
        let mut lower: Option<usize> = None;
        let mut upper: Option<usize> = None;
        for i in (0..3).filter(|&i| velocity(i) > 0.0) {
            if velocity(i) <= v {
                lower = Some(i);
            } else if upper.is_none() {
                upper = Some(i);
            }
        }
        match (lower, upper) {
            (Some(a), Some(b)) => Some((a, b, (v - velocity(a)) / (velocity(b) - velocity(a)))),
            (Some(a), None) => Some((a, a, 0.0)),
            (None, Some(b)) => Some((b, b, 0.0)),
            (None, None) => None,
        }
    }
}

/// An in-place radix-2 FFT of `re + i·im`, whose length is a power of two.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let (wi, wr) = (-TAU / len as f64).sin_cos();
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0, 0.0);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let tr = re[b] * cr - im[b] * ci;
                let ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                (cr, ci) = (cr * wr - ci * wi, cr * wi + ci * wr);
            }
        }
        len <<= 1;
    }
}

/// The metal's band tables, one set for each reading with its own tone (`Metal::tones_at`): the
/// hardest's, then the softest's and the middle's, empty where the reading has none.
fn metal_tables(m: &Metal) -> Vec<Vec<MetalTable>> {
    let hard = BandTone {
        partials: m.bands.map(|b| b.partials),
        spread_db: m.bands.map(|b| b.spread_db),
        floor: m.bands.map(|b| b.floor),
        exact_hz: &m.exact_hz,
        exact_db: &m.exact_db,
        exact_trim_db: m.exact_trim_db,
        seed: m.seed,
    };
    let mut sets = vec![band_tables(m, &hard)];
    for (i, s) in m.softer.iter().enumerate() {
        sets.push(match s.tone {
            Some(t) if s.velocity > 0.0 => band_tables(
                m,
                &BandTone {
                    partials: t.partials,
                    spread_db: t.spread_db,
                    floor: t.floor,
                    exact_hz: &t.exact_hz,
                    exact_db: &t.exact_db,
                    exact_trim_db: Some(&t.exact_trim_db),
                    seed: m.seed.wrapping_add(STRIKE_SEED.wrapping_mul(i as u64 + 1)),
                },
            ),
            _ => Vec::new(),
        });
    }
    sets
}

/// What a set of band tables holds (`band_tables`).
struct BandTone<'a> {
    partials: [f64; METAL_BANDS],
    spread_db: [f64; METAL_BANDS],
    floor: [f64; METAL_BANDS],
    exact_hz: &'a [f64; METAL_EXACT],
    exact_db: &'a [f64; METAL_EXACT],
    exact_trim_db: Option<&'a [f64; METAL_EXACT]>,
    seed: u64,
}

/// A tone's band tables, each one period of its partials and its floor at unit RMS: the
/// scattered partials by seed (log-uniform in the band, levels log-normal), the exact lines that
/// fall in it, and noise in every bin between. Empty for a silent band. Built once by an inverse
/// FFT; the tables do not depend on the rate. A trimmed exact line is then moved by its trim at the
/// phase it was given, the band's scale left as read.
fn band_tables(m: &Metal, tone: &BandTone) -> Vec<MetalTable> {
    (0..METAL_BANDS)
        .map(|b| {
            let (lo, hi) = m.edges(b);
            let mut white = white(
                tone.seed
                    .wrapping_add(METAL_SEED.wrapping_mul(b as u64 + 1)),
            );
            let mut uniform = || 0.5 * (white.bipolar_f64() + 1.0);
            // Each line, its amplitude and, an exact one's, its trim.
            let mut lines: Vec<(f64, f64, f64)> = Vec::new();
            for _ in 0..tone.partials[b].round().max(0.0) as usize {
                let hz = lo * (hi / lo).powf(uniform());
                let (u1, u2) = (uniform().max(1e-12), uniform());
                let z = (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos();
                lines.push((hz, 10f64.powf(tone.spread_db[b] * z / 20.0), 0.0));
            }
            for (i, (&hz, &db)) in tone.exact_hz.iter().zip(tone.exact_db).enumerate() {
                if hz >= lo && hz < hi {
                    let trim = tone.exact_trim_db.map_or(0.0, |t| t[i]);
                    lines.push((hz, 10f64.powf(db / 20.0), trim));
                }
            }
            let period = metal_period_s(b);
            let floor = if lines.is_empty() {
                1.0
            } else {
                tone.floor[b].clamp(0.0, 1.0)
            };
            let mut re = vec![0.0; METAL_LEN];
            let mut im = vec![0.0; METAL_LEN];
            // What the trims move, as each line's bin, amplitude and phase.
            let mut trimmed: Vec<(usize, f64, f64)> = Vec::new();
            for (hz, amp, trim) in lines {
                let k = ((hz * period).round() as usize).clamp(1, METAL_LEN / 2 - 1);
                let phase = TAU * uniform();
                // The conjugate's transform is the inverse's conjugate: its real part is the sum
                // of the cosines.
                re[k] += amp * phase.cos();
                im[k] -= amp * phase.sin();
                if trim != 0.0 {
                    let to = if trim <= METAL_EXACT_GONE_DB {
                        0.0
                    } else {
                        10f64.powf(trim / 20.0)
                    };
                    trimmed.push((k, amp * (to - 1.0), phase));
                }
            }
            // The floor: noise in every bin of the band at random phases, `floor` of the band's
            // power, all of it in a band without lines. Band-limited as the partials are: a
            // band-pass's skirts let the stack's loud middle bands' noise into its 50–273 Hz bands,
            // 13–39 dB over the recording's through the strike, its floors left out 1–13.
            let bin = |hz: f64| ((hz * period).ceil() as usize).clamp(1, METAL_LEN / 2);
            let (k0, k1) = (bin(lo), bin(hi));
            if floor > 0.0 && k1 > k0 {
                let lines_power: f64 = re.iter().zip(&im).map(|(r, i)| r * r + i * i).sum();
                let wanted = if floor >= 1.0 {
                    1.0
                } else {
                    lines_power * floor / (1.0 - floor)
                };
                // Each bin's Rayleigh amplitude has mean square twice its σ².
                let each = (wanted / (2.0 * (k1 - k0) as f64)).sqrt();
                for k in k0..k1 {
                    let (u1, u2) = (uniform().max(1e-12), uniform());
                    let r = (-2.0 * u1.ln()).sqrt() * each;
                    re[k] += r * (TAU * u2).cos();
                    im[k] += r * (TAU * u2).sin();
                }
            }
            if re.iter().all(|v| *v == 0.0) && im.iter().all(|v| *v == 0.0) {
                return MetalTable::default();
            }
            fft(&mut re, &mut im);
            let rms = (re.iter().map(|v| v * v).sum::<f64>() / METAL_LEN as f64).sqrt();
            let scale = if rms > 0.0 { 1.0 / rms } else { 0.0 };
            if !trimmed.is_empty() {
                let mut moved_re = vec![0.0; METAL_LEN];
                let mut moved_im = vec![0.0; METAL_LEN];
                for (k, amp, phase) in trimmed {
                    moved_re[k] += amp * phase.cos();
                    moved_im[k] -= amp * phase.sin();
                }
                fft(&mut moved_re, &mut moved_im);
                for (v, d) in re.iter_mut().zip(&moved_re) {
                    *v += d;
                }
            }
            let samples: Vec<f32> = re.iter().map(|v| (v * scale) as f32).collect();
            let starts = calm_starts(&samples, period);
            MetalTable {
                samples: Arc::from(samples),
                starts: Arc::from(starts),
            }
        })
        .collect()
}

/// Where a band's table may start: of `CALM_TRIED` places evenly round it, the `CALM_KEPT` whose
/// first `CALM_S` windows sound nearest its mean level (unit power), as the sum of their misses in
/// dB.
fn calm_starts(samples: &[f32], period: f64) -> Vec<u32> {
    let n = samples.len();
    // Squares summed from the start, twice round, so a window may wrap.
    let mut prefix = vec![0.0f64; 2 * n + 1];
    for i in 0..2 * n {
        let s = f64::from(samples[i % n]);
        prefix[i + 1] = prefix[i] + s * s;
    }
    let per_s = n as f64 / period;
    let (w1, w2) = (
        ((CALM_S[0] * per_s) as usize).clamp(1, n),
        ((CALM_S[1] * per_s) as usize).clamp(2, n),
    );
    let miss = |from: usize, to: usize, at: usize| {
        let power = (prefix[at + to] - prefix[at + from]) / (to - from) as f64;
        (10.0 * power.max(1e-12).log10()).abs()
    };
    let mut tried: Vec<(f64, u32)> = (0..CALM_TRIED)
        .map(|c| {
            let at = c * n / CALM_TRIED;
            (miss(0, w1, at) + miss(w1, w2.max(w1 + 1), at), at as u32)
        })
        .collect();
    tried.sort_by(|a, b| a.0.total_cmp(&b.0));
    tried.iter().take(CALM_KEPT).map(|t| t.1).collect()
}

/// One band's table, and the places it may start (`calm_starts`).
#[derive(Debug, Clone, Default)]
struct MetalTable {
    samples: Arc<[f32]>,
    starts: Arc<[u32]>,
}

/// One band of the metal sounding: its table's place, its noise filter and its level curve.
#[derive(Debug, Clone, Copy, Default)]
struct MetalBandRun {
    /// The table's place and its step a sample, in table samples.
    at: f64,
    step: f64,
    /// Between two readings' tones (`Metal::tones_at`), the second set's table, its place and
    /// both tables' gains; the second's 0, it is not played.
    second: usize,
    at_second: f64,
    mix: [f64; 2],
    env: f64,
    mult: f64,
    knot: usize,
    until: u64,
    times: [u64; METAL_KNOTS],
    gains: [f64; METAL_KNOTS],
    tail: f64,
}

impl MetalBandRun {
    /// The envelope one sample on, `age` the sample just played.
    fn advance(&mut self, age: u64) {
        if age + 1 < self.until {
            self.env *= self.mult;
            return;
        }
        self.env = self.gains[self.knot];
        self.knot += 1;
        if self.knot < METAL_KNOTS {
            let span = self.times[self.knot]
                .saturating_sub(self.times[self.knot - 1])
                .max(1);
            self.mult = (self.gains[self.knot] / self.env.max(1e-300)).powf(1.0 / span as f64);
            self.until = self.times[self.knot];
        } else {
            self.knot = METAL_KNOTS - 1;
            self.mult = self.tail;
            self.until = u64::MAX;
        }
    }
}

/// The metal sounding. Its tables are shared, as the other tables are: a set for each reading's
/// tone (`metal_tables`), and the one this strike plays first.
#[derive(Debug, Clone)]
struct MetalRun {
    tables: Vec<Vec<MetalTable>>,
    first: usize,
    bands: [MetalBandRun; METAL_BANDS],
    noise: Xorshift64Star,
    level: f64,
    /// The tick's peak, its width and centre in samples.
    strike: f64,
    strike_w: f64,
    strike_at: f64,
    /// The tick's low cut, the gain that restores its peak, and the age it stops ringing at.
    tick_cut: [Svf; 2],
    tick_gain: f64,
    tick_until: u64,
    active: bool,
    rattle: RattleRun,
}

/// The rattle sounding (`Rattle`): the kicks' sum in dB, its rise, and each band's depth.
#[derive(Debug, Clone)]
struct RattleRun {
    on: bool,
    /// SplitMix64, whose every draw mixes every bit of its seed: the seeds here differ in a few
    /// bits, and xorshift64*'s first draws from them kick in step.
    draws: SplitMix64,
    /// The age of the next kick, a kick's lift, and the rate they come at, a sample.
    next: u64,
    jump: f64,
    rate: f64,
    /// The kicks' sum, dB: each lifts it by `jump` and it falls by `fall` a sample, its mean `mean`.
    sum: f64,
    fall: f64,
    mean: f64,
    /// The sum risen (`RATTLE_RISE_S`), a one-pole's coefficient.
    risen: f64,
    rise: f64,
    /// Each band's depth, as a gain's exponent: ln 10 / 20 × `rattle_depth`.
    depth: [f64; METAL_BANDS],
}

impl Default for RattleRun {
    fn default() -> Self {
        Self {
            on: false,
            draws: SplitMix64::new(1),
            next: u64::MAX,
            jump: 0.0,
            rate: 0.0,
            sum: 0.0,
            fall: 1.0,
            mean: 0.0,
            risen: 0.0,
            rise: 0.0,
            depth: [0.0; METAL_BANDS],
        }
    }
}

impl RattleRun {
    fn start(&mut self, m: &Metal, seed: u64, sample_rate: f64) {
        let r = &m.rattle;
        self.on = r.is_on();
        if !self.on {
            return;
        }
        self.draws = SplitMix64::new(seed);
        self.jump = r.jump_db;
        self.rate = r.rate_hz / sample_rate;
        self.fall = (-1.0 / (r.relax_s * sample_rate)).exp();
        // Shot noise's mean: rate × lift × time constant. Started there, as if it had rung on.
        self.mean = r.rate_hz * r.jump_db * r.relax_s;
        self.sum = self.mean;
        self.risen = self.mean;
        self.rise = 1.0 - (-1.0 / (RATTLE_RISE_S * sample_rate)).exp();
        self.depth = std::array::from_fn(|b| rattle_depth(m, b) * std::f64::consts::LN_10 / 20.0);
        self.next = self.after(0);
    }

    /// The age of the kick after one at `age`: an exponential wait at the rate.
    fn after(&mut self, age: u64) -> u64 {
        let u = 1.0 - self.draws.unit_f64();
        age.saturating_add((-u.ln() / self.rate).ceil().max(1.0) as u64)
    }

    /// Its lift now, about its mean, dB; and on a sample.
    fn step(&mut self, age: u64) -> f64 {
        if age >= self.next {
            self.sum += self.jump * (0.5 + self.draws.unit_f64());
            self.next = self.after(age);
        }
        self.sum *= self.fall;
        self.risen += self.rise * (self.sum - self.risen);
        self.risen - self.mean
    }
}

impl MetalRun {
    fn new(tables: Vec<Vec<MetalTable>>) -> Self {
        Self {
            tables,
            first: 0,
            bands: [MetalBandRun::default(); METAL_BANDS],
            noise: white(1),
            level: 0.0,
            strike: 0.0,
            strike_w: 1.0,
            strike_at: 0.0,
            tick_cut: [Svf::default(); 2],
            tick_gain: 0.0,
            tick_until: 0,
            active: false,
            rattle: RattleRun::default(),
        }
    }

    /// The tick's Gaussian at `age`, before its cut.
    fn pulse(&self, age: u64) -> f64 {
        let from_tick = (age as f64 - self.strike_at) / self.strike_w;
        if from_tick < 3.0 {
            (-0.5 * from_tick * from_tick).exp()
        } else {
            0.0
        }
    }

    /// `strike` counts the voice's strikes: the rattle's kicks fall anew each one, as a cymbal's
    /// rivets do, where the rest of the metal plays its seed's.
    fn start(&mut self, m: &Metal, v: f64, seed: u64, strike: u64, sample_rate: f64) {
        self.level = m.level;
        self.active = m.level > 0.0 && self.tables.first().is_some_and(|t| !t.is_empty());
        if !self.active {
            return;
        }
        // The two tones either side of the strike, each set's tables uncorrelated: their powers
        // add, so their gains are the roots of their shares.
        let ([first, second], w) = m.tones_at(v);
        let usable = |set: usize| self.tables.get(set).is_some_and(|t| !t.is_empty());
        let (first, second) = (
            if usable(first) { first } else { 0 },
            if usable(second) { second } else { 0 },
        );
        let (mix, second) = if first == second || w <= 0.0 {
            ([1.0, 0.0], first)
        } else {
            ([(1.0 - w).max(0.0).sqrt(), w.min(1.0).sqrt()], second)
        };
        self.first = first;
        self.noise = white(seed);
        self.strike = m.strike;
        self.strike_w = (m.strike_s * sample_rate).max(1e-3);
        self.strike_at = 3.0 * self.strike_w;
        self.tick_cut = TICK_CUT_Q.map(|q| Svf::new(TICK_CUT_HZ, q, sample_rate));
        // The cut lowers the spike's peak: scale it back to the peak read.
        let mut probe = self.tick_cut;
        let spike = (self.strike_at + 3.0 * self.strike_w).ceil() as u64 + 2;
        let top = (0..spike)
            .map(|age| {
                probe
                    .iter_mut()
                    .fold(self.pulse(age), |v, s| s.high(v))
                    .abs()
            })
            .fold(0.0f64, f64::max);
        self.tick_gain = if top > 0.0 { 1.0 / top } else { 0.0 };
        self.tick_until = spike + (TICK_RING_S * sample_rate) as u64;
        for (b, run) in self.bands.iter_mut().enumerate() {
            let band = &m.bands[b];
            let times =
                METAL_KNOT_S.map(|t| (t * band.stretch.max(0.0) * sample_rate).round() as u64);
            let gains = band.curve_db.map(|db| 10f64.powf(db / 20.0));
            let (k, j) = (METAL_KNOTS - 1, METAL_KNOTS - 2);
            let seconds = ((times[k] - times[j]) as f64 / sample_rate).max(1e-3);
            let slope = ((band.curve_db[k] - band.curve_db[j]) / seconds).min(-METAL_TAIL_DB_S);
            // Where a table starts, one of its calm places by the strike's seed: a new one each
            // strike with `fresh_noise`.
            let noise = &mut self.noise;
            let mut start_of = |table: &MetalTable| {
                let starts = &table.starts;
                let pick = (0.5 * (noise.bipolar_f64() + 1.0) * starts.len() as f64) as usize;
                starts
                    .get(pick.min(starts.len().saturating_sub(1)))
                    .map_or(0.0, |&s| f64::from(s))
            };
            let at = start_of(&self.tables[first][b]);
            let at_second = if mix[1] > 0.0 {
                start_of(&self.tables[second][b])
            } else {
                0.0
            };
            *run = MetalBandRun {
                at,
                step: m.rate * METAL_LEN as f64 / (metal_period_s(b) * sample_rate),
                second,
                at_second,
                mix,
                env: gains[0],
                mult: 1.0,
                knot: 0,
                until: 0,
                times,
                gains,
                tail: 10f64.powf(slope / 20.0 / sample_rate),
            };
            run.advance(0);
            run.env = gains[0];
        }
        // Its own settings in its seed too: the kit's patches share one seed, and their first
        // strikes would kick in step, a rare rattle's first wait as long in all of them.
        let r = &m.rattle;
        let rattle_seed = seed
            .wrapping_add(RATTLE_SEED)
            .wrapping_add(strike.wrapping_mul(STRIKE_SEED))
            ^ r.rate_hz.to_bits()
            ^ r.jump_db.to_bits().rotate_left(21)
            ^ r.relax_s.to_bits().rotate_left(42);
        self.rattle.start(m, rattle_seed, sample_rate);
    }

    fn tick(&mut self, age: u64) -> f64 {
        if !self.active {
            return 0.0;
        }
        let mut out = 0.0;
        if age < self.tick_until {
            let pulse = self.pulse(age);
            let cut = self.tick_cut.iter_mut().fold(pulse, |v, s| s.high(v));
            out += self.strike * self.tick_gain * cut;
        }
        let lift = if self.rattle.on {
            self.rattle.step(age)
        } else {
            0.0
        };
        let first = &self.tables[self.first];
        for (b, (run, table)) in self.bands.iter_mut().zip(first).enumerate() {
            let mut ring = Self::read(&table.samples, &mut run.at, run.step);
            if run.mix[1] > 0.0 {
                let second = &self.tables[run.second][b].samples;
                ring = run.mix[0] * ring
                    + run.mix[1] * Self::read(second, &mut run.at_second, run.step);
            }
            if self.rattle.on && self.rattle.depth[b] > 0.0 {
                ring *= (lift * self.rattle.depth[b]).exp();
            }
            out += run.env * ring;
            run.advance(age);
        }
        self.level * out
    }

    /// A table's sample at `at`, which moves on by `step` round it; 0 for an empty table.
    fn read(table: &[f32], at: &mut f64, step: f64) -> f64 {
        if table.is_empty() {
            return 0.0;
        }
        // Four points, cubic (Catmull–Rom): a straight line between two leaves images of the high
        // partials that fold into the middle bands.
        let i = *at as usize;
        let t = *at - i as f64;
        let point = |k: usize| f64::from(table[(i + k + METAL_LEN - 1) % METAL_LEN]);
        let (p0, p1, p2, p3) = (point(0), point(1), point(2), point(3));
        *at += step;
        if *at >= METAL_LEN as f64 {
            *at -= METAL_LEN as f64;
        }
        p1 + 0.5
            * t
            * (p2 - p0
                + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))
    }

    fn loudness(&self) -> f64 {
        if !self.active {
            return 0.0;
        }
        self.level * self.bands.iter().map(|b| b.env).fold(0.0, f64::max)
    }
}

/// A synthesized kick's patch: its sections and the recording chain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SynthKickSpec {
    pub body: Body,
    pub second: Tone,
    pub partials: Partials,
    pub click: Noise,
    pub noise: Noise,
    pub noise2: Noise,
    pub bank: Bank,
    pub tables: [Table; TABLES],
    /// The controls at the softest strike: a strike at velocity `v` sits between the preset (`v = 1`)
    /// and these (`v = 0`) — softer darker, less click, less pitch drop, as a player's softer stroke
    /// (the owner's controls, 2026-09-30). Neutral: no response beyond the level.
    pub velocity: Controls,
    /// A middle strike between the hardest and the softest, off by default: a drum whose middle
    /// strike is not on the line between them (Big Rusty's pillowed kick: damped as much at a medium
    /// strike as at a hard one, only a soft one ringing longer).
    pub middle: Middle,
    /// A cymbal's ring, off by default (`Metal`).
    pub metal: Metal,
    /// The recording chain's low cut on the mix, Hz: 0 none.
    pub low_cut_hz: f64,
    /// The noise's seed: a render is reproducible, and a fit can be read at other seeds.
    pub seed: u64,
    /// New noise every strike: each strike after a voice's first moves the seed on, so a roll's hits
    /// are not one noise again and again, as a snare's wires never land the same way twice. A voice
    /// built again starts from `seed`; off, every strike is the first's noise.
    pub fresh_noise: bool,
    pub output_gain: f64,
}

impl Default for SynthKickSpec {
    /// A plain body: 55 Hz dropping from twice that over 15 ms, a 1 ms attack, a 300 ms T60; every
    /// other section off. **Chosen**, a starting point, not a drum.
    fn default() -> Self {
        let noise = Noise {
            level: 0.0,
            hz: 1_000.0,
            q: FRAC_1_SQRT_2,
            attack_s: 0.0,
            t60_s: 0.01,
            tail: 0.0,
            tail_s: 1.0,
            vel_level: 0.0,
            vel_hz: 0.0,
        };
        Self {
            body: Body {
                level: 1.0,
                hz: 55.0,
                drop: 1.0,
                drop_s: 0.015,
                shape: 0.0,
                phase: 0.0,
                attack_s: 0.001,
                decay_s: 0.3,
                tail: 0.0,
                tail_s: 1.0,
                drive: 0.0,
                vel_drop: 0.0,
                vel_decay: 0.0,
            },
            second: Tone {
                level: 0.0,
                hz: 90.0,
                drop: 0.0,
                drop_s: 0.015,
                attack_s: 0.001,
                t60_s: 0.3,
                vel_level: 0.0,
            },
            partials: Partials {
                partials: [Partial::default(); PARTIALS],
                drop: 0.0,
                drop_s: 0.015,
                attack_s: 0.001,
                vel_level: 0.0,
            },
            click: noise,
            noise: Noise {
                hz: 600.0,
                attack_s: 0.001,
                t60_s: 0.06,
                ..noise
            },
            noise2: Noise {
                hz: 400.0,
                attack_s: 0.001,
                t60_s: 0.06,
                ..noise
            },
            bank: Bank::default(),
            tables: [Table::default(); TABLES],
            velocity: Controls::default(),
            middle: Middle::default(),
            metal: Metal::default(),
            low_cut_hz: 0.0,
            seed: 1,
            fresh_noise: false,
            output_gain: 1.0,
        }
    }
}

/// The trapezoidal state-variable filter (see *Filters*), `f64` state.
#[derive(Debug, Clone, Copy, Default)]
struct Svf {
    k: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    ic1: f64,
    ic2: f64,
}

impl Svf {
    fn new(centre_hz: f64, q: f64, sample_rate: f64) -> Self {
        let g = (PI * centre_hz.clamp(1.0, 0.45 * sample_rate) / sample_rate).tan();
        let k = 1.0 / q.max(0.05);
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        Self {
            k,
            a1,
            a2,
            a3: g * a2,
            ic1: 0.0,
            ic2: 0.0,
        }
    }

    /// The band output and the high-pass output for one sample.
    fn tick(&mut self, v0: f64) -> (f64, f64) {
        let v3 = v0 - self.ic2;
        let v1 = self.a1 * self.ic1 + self.a2 * v3;
        let v2 = self.ic2 + self.a2 * self.ic1 + self.a3 * v3;
        self.ic1 = 2.0 * v1 - self.ic1;
        self.ic2 = 2.0 * v2 - self.ic2;
        (v1, v0 - self.k * v1 - v2)
    }

    /// Band-pass, unity at its centre.
    fn band(&mut self, v0: f64) -> f64 {
        self.k * self.tick(v0).0
    }

    /// High-pass, unity above its corner.
    fn high(&mut self, v0: f64) -> f64 {
        self.tick(v0).1
    }

    /// Low-pass, unity below its corner.
    fn low(&mut self, v0: f64) -> f64 {
        let (band, high) = self.tick(v0);
        v0 - self.k * band - high
    }

    /// A new corner and Q, the state kept: a corner that moves.
    fn retune(&mut self, centre_hz: f64, q: f64, sample_rate: f64) {
        let fresh = Self::new(centre_hz, q, sample_rate);
        self.k = fresh.k;
        self.a1 = fresh.a1;
        self.a2 = fresh.a2;
        self.a3 = fresh.a3;
    }

    fn clear(&mut self) {
        self.ic1 = 0.0;
        self.ic2 = 0.0;
    }
}

/// The kick's white noise: `mxm-rng`'s xorshift64*, drawn uniform in [−1, 1). A zero seed is moved
/// to one before the golden-ratio mix, the rule every stored seed was rendered with.
fn white(seed: u64) -> Xorshift64Star {
    Xorshift64Star::from_state(seed.max(1) ^ 0x9E37_79B9_7F4A_7C15)
}

/// A linear rise over `samples`, then held: the attack every section starts with.
fn rise(age: u64, samples: f64) -> f64 {
    ((age as f64 + 1.0) / samples.max(1.0)).min(1.0)
}

/// A sine with a pitch drop and a decay in one or two stages: the second oscillator and each partial.
#[derive(Debug, Clone, Copy, Default)]
struct Sine {
    level: f64,
    hz: f64,
    env: f64,
    fall: f64,
    tail: f64,
    tail_env: f64,
    tail_fall: f64,
    phase: f64,
    /// Its own attack, samples (0: none of its own); the level holds until it is over.
    attack_samples: f64,
}

impl Sine {
    fn start(level: f64, hz: f64, t60: f64, sample_rate: f64) -> Self {
        Self::line(
            level,
            hz,
            &Partial {
                t60_s: t60,
                ..Partial::default()
            },
            sample_rate,
        )
    }

    /// A partial's line at `level` and `hz`.
    fn line(level: f64, hz: f64, p: &Partial, sample_rate: f64) -> Self {
        Self {
            level,
            hz,
            env: 1.0,
            fall: fall(p.t60_s, sample_rate),
            tail: p.tail.clamp(0.0, 1.0),
            tail_env: 1.0,
            tail_fall: fall(p.tail_s, sample_rate),
            phase: p.phase,
            attack_samples: (p.attack_s * sample_rate).max(0.0),
        }
    }

    /// One sample at pitch factor `1 + k·e^(−t/τ)`, `age` samples after the strike.
    fn tick(&mut self, pitch: f64, dt: f64, age: u64) -> f64 {
        if self.level <= 0.0 {
            return 0.0;
        }
        let r = if self.attack_samples > 0.0 {
            rise(age, self.attack_samples)
        } else {
            1.0
        };
        let out = self.level * r * self.envelope() * self.phase.sin();
        self.phase += TAU * self.hz * pitch * dt;
        if r >= 1.0 {
            self.env *= self.fall;
            self.tail_env *= self.tail_fall;
        }
        out
    }

    fn envelope(&self) -> f64 {
        (1.0 - self.tail) * self.env + self.tail * self.tail_env
    }

    fn loudness(&self) -> f64 {
        self.level * self.envelope()
    }
}

/// Band-passed noise with its envelope, running.
#[derive(Debug, Clone, Copy)]
struct NoiseRun {
    level: f64,
    filter: Svf,
    white: Xorshift64Star,
    env: f64,
    fall: f64,
    /// The second stage: its share, level and fall.
    tail: f64,
    tail_env: f64,
    tail_fall: f64,
    attack_samples: f64,
}

impl NoiseRun {
    fn start(spec: &Noise, velocity: f64, seed: u64, sample_rate: f64) -> Self {
        Self {
            level: routed(spec.level, spec.vel_level, velocity) * velocity,
            filter: Svf::new(routed(spec.hz, spec.vel_hz, velocity), spec.q, sample_rate),
            white: white(seed),
            env: 1.0,
            fall: fall(spec.t60_s, sample_rate),
            tail: spec.tail.clamp(0.0, 1.0),
            tail_env: 1.0,
            tail_fall: fall(spec.tail_s, sample_rate),
            attack_samples: (spec.attack_s * sample_rate).max(1.0),
        }
    }

    fn tick(&mut self, age: u64) -> f64 {
        if self.level <= 0.0 {
            return 0.0;
        }
        let n = self.white.bipolar_f64();
        let r = rise(age, self.attack_samples);
        let out = self.level * r * self.envelope() * self.filter.band(n);
        if r >= 1.0 {
            self.env *= self.fall;
            self.tail_env *= self.tail_fall;
        }
        out
    }

    fn envelope(&self) -> f64 {
        (1.0 - self.tail) * self.env + self.tail * self.tail_env
    }

    fn loudness(&self) -> f64 {
        self.level * self.envelope()
    }
}

/// The editor's controls over a preset (the owner, 2026-09-30: "natural synth parameters like mix of
/// the three waveforms and decay on each. Attack, Click, pitch decays etc."): a handful a user turns,
/// each neutral at the preset's own sound, mapped by `apply` onto the engine's many settings. A pure
/// function, so it runs whenever a control moves; the tables are rebuilt only when their overtones
/// change, which no control does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Controls {
    /// Semitones: every table's pitch, the body's, and the animated filter's rings with them.
    pub tune: f64,
    /// dB: each table's level — the three waveforms' mix (body, second skin, long ring).
    pub mix_db: [f64; TABLES],
    /// Times: each table's decays, both stages.
    pub decay: [f64; TABLES],
    /// Times: the glide's depth and its time, every table's and the body's.
    pub pitch_depth: f64,
    pub pitch_time: f64,
    /// Times: the strike's hardness — each doubling halves the tables' attacks and lifts the contact
    /// 6 dB (**chosen**).
    pub attack: f64,
    /// dB: the click's level.
    pub click_db: f64,
    /// dB: the animated filter's level; times: its decays (the broad band's and the rings').
    pub noise_db: f64,
    pub noise_decay: f64,
    /// Times: brightness — the tables' low-pass corners, and each doubling tilts the noise 3 dB an
    /// octave up (**chosen**).
    pub tone: f64,
    /// dB: a snare's wires' level; times: their fall (both stages) and their return. A softer strike
    /// launches them less far: they return sooner and fall sooner (see `Wires`).
    pub wires_db: f64,
    pub wires_decay: f64,
    pub wires_delay: f64,
}

/// The controls at a middle strike (`SynthKickSpec::middle`): at `velocity` a strike plays
/// `controls`; above it a strike sits between the preset and them, below it between them and the
/// softest strike's. Off at `velocity` 0, where a strike sits between the preset and the softest's
/// in one line. Set to that line's own controls at its velocity, it plays as off, but for rounding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Middle {
    pub velocity: f64,
    pub controls: Controls,
}

impl Default for Middle {
    /// Off.
    fn default() -> Self {
        Self {
            velocity: 0.0,
            controls: Controls::default(),
        }
    }
}

impl Default for Controls {
    /// Neutral: the preset as it is.
    fn default() -> Self {
        Self {
            tune: 0.0,
            mix_db: [0.0; TABLES],
            decay: [1.0; TABLES],
            pitch_depth: 1.0,
            pitch_time: 1.0,
            attack: 1.0,
            click_db: 0.0,
            noise_db: 0.0,
            noise_decay: 1.0,
            tone: 1.0,
            wires_db: 0.0,
            wires_decay: 1.0,
            wires_delay: 1.0,
        }
    }
}

impl Controls {
    /// These controls moved a share `u` (0–1) of the way toward `other`: levels and semitones in a
    /// line, ratios geometrically.
    #[must_use]
    pub fn toward(&self, other: &Self, u: f64) -> Self {
        let u = u.clamp(0.0, 1.0);
        let add = |a: f64, b: f64| a + u * (b - a);
        let mul = |a: f64, b: f64| a * (b / a.max(1e-9)).max(1e-9).powf(u);
        Self {
            tune: add(self.tune, other.tune),
            mix_db: std::array::from_fn(|i| add(self.mix_db[i], other.mix_db[i])),
            decay: std::array::from_fn(|i| mul(self.decay[i], other.decay[i])),
            pitch_depth: add(self.pitch_depth, other.pitch_depth),
            pitch_time: mul(self.pitch_time, other.pitch_time),
            attack: mul(self.attack, other.attack),
            click_db: add(self.click_db, other.click_db),
            noise_db: add(self.noise_db, other.noise_db),
            noise_decay: mul(self.noise_decay, other.noise_decay),
            tone: mul(self.tone, other.tone),
            wires_db: add(self.wires_db, other.wires_db),
            wires_decay: mul(self.wires_decay, other.wires_decay),
            wires_delay: mul(self.wires_delay, other.wires_delay),
        }
    }

    /// These controls' distance from neutral taken `k` times (0 neutral, 1 themselves, past 1 further
    /// the same way): levels and semitones in a line, ratios geometrically.
    #[must_use]
    pub fn scaled(&self, k: f64) -> Self {
        if k == 1.0 {
            return *self;
        }
        let k = k.max(0.0);
        let mul = |a: f64| a.max(1e-9).powf(k);
        Self {
            tune: self.tune * k,
            mix_db: self.mix_db.map(|d| d * k),
            decay: self.decay.map(mul),
            pitch_depth: (1.0 + k * (self.pitch_depth - 1.0)).max(0.0),
            pitch_time: mul(self.pitch_time),
            attack: mul(self.attack),
            click_db: self.click_db * k,
            noise_db: self.noise_db * k,
            noise_decay: mul(self.noise_decay),
            tone: mul(self.tone),
            wires_db: self.wires_db * k,
            wires_decay: mul(self.wires_decay),
            wires_delay: mul(self.wires_delay),
        }
    }

    /// These controls and `other` together: levels and semitones add, ratios multiply. Neutral is
    /// the identity either way round, exactly.
    #[must_use]
    pub fn and(&self, other: &Self) -> Self {
        Self {
            tune: self.tune + other.tune,
            mix_db: std::array::from_fn(|i| self.mix_db[i] + other.mix_db[i]),
            decay: std::array::from_fn(|i| self.decay[i] * other.decay[i]),
            pitch_depth: self.pitch_depth * other.pitch_depth,
            pitch_time: self.pitch_time * other.pitch_time,
            attack: self.attack * other.attack,
            click_db: self.click_db + other.click_db,
            noise_db: self.noise_db + other.noise_db,
            noise_decay: self.noise_decay * other.noise_decay,
            tone: self.tone * other.tone,
            wires_db: self.wires_db + other.wires_db,
            wires_decay: self.wires_decay * other.wires_decay,
            wires_delay: self.wires_delay * other.wires_delay,
        }
    }

    /// The preset under these controls. Neutral controls return it unchanged.
    #[must_use]
    pub fn apply(&self, preset: &SynthKickSpec) -> SynthKickSpec {
        let mut s = *preset;
        self.apply_to(&mut s);
        s
    }

    /// [`Controls::apply`] in place: a strike moves the voice's own copy rather than building one
    /// on the stack, where a patch with a cymbal's metal is ten kilobytes a copy (Big Rusty's kit
    /// overflowed a host's audio thread, 2026-10-16). Neutral controls leave it unchanged.
    pub fn apply_to(&self, s: &mut SynthKickSpec) {
        if *self == Self::default() {
            return;
        }
        let pitch = 2f64.powf(self.tune / 12.0);
        let gain = |db: f64| 10f64.powf(db / 20.0);
        let hardness = self.attack.max(1e-3);
        let tone = self.tone.max(1e-3);
        s.body.hz *= pitch;
        s.body.drop *= self.pitch_depth;
        s.body.drop_s *= self.pitch_time;
        s.second.hz *= pitch;
        for (i, t) in s.tables.iter_mut().enumerate() {
            t.hz *= pitch;
            t.level *= gain(self.mix_db[i]);
            t.t60_s *= self.decay[i];
            t.tail_s *= self.decay[i];
            t.drop *= self.pitch_depth;
            t.drop_s *= self.pitch_time;
            t.drop2 *= self.pitch_depth;
            t.drop2_s *= self.pitch_time;
            t.attack_s /= hardness;
            t.lp_from_hz = (t.lp_from_hz * tone).min(20_000.0);
            t.lp_to_hz = (t.lp_to_hz * tone).min(20_000.0);
        }
        s.click.level *= gain(self.click_db);
        if s.metal.level > 0.0 {
            let m = &mut s.metal;
            m.rate *= pitch;
            m.strike *= hardness;
            let edges: [(f64, f64); METAL_BANDS] = std::array::from_fn(|b| m.edges(b));
            let floor_gain = gain(self.noise_db).powi(2);
            for (b, band) in m.bands.iter_mut().enumerate() {
                // Three bands to each of the four groups the mix and decay controls name.
                let g = b * TABLES / METAL_BANDS;
                band.stretch *= self.decay[g];
                let centre = (edges[b].0 * edges[b].1).sqrt();
                let shift = self.mix_db[g] + 3.0 * tone.log2() * (centre / 1000.0).log2();
                for db in &mut band.curve_db {
                    *db += shift;
                }
                // The strike: its first two points.
                let lift = 6.0 * hardness.log2();
                band.curve_db[0] += lift;
                band.curve_db[1] += lift;
                // Noise against partials, the band's power kept.
                let f = band.floor;
                band.floor = f * floor_gain / (f * floor_gain + 1.0 - f).max(1e-12);
            }
        }
        let b = &mut s.bank;
        b.level *= gain(self.noise_db);
        b.broad.t60_s *= self.noise_decay;
        b.broad.tail_s *= self.noise_decay;
        b.broad.slope_db += 3.0 * tone.log2();
        for r in &mut b.rings {
            r.from_hz *= pitch;
            r.to_hz *= pitch;
            r.t60_s *= self.noise_decay;
            r.tail_s *= self.noise_decay;
        }
        for db in &mut b.contact.level_db {
            if *db > BANK_OFF_DB {
                *db += 6.0 * hardness.log2();
            }
        }
        let w = &mut b.wires;
        if w.band.level_db > BANK_OFF_DB {
            w.band.level_db += self.wires_db;
        }
        w.band.t60_s *= self.wires_decay;
        w.band.tail_s *= self.wires_decay;
        w.band.slope_db += 3.0 * tone.log2();
        w.delay_s *= self.wires_delay;
    }
}

/// The sections' own seeds, derived from the patch's.
const CLICK_SEED: u64 = 0;
const NOISE_SEED: u64 = 0x5851_F42D_4C95_7F2D;
const NOISE2_SEED: u64 = 0x2545_F491_4F6C_DD1D;
const BANK_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const METAL_SEED: u64 = 0x94D0_49BB_1331_11EB;
/// The metal's rattle's, from the metal's.
const RATTLE_SEED: u64 = 0xBF58_476D_1CE4_E5B9;
/// How far each strike moves the seed with `fresh_noise`: an odd constant, so no two strikes of the
/// first 2^64 share one.
const STRIKE_SEED: u64 = 0xD1B5_4A32_D192_ED03;

/// A synthesized kick voice.
#[derive(Debug, Clone)]
pub struct SynthKick {
    /// The patch as given; `spec` is it under the velocity's controls, set at each strike.
    preset: SynthKickSpec,
    spec: SynthKickSpec,
    /// A strike's working copies, the patch read at its velocity and that under the controls: held
    /// here, on the heap with the voice, not built on the audio thread's stack at every strike.
    at: SynthKickSpec,
    work: SynthKickSpec,
    sample_rate: f64,
    dt: f64,
    velocity: f64,
    age: u64,
    /// Strikes since the voice was built: `fresh_noise` moves the seed by them.
    strikes: u64,
    // The body.
    body_level: f64,
    body_drop: f64,
    body_drop_env: f64,
    body_drop_fall: f64,
    body_phase: f64,
    body_first: f64,
    body_first_fall: f64,
    body_second: f64,
    body_second_fall: f64,
    // The second oscillator and the partials, each with their drop.
    second: Sine,
    second_drop_env: f64,
    second_drop_fall: f64,
    partials: [Sine; PARTIALS],
    partial_drop_env: f64,
    partial_drop_fall: f64,
    // The click and the noise.
    click: NoiseRun,
    noise: NoiseRun,
    noise2: NoiseRun,
    bank: BankRun,
    tables: Vec<TableRun>,
    metal: MetalRun,
    low_cut: Option<Svf>,
    active: bool,
}

impl SynthKick {
    /// Build the voice at a sample rate, its tables with it. It is silent until struck.
    pub fn new(spec: SynthKickSpec, sample_rate: f64) -> Self {
        let periods = spec.tables.map(|t| {
            if t.level > 0.0 {
                Arc::from(table_period(&t))
            } else {
                Arc::from(Vec::new())
            }
        });
        let readings = std::array::from_fn(|n| table_readings(&spec, n));
        let metal = if spec.metal.level > 0.0 {
            metal_tables(&spec.metal)
        } else {
            Vec::new()
        };
        Self::build(spec, periods, readings, metal, sample_rate)
    }

    /// The same voice at another sample rate, silent, its tables shared rather than built again
    /// (they do not depend on the rate). Allocates: not on the audio thread.
    #[must_use]
    pub fn at_rate(&self, sample_rate: f64) -> Self {
        let periods = std::array::from_fn(|i| Arc::clone(&self.tables[i].period));
        let readings = std::array::from_fn(|i| self.tables[i].readings.clone());
        Self::build(
            self.preset,
            periods,
            readings,
            self.metal.tables.clone(),
            sample_rate,
        )
    }

    /// The voice under `spec`, silent, this voice's metal tables shared rather than built again:
    /// `spec`'s metal must hold this voice's tones, edges and seed (its levels, curves and tick may
    /// differ), and its drum tables are built from `spec`. A patch's parts are rendered alone this way
    /// offline (the synth fit's `stems` and `solve`), each without the metal's transforms. Allocates:
    /// not on the audio thread.
    #[must_use]
    pub fn sharing_metal(&self, spec: SynthKickSpec) -> Self {
        let periods = spec.tables.map(|t| {
            if t.level > 0.0 {
                Arc::from(table_period(&t))
            } else {
                Arc::from(Vec::new())
            }
        });
        let readings = std::array::from_fn(|n| table_readings(&spec, n));
        Self::build(
            spec,
            periods,
            readings,
            self.metal.tables.clone(),
            self.sample_rate,
        )
    }

    fn build(
        spec: SynthKickSpec,
        periods: [Arc<[f32]>; TABLES],
        readings: [Vec<Arc<[f32]>>; TABLES],
        metal: Vec<Vec<MetalTable>>,
        sample_rate: f64,
    ) -> Self {
        let low_cut =
            (spec.low_cut_hz > 0.0).then(|| Svf::new(spec.low_cut_hz, FRAC_1_SQRT_2, sample_rate));
        let idle = NoiseRun::start(&spec.click, 0.0, spec.seed, sample_rate);
        Self {
            preset: spec,
            spec,
            at: spec,
            work: spec,
            sample_rate,
            dt: 1.0 / sample_rate,
            velocity: 0.0,
            age: 0,
            strikes: 0,
            body_level: 0.0,
            body_drop: 0.0,
            body_drop_env: 0.0,
            body_drop_fall: 0.0,
            body_phase: 0.0,
            body_first: 0.0,
            body_first_fall: 0.0,
            body_second: 0.0,
            body_second_fall: 0.0,
            second: Sine::default(),
            second_drop_env: 0.0,
            second_drop_fall: 0.0,
            partials: [Sine::default(); PARTIALS],
            partial_drop_env: 0.0,
            partial_drop_fall: 0.0,
            click: idle,
            noise: idle,
            noise2: idle,
            bank: BankRun::new(sample_rate),
            tables: spec
                .tables
                .iter()
                .zip(periods)
                .zip(readings)
                .map(|((t, period), readings)| TableRun::new(t, period, readings, sample_rate))
                .collect(),
            metal: MetalRun::new(metal),
            low_cut,
            active: false,
        }
    }

    /// Strike at `velocity` (0–1): every section starts again. Allocates nothing.
    pub fn strike(&mut self, velocity: f64) {
        self.strike_with(&Controls::default(), 1.0, velocity);
    }

    /// Strike at `velocity` (0–1) under the player's `controls`, the velocity's own controls reaching
    /// `reach` times as far as the patch's (0: velocity moves the level alone). Allocates nothing.
    pub fn strike_with(&mut self, controls: &Controls, reach: f64, velocity: f64) {
        let v = velocity.clamp(0.0, 1.0);
        let soft = self.preset.velocity.scaled(reach);
        let m = &self.preset.middle;
        let struck = if m.velocity > 0.0 && m.velocity < 1.0 {
            let middle = m.controls.scaled(reach);
            if v >= m.velocity {
                Controls::default().toward(&middle, (1.0 - v) / (1.0 - m.velocity))
            } else {
                middle.toward(&soft, (m.velocity - v) / m.velocity)
            }
        } else {
            Controls::default().toward(&soft, 1.0 - v)
        };
        self.spec = self.preset;
        controls.and(&struck).apply_to(&mut self.spec);
        if self.preset.metal.level > 0.0 && self.preset.metal.reads_softer() {
            // The metal's own readings carry the strike's balance, its tables' levels and decays
            // with it; the velocity's controls would move them twice. The player's controls still
            // reach them, and the strike's attack and tone the tables.
            self.at = self.preset;
            self.at.metal = self.preset.metal.at_velocity(v);
            for (t, (db, decay)) in self
                .at
                .tables
                .iter_mut()
                .zip(self.preset.metal.tables_at(v))
            {
                t.level *= 10f64.powf(db / 20.0);
                t.t60_s *= decay;
                t.tail_s *= decay;
            }
            let read = Controls {
                mix_db: [0.0; TABLES],
                decay: [1.0; TABLES],
                ..struck
            };
            self.work = self.at;
            controls.and(&read).apply_to(&mut self.work);
            self.spec.tables = self.work.tables;
            self.work = self.at;
            controls.apply_to(&mut self.work);
            self.spec.metal = self.work.metal;
        }
        let fs = self.sample_rate;
        let dt = self.dt;
        let drop_fall = |seconds: f64| (-dt / seconds.max(1e-4)).exp();
        self.velocity = v;
        self.age = 0;

        let b = &self.spec.body;
        let decay = routed(1.0, b.vel_decay, v);
        self.body_level = b.level;
        self.body_drop = routed(b.drop, b.vel_drop, v);
        self.body_drop_env = 1.0;
        self.body_drop_fall = drop_fall(b.drop_s);
        self.body_phase = b.phase;
        self.body_first = 1.0 - b.tail.clamp(0.0, 1.0);
        self.body_first_fall = fall(b.decay_s * decay, fs);
        self.body_second = b.tail.clamp(0.0, 1.0);
        self.body_second_fall = fall(b.tail_s * decay, fs);

        let t = &self.spec.second;
        self.second = Sine::start(routed(t.level, t.vel_level, v), t.hz, t.t60_s, fs);
        self.second_drop_env = 1.0;
        self.second_drop_fall = drop_fall(t.drop_s);

        let p = &self.spec.partials;
        let scale = routed(1.0, p.vel_level, v);
        for (sine, partial) in self.partials.iter_mut().zip(&p.partials) {
            *sine = Sine::line(
                partial.level * scale,
                partial.ratio * self.spec.body.hz,
                partial,
                fs,
            );
        }
        self.partial_drop_env = 1.0;
        self.partial_drop_fall = drop_fall(p.drop_s);

        let seed = if self.spec.fresh_noise {
            self.spec
                .seed
                .wrapping_add(self.strikes.wrapping_mul(STRIKE_SEED))
        } else {
            self.spec.seed
        };
        self.strikes = self.strikes.wrapping_add(1);
        self.click = NoiseRun::start(&self.spec.click, v, seed.wrapping_add(CLICK_SEED), fs);
        self.noise = NoiseRun::start(&self.spec.noise, v, seed.wrapping_add(NOISE_SEED), fs);
        self.noise2 = NoiseRun::start(&self.spec.noise2, v, seed.wrapping_add(NOISE2_SEED), fs);
        self.bank
            .start(&self.spec.bank, v, seed.wrapping_add(BANK_SEED));
        for (run, table) in self.tables.iter_mut().zip(&self.spec.tables) {
            run.start(table, v, fs);
        }
        if self.preset.metal.level > 0.0 && self.preset.metal.reads_softer() {
            let (sets, w) = self.preset.metal.tones_at(v);
            for run in &mut self.tables {
                run.choose(sets, w);
            }
        }
        self.metal.start(
            &self.spec.metal,
            v,
            seed.wrapping_add(METAL_SEED),
            self.strikes,
            fs,
        );
        if let Some(cut) = &mut self.low_cut {
            cut.clear();
        }
        self.active = true;
    }

    /// The body's next sample, after its drive.
    fn body(&mut self) -> f64 {
        let b = &self.spec.body;
        if self.body_level <= 0.0 {
            return 0.0;
        }
        let hz = b.hz * (1.0 + self.body_drop * self.body_drop_env);
        let sine = self.body_phase.sin();
        let wave = if b.shape > 0.0 {
            // A triangle through the same zero crossings and peaks as the sine.
            let p = self.body_phase.rem_euclid(TAU) / TAU;
            let tri = if p < 0.25 {
                4.0 * p
            } else if p < 0.75 {
                2.0 - 4.0 * p
            } else {
                4.0 * p - 4.0
            };
            sine + b.shape.clamp(0.0, 1.0) * (tri - sine)
        } else {
            sine
        };
        let attack = rise(self.age, b.attack_s * self.sample_rate);
        let x =
            self.velocity * self.body_level * attack * (self.body_first + self.body_second) * wave;
        self.body_phase += TAU * hz * self.dt;
        self.body_drop_env *= self.body_drop_fall;
        self.body_first *= self.body_first_fall;
        self.body_second *= self.body_second_fall;
        if b.drive > 0.0 {
            (b.drive * x).tanh() / b.drive
        } else {
            x
        }
    }

    fn body_loudness(&self) -> f64 {
        self.body_level * (self.body_first + self.body_second)
    }

    /// One sample of the mix.
    pub fn tick(&mut self) -> f32 {
        if !self.active {
            return 0.0;
        }
        let (dt, age, v) = (self.dt, self.age, self.velocity);
        let body = self.body();

        let second_attack = rise(age, self.spec.second.attack_s * self.sample_rate);
        let second_pitch = 1.0 + self.spec.second.drop * self.second_drop_env;
        let second = v * second_attack * self.second.tick(second_pitch, dt, age);
        self.second_drop_env *= self.second_drop_fall;

        let partial_attack = rise(age, self.spec.partials.attack_s * self.sample_rate);
        let partial_pitch = 1.0 + self.spec.partials.drop * self.partial_drop_env;
        let partials: f64 = self
            .partials
            .iter_mut()
            .map(|p| p.tick(partial_pitch, dt, age))
            .sum::<f64>()
            * v
            * partial_attack;
        self.partial_drop_env *= self.partial_drop_fall;

        let click = self.click.tick(age);
        let noise = self.noise.tick(age) + self.noise2.tick(age);
        let bank = if self.bank.level > 0.0 {
            if age >= self.bank.next_plan {
                self.bank.plan(&self.spec.bank, age, self.sample_rate);
            }
            self.bank.tick()
        } else {
            0.0
        };
        let fs = self.sample_rate;
        let tables: f64 = self
            .tables
            .iter_mut()
            .zip(&self.spec.tables)
            .map(|(run, table)| run.tick(table, age, fs))
            .sum();
        let mut out = body + second + partials + click + noise + bank + tables;
        if self.metal.active {
            out += v * self.metal.tick(age);
        }
        self.age += 1;

        if let Some(cut) = &mut self.low_cut {
            out = cut.tick(out).1;
        }
        let out = self.spec.output_gain * out;

        let loudest = self
            .body_loudness()
            .max(self.second.loudness())
            .max(self.partials.iter().map(Sine::loudness).fold(0.0, f64::max))
            .max(self.click.loudness())
            .max(self.noise.loudness())
            .max(self.noise2.loudness())
            .max(self.bank.loudness())
            .max(self.metal.loudness())
            .max(
                self.tables
                    .iter()
                    .map(TableRun::loudness)
                    .fold(0.0, f64::max),
            );
        let attacks = [
            self.spec.body.attack_s,
            self.spec.second.attack_s,
            self.spec.partials.attack_s,
            self.spec.noise.attack_s,
            self.spec.noise2.attack_s,
            self.spec.bank.rises(),
            self.spec
                .tables
                .iter()
                .map(|t| t.attack_s)
                .fold(0.0, f64::max),
            self.spec
                .partials
                .partials
                .iter()
                .map(|p| p.attack_s)
                .fold(0.0, f64::max),
        ]
        .iter()
        .fold(0.0f64, |m, a| m.max(*a))
            * self.sample_rate;
        // Parked once every section has faded and the attacks are over.
        if loudest * v < PARK_LEVEL && out.abs() < PARK_LEVEL && (age as f64) > attacks {
            self.active = false;
            return 0.0;
        }
        out as f32
    }

    /// Fill `out`.
    pub fn render(&mut self, out: &mut [f32]) {
        for sample in out {
            *sample = self.tick();
        }
    }

    /// Whether the voice is sounding.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Silence at once: panic, not a musical gesture.
    pub fn reset(&mut self) {
        self.active = false;
        self.click.filter.clear();
        self.noise.filter.clear();
        self.noise2.filter.clear();
        for band in &mut self.bank.bands[..self.bank.count] {
            band.filter.clear();
        }
        for run in &mut self.tables {
            run.lp.clear();
            run.level = 0.0;
        }
        self.metal.active = false;
        if let Some(cut) = &mut self.low_cut {
            cut.clear();
        }
    }
}
