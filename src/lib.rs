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
//!   (`docs/drum-model-fitting.md` §6; an exponential sweep in semitones does not fit a drum
//!   machine's kick). Its level rises linearly over its attack — nothing starts at its peak (§6) —
//!   and falls in two stages, a first with its own T60 and a share `tail` falling with a second.
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
//!
//! **Filters (published).** The band-passes and the low cut are the trapezoidal state-variable filter
//! (`docs/filters/02-topologies.md` §2.2: Simper's linear trapezoidal SVF, Zavalishin's TPT SVF, *The
//! Art of VA Filter Design* ch. 4), prewarped with `g = tan(π·fc/fs)`; the band-pass is its band
//! output times `1/Q`, unity at its centre, and the low cut its high-pass output at `Q = 1/√2`, the
//! second-order Butterworth.
//!
//! **Velocity (chosen law).** Every section's level follows the strike's velocity `v`, and each of a
//! section's routes moves one of its parameters by `route` octaves per unit of velocity about
//! `v = ½`: `p(v) = p·2^(route·(v − ½))`. A section's own values are the middle strike's.

use std::f64::consts::{FRAC_1_SQRT_2, PI, TAU};
use std::sync::Arc;

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

/// The animated filter: white noise through third-octave bands whose levels follow a law, a broad
/// band, the strike's contact and the rings, `level` (linear) on top of it, 0 off. The contact is
/// the strike's alone, so the ring's law is not spent on it (fitted from the strike together, the
/// rings went to +33 dB bursts dying in 3 ms instead of Virtuosity's ring near 560 Hz).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bank {
    pub level: f64,
    pub broad: Broad,
    pub contact: Contact,
    pub rings: [Ring; RINGS],
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
            vel_level: 0.0,
        }
    }
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
        for b in [&self.broad] {
            if b.level_db > BANK_OFF_DB {
                let db = b.level_db
                    + b.slope_db * x
                    + b.curve_db * x * x
                    + rise_db(t, b.rise_s)
                    + stages_db(t, b.t60_s, b.tail, b.tail_s)
                    + b.tilt_db * x * (1.0 - (-t / b.tilt_s.max(1e-4)).exp());
                power += 10f64.powf(db / 10.0);
            }
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
        if power > 0.0 {
            10.0 * power.log10()
        } else {
            BANK_OFF_DB
        }
    }

    fn rises(&self) -> f64 {
        self.rings
            .iter()
            .map(|r| r.rise_s)
            .fold(self.broad.rise_s.max(self.contact.rise_s), f64::max)
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
    white: White,
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
            white: White::new(1),
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
                white: White::new(0x00DD_BA11 + k as u64),
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
                *v += band.filter.tick(band.white.next());
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
                White::new(seed.wrapping_add((k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
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
            let n = band.white.next();
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
/// Tables the voice holds: by default a body, a second skin and a long ring.
pub const TABLES: usize = 3;
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
/// tables without building them again (half a megabyte each).
#[derive(Debug, Clone)]
struct TableRun {
    period: Arc<[f32]>,
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
    fn new(spec: &Table, period: Arc<[f32]>, sample_rate: f64) -> Self {
        Self {
            period,
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
        let (a, b) = (
            f64::from(self.period[i]),
            f64::from(self.period[(i + 1) % TABLE_LEN]),
        );
        let wave = a + frac * (b - a);
        let pitch = 1.0 + spec.drop * self.drop_env + spec.drop2 * self.drop2_env;
        self.phase = (self.phase + self.base * pitch / sample_rate).fract();
        self.drop_env *= self.drop_fall;
        self.drop2_env *= self.drop2_fall;
        let env = rise(age, spec.attack_s * sample_rate) * (self.first + self.second);
        self.first *= self.first_fall;
        self.second *= self.second_fall;
        self.lp.low(self.level * env * wave)
    }

    fn loudness(&self) -> f64 {
        self.level * (self.first + self.second)
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
    /// The recording chain's low cut on the mix, Hz: 0 none.
    pub low_cut_hz: f64,
    /// The noise's seed: a render is reproducible, and a fit can be read at other seeds.
    pub seed: u64,
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
            low_cut_hz: 0.0,
            seed: 1,
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

/// xorshift64*, uniform in [−1, 1).
#[derive(Debug, Clone, Copy)]
struct White(u64);

impl White {
    fn new(seed: u64) -> Self {
        Self(seed.max(1) ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
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
    white: White,
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
            white: White::new(seed),
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
        let n = self.white.next();
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
        }
    }

    /// The preset under these controls. Neutral controls return it unchanged.
    #[must_use]
    pub fn apply(&self, preset: &SynthKickSpec) -> SynthKickSpec {
        if *self == Self::default() {
            return *preset;
        }
        let mut s = *preset;
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
        s
    }
}

/// The sections' own seeds, derived from the patch's.
const CLICK_SEED: u64 = 0;
const NOISE_SEED: u64 = 0x5851_F42D_4C95_7F2D;
const NOISE2_SEED: u64 = 0x2545_F491_4F6C_DD1D;
const BANK_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// A synthesized kick voice.
#[derive(Debug, Clone)]
pub struct SynthKick {
    /// The patch as given; `spec` is it under the velocity's controls, set at each strike.
    preset: SynthKickSpec,
    spec: SynthKickSpec,
    sample_rate: f64,
    dt: f64,
    velocity: f64,
    age: u64,
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
        Self::build(spec, periods, sample_rate)
    }

    /// The same voice at another sample rate, silent, its tables shared rather than built again
    /// (they do not depend on the rate). Allocates: not on the audio thread.
    #[must_use]
    pub fn at_rate(&self, sample_rate: f64) -> Self {
        let periods = std::array::from_fn(|i| Arc::clone(&self.tables[i].period));
        Self::build(self.preset, periods, sample_rate)
    }

    fn build(spec: SynthKickSpec, periods: [Arc<[f32]>; TABLES], sample_rate: f64) -> Self {
        let low_cut =
            (spec.low_cut_hz > 0.0).then(|| Svf::new(spec.low_cut_hz, FRAC_1_SQRT_2, sample_rate));
        let idle = NoiseRun::start(&spec.click, 0.0, spec.seed, sample_rate);
        Self {
            preset: spec,
            spec,
            sample_rate,
            dt: 1.0 / sample_rate,
            velocity: 0.0,
            age: 0,
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
                .map(|(t, period)| TableRun::new(t, period, sample_rate))
                .collect(),
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
        self.spec = controls
            .and(&Controls::default().toward(&soft, 1.0 - v))
            .apply(&self.preset);
        let s = self.spec;
        let fs = self.sample_rate;
        let drop_fall = |seconds: f64| (-self.dt / seconds.max(1e-4)).exp();
        self.velocity = v;
        self.age = 0;

        let b = &s.body;
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

        let t = &s.second;
        self.second = Sine::start(routed(t.level, t.vel_level, v), t.hz, t.t60_s, fs);
        self.second_drop_env = 1.0;
        self.second_drop_fall = drop_fall(t.drop_s);

        let p = &s.partials;
        let scale = routed(1.0, p.vel_level, v);
        for (sine, partial) in self.partials.iter_mut().zip(&p.partials) {
            *sine = Sine::line(
                partial.level * scale,
                partial.ratio * s.body.hz,
                partial,
                fs,
            );
        }
        self.partial_drop_env = 1.0;
        self.partial_drop_fall = drop_fall(p.drop_s);

        self.click = NoiseRun::start(&s.click, v, s.seed.wrapping_add(CLICK_SEED), fs);
        self.noise = NoiseRun::start(&s.noise, v, s.seed.wrapping_add(NOISE_SEED), fs);
        self.noise2 = NoiseRun::start(&s.noise2, v, s.seed.wrapping_add(NOISE2_SEED), fs);
        self.bank.start(&s.bank, v, s.seed.wrapping_add(BANK_SEED));
        for (run, table) in self.tables.iter_mut().zip(&s.tables) {
            run.start(table, v, fs);
        }
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
        self.age += 1;

        let mut out = body + second + partials + click + noise + bank + tables;
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
        if let Some(cut) = &mut self.low_cut {
            cut.clear();
        }
    }
}
