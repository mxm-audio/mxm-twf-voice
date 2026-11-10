//! `SynthKick::park_quiet`, a keyboard's: each part stops once it is under the level asked, the
//! voice once all of it is, never during an attack; what it takes away is under that level. That a
//! voice which never calls it plays as before is the drums' fingerprint's to show (`AGENTS.md`).

use mxm_twf_voice::{Overtone, Partial, SynthKick, SynthKickSpec, Table};

const RATE: f64 = 48_000.0;
const BLOCK: usize = 64;

/// A bar: a table ringing long at `hz`, strike lines dying within a second, every other section
/// off.
fn bar(hz: f64) -> SynthKickSpec {
    let mut spec = SynthKickSpec::default();
    spec.body.level = 0.0;
    let mut overtones = [Overtone {
        ratio: 1.0,
        level: 0.0,
        phase: 0.0,
    }; mxm_twf_voice::OVERTONES];
    overtones[0].level = 1.0;
    spec.tables[0] = Table {
        level: 0.5,
        hz,
        overtones,
        step: 1.0 / 256.0,
        attack_s: 0.001,
        t60_s: 3.0,
        ..Table::default()
    };
    spec.partials.attack_s = 0.001;
    for (i, p) in spec.partials.partials.iter_mut().take(12).enumerate() {
        *p = Partial {
            level: 0.05,
            ratio: 2.7 + 1.9 * i as f64,
            t60_s: 0.1 + 0.05 * i as f64,
            ..Partial::default()
        };
    }
    spec
}

/// `n` samples of `spec` struck, `park` asked between blocks of [`BLOCK`] if given.
fn played(spec: SynthKickSpec, n: usize, park: Option<f64>) -> (Vec<f32>, SynthKick) {
    let mut v = SynthKick::new(spec, RATE);
    v.strike(1.0);
    let mut out = vec![0.0f32; n];
    for block in out.chunks_mut(BLOCK) {
        if let Some(level) = park {
            v.park_quiet(level);
        }
        v.render(block);
    }
    (out, v)
}

#[test]
fn what_it_takes_away_is_under_the_level_asked() {
    let n = 4 * RATE as usize;
    let (whole, _) = played(bar(220.0), n, None);
    for level in [1e-5, 1e-4, 1e-3] {
        let (parked, _) = played(bar(220.0), n, Some(level));
        let apart = whole
            .iter()
            .zip(&parked)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Every part stopped under the level, twelve lines and a table at most.
        assert!(f64::from(apart) < 13.0 * level, "{level}: {apart}");
        // And the strike is the strike: the first 50 ms alike.
        let strike = (0.05 * RATE) as usize;
        assert_eq!(whole[..strike], parked[..strike], "{level}");
    }
}

#[test]
fn the_voice_parks_once_all_of_it_is_under_the_level_and_never_in_an_attack() {
    // Asked at a level over the whole voice: it parks at the first block after its attacks.
    let mut slow = bar(220.0);
    slow.tables[0].attack_s = 0.1;
    let mut v = SynthKick::new(slow, RATE);
    v.strike(1.0);
    let mut out = [0.0f32; BLOCK];
    let mut t = 0;
    while v.is_active() && t < RATE as usize {
        v.park_quiet(10.0);
        v.render(&mut out);
        t += BLOCK;
    }
    assert!(!v.is_active());
    let rise = (0.1 * RATE) as usize;
    assert!(t > rise && t <= rise + 2 * BLOCK, "parked at {t}");
    // A level under all of it parks nothing that sounds: the voice rings on.
    let (_, v) = played(bar(220.0), RATE as usize, Some(1e-9));
    assert!(v.is_active());
}
