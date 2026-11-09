//! `SynthKick::play_as`: a voice moved onto another's tables plays as a voice built for the patch,
//! bit for bit, and moving it, striking it and rendering it allocate nothing.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use mxm_twf_voice::{Overtone, SynthKick, SynthKickSpec, Table};

/// Counts every allocation and free, each thread its own (the tests run side by side).
struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

fn count() {
    // A thread's last frees, after its counter is gone, are not counted.
    let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count();
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const RATE: f64 = 48_000.0;

/// A patch of two tables at `hz`, their overtones `ratios`, every other section off.
fn patch(hz: f64, ratios: [f64; 3], t60_s: f64) -> SynthKickSpec {
    let mut spec = SynthKickSpec::default();
    spec.body.level = 0.0;
    for (n, slot) in spec.tables.iter_mut().take(2).enumerate() {
        let mut overtones = [Overtone {
            ratio: 1.0,
            level: 0.0,
            phase: 0.0,
        }; mxm_twf_voice::OVERTONES];
        for (i, (o, r)) in overtones.iter_mut().zip(ratios).enumerate() {
            *o = Overtone {
                ratio: r * (n + 1) as f64,
                level: 1.0 / (i + 1) as f64,
                phase: 0.3 * i as f64,
            };
        }
        *slot = Table {
            level: 1.0,
            hz,
            overtones,
            step: 1.0 / 256.0,
            attack_s: 0.002,
            t60_s: t60_s / (n + 1) as f64,
            tail: 0.2,
            tail_s: t60_s * 2.0,
            ..Table::default()
        };
    }
    spec
}

fn rendered(voice: &mut SynthKick, velocity: f64) -> Vec<f32> {
    voice.strike(velocity);
    let mut out = vec![0.0f32; 24_000];
    voice.render(&mut out);
    out
}

#[test]
fn a_voice_moved_onto_another_s_tables_plays_as_one_built_for_the_patch() {
    let a = patch(220.0, [1.0, 3.9, 9.8], 1.5);
    let b = patch(330.0, [1.0, 2.76, 5.4], 0.9);
    // B's tables at another key, its falls shorter: a key the anchor is moved to.
    let mut moved = b;
    for t in &mut moved.tables {
        t.hz *= 1.5;
        t.t60_s /= 1.5;
        t.tail_s /= 1.5;
    }
    let anchor = SynthKick::new(b, RATE);
    let mut built = SynthKick::new(moved, RATE);
    let mut voice = SynthKick::new(a, RATE);
    // Sounding A when it is moved, as a stolen voice is.
    let _ = rendered(&mut voice, 0.8);
    voice.play_as(&moved, &anchor);
    assert_eq!(rendered(&mut voice, 0.7), rendered(&mut built, 0.7));
    // And back.
    let mut own = SynthKick::new(a, RATE);
    let from_a = SynthKick::new(a, RATE);
    voice.play_as(&a, &from_a);
    assert_eq!(rendered(&mut voice, 1.0), rendered(&mut own, 1.0));
}

#[test]
fn moving_striking_and_rendering_allocate_nothing() {
    let a = patch(220.0, [1.0, 3.9, 9.8], 1.5);
    let b = patch(330.0, [1.0, 2.76, 5.4], 0.9);
    let anchors = [SynthKick::new(a, RATE), SynthKick::new(b, RATE)];
    // A keyboard's voice, built off the audio thread from an anchor's: its tables shared.
    let mut voice = anchors[0].clone();
    let mut out = vec![0.0f32; 512];
    let before = allocations();
    for round in 0..8 {
        let (spec, from) = if round % 2 == 0 {
            (b, &anchors[1])
        } else {
            (a, &anchors[0])
        };
        voice.play_as(&spec, from);
        voice.strike(0.5);
        voice.render(&mut out);
    }
    assert_eq!(allocations(), before);
}
