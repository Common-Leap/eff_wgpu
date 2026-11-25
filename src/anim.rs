//! Emitter animations: values of the emitter itself that run on key frames over its time.
//!
//! EmitterResource::ResolveBinaryData files each section (EAES, EAER, ...) under a slot, and
//! EmitterCalculator::Calculate (Smash 0x8450) evaluates every slot that has one into the
//! emitter's values (emitter+0x600, three floats a slot) before it places the emitter, emits or
//! moves anything. The rest of the runtime reads those values instead of the resource's.

use glam::{Mat4, Vec3};

fn f32_at(b: &[u8], at: usize) -> f32 {
    b.get(at..at + 4)
        .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
        .unwrap_or(0.0)
}

fn i32_at(b: &[u8], at: usize) -> i32 {
    b.get(at..at + 4)
        .map(|s| i32::from_le_bytes(s.try_into().unwrap()))
        .unwrap_or(0)
}

fn vec3_at(b: &[u8], at: usize) -> Vec3 {
    Vec3::new(f32_at(b, at), f32_at(b, at + 4), f32_at(b, at + 8))
}

/// What a slot's value is, in the runtime's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Slot {
    /// The emitter's scale in its set.
    Scale,
    /// Its rotation: degrees in the section, radians in the resource.
    Rotation,
    Translation,
    Colour0,
    Colour1,
    /// Particles an emission (x).
    Rate,
    /// Particle life in frames (x), never past the resource's.
    Life,
    Alpha0,
    Alpha1,
    /// Outward speed (x).
    Outward,
    /// Speed along the designated direction (x).
    Directed,
    /// A new particle's base scale.
    ParticleScale,
    /// The emission shape's scale.
    ShapeScale,
    /// Gravity's strength (x).
    Gravity,
}

const SECTIONS: [(&[u8; 4], Slot); 14] = [
    (b"EAES", Slot::Scale),
    (b"EAER", Slot::Rotation),
    (b"EAET", Slot::Translation),
    (b"EAC0", Slot::Colour0),
    (b"EAC1", Slot::Colour1),
    (b"EATR", Slot::Rate),
    (b"EAPL", Slot::Life),
    (b"EAA0", Slot::Alpha0),
    (b"EAA1", Slot::Alpha1),
    (b"EAOV", Slot::Outward),
    (b"EADV", Slot::Directed),
    (b"EASL", Slot::ParticleScale),
    (b"EASS", Slot::ShapeScale),
    (b"EAGV", Slot::Gravity),
];

/// One section (ResAnimEmitterKeyParamSet): an enable byte, a loop byte, the key count at +4 and
/// from +0xc the keys, each a value and the frame it is at.
#[derive(Debug, Clone, PartialEq)]
struct Curve {
    loops: bool,
    keys: Vec<(Vec3, f32)>,
}

impl Curve {
    fn read(d: &[u8]) -> Option<Self> {
        if d.first() == Some(&0) {
            return None;
        }
        let count = usize::try_from(i32_at(d, 4)).ok()?;
        let keys: Vec<_> = d
            .get(0xc..)?
            .chunks_exact(16)
            .take(count)
            .map(|k| (vec3_at(k, 0), f32_at(k, 12)))
            .collect();
        (!keys.is_empty()).then(|| Self {
            loops: d.get(1).is_some_and(|b| *b != 0),
            keys,
        })
    }

    /// The value at `time` (CalculateEmitterKeyFrameAnimation, Smash 0xd9c0): a curve of one key
    /// is that key. A looping curve wraps its time at its last key. Before the first key it is
    /// that key's value, from the last key on the last's, and between two keys a straight line.
    fn at(&self, time: f32) -> Vec3 {
        let (first, last) = (self.keys[0], self.keys[self.keys.len() - 1]);
        if self.keys.len() == 1 {
            return first.0;
        }
        let time = if self.loops && last.1 > 0.0 {
            time % last.1
        } else {
            time
        };
        if time < first.1 {
            return first.0;
        }
        if last.1 <= time {
            return last.0;
        }
        self.keys
            .windows(2)
            .find(|k| k[0].1 <= time && time < k[1].1)
            .map(|k| k[0].0 + (k[1].0 - k[0].0) * ((time - k[0].1) / (k[1].1 - k[0].1)))
            .unwrap_or(last.0)
    }
}

/// An emitter's animation sections.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Anims {
    curves: [Option<Curve>; 14],
}

impl Anims {
    /// The animations an emitter's attribute sections hold.
    pub fn of(attributes: &[([u8; 4], Vec<u8>)]) -> Self {
        let mut anims = Self::default();
        for (magic, slot) in SECTIONS {
            anims.curves[slot as usize] = attributes
                .iter()
                .find(|(m, _)| m == magic)
                .and_then(|(_, d)| Curve::read(d));
        }
        anims
    }

    pub fn is_empty(&self) -> bool {
        self.curves.iter().all(Option::is_none)
    }

    /// Whether the emitter's place, turn or size is animated. Its matrix is then built from its
    /// values, the ones that are not animated without the resource's random spreads
    /// (UpdateEmitterMatrix, Smash 0x8090).
    pub fn moves_emitter(&self) -> bool {
        self.curves[..=Slot::Translation as usize]
            .iter()
            .any(Option::is_some)
    }
}

/// The emitter's values (emitter+0x600), as Emitter::Initialize takes them from the resource.
#[derive(Debug, Clone, PartialEq)]
pub struct Values([Vec3; 14]);

impl Values {
    pub fn of(raw: &[u8]) -> Self {
        let x = |at: usize| Vec3::new(f32_at(raw, at), 0.0, 0.0);
        let mut values = [Vec3::ZERO; 14];
        for (slot, value) in [
            (Slot::Scale, vec3_at(raw, 0x7a0)),
            (Slot::Rotation, vec3_at(raw, 0x788)),
            (Slot::Translation, vec3_at(raw, 0x770)),
            (Slot::Colour0, vec3_at(raw, 0x7ac)),
            (Slot::Colour1, vec3_at(raw, 0x7bc)),
            (Slot::Rate, x(0x800)),
            (Slot::Life, Vec3::new(i32_at(raw, 0x8b8) as f32, 0.0, 0.0)),
            (Slot::Alpha0, x(0x7b8)),
            (Slot::Alpha1, x(0x7c8)),
            (Slot::Outward, x(0x964)),
            (Slot::Directed, x(0x968)),
            (Slot::ParticleScale, vec3_at(raw, 0x9c0)),
            (Slot::ShapeScale, vec3_at(raw, 0x868)),
            (Slot::Gravity, x(0x814)),
        ] {
            values[slot as usize] = value;
        }
        Self(values)
    }

    pub fn get(&self, slot: Slot) -> Vec3 {
        self.0[slot as usize]
    }

    /// A slot that holds one number.
    pub fn x(&self, slot: Slot) -> f32 {
        self.0[slot as usize].x
    }

    /// Run `anims` at `time`.
    pub fn run(&mut self, anims: &Anims, time: f32, raw: &[u8]) {
        for (curve, value) in anims.curves.iter().zip(&mut self.0) {
            if let Some(curve) = curve {
                *value = curve.at(time);
            }
        }
        if !anims.is_empty() {
            let life = &mut self.0[Slot::Life as usize].x;
            *life = life.min(i32_at(raw, 0x8b8) as f32);
        }
    }

    /// The emitter's matrix in its set, with scale and without (ApplyEmitterAnimation): the
    /// columns of T * Rz * Ry * Rx * S, as Emitter::CreateResMatrix builds the resource's.
    pub fn matrix(&self, anims: &Anims) -> (Mat4, Mat4) {
        let turn = match &anims.curves[Slot::Rotation as usize] {
            Some(_) => self.get(Slot::Rotation) * (std::f32::consts::PI / 180.0),
            None => self.get(Slot::Rotation),
        };
        let turn = Mat4::from_euler(glam::EulerRot::ZYX, turn.z, turn.y, turn.x);
        let rt = Mat4::from_translation(self.get(Slot::Translation)) * turn;
        (rt * Mat4::from_scale(self.get(Slot::Scale)), rt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section(enabled: u8, loops: u8, keys: &[[f32; 4]]) -> Vec<u8> {
        let mut d = vec![enabled, loops, 0, 0];
        d.extend((keys.len() as u32).to_le_bytes());
        d.extend(0u32.to_le_bytes());
        d.extend(keys.iter().flatten().flat_map(|v| v.to_le_bytes()));
        d
    }

    #[test]
    fn a_curve_holds_its_ends_and_runs_between_its_keys() {
        let keys = [
            [1.0, 0.0, 0.0, 10.0],
            [3.0, 4.0, 0.0, 20.0],
            [0.0; 4].map(|_| 8.0),
        ];
        let curve = Curve::read(&section(1, 0, &keys[..2])).unwrap();
        for (time, value) in [
            (0.0, [1.0, 0.0]),
            (10.0, [1.0, 0.0]),
            (15.0, [2.0, 2.0]),
            (20.0, [3.0, 4.0]),
            (99.0, [3.0, 4.0]),
        ] {
            let got = curve.at(time);
            assert_eq!([got.x, got.y], value, "at {time}");
        }
        // A loop wraps at its last key: 25 is 5, before the first.
        let looped = Curve::read(&section(1, 1, &keys[..2])).unwrap();
        assert_eq!(looped.at(25.0).x, 1.0);
        assert_eq!(looped.at(35.0).x, 2.0);
        // One key is its value whenever; none, or one that is off, animates nothing.
        assert_eq!(
            Curve::read(&section(1, 0, &keys[..1])).unwrap().at(0.0).x,
            1.0
        );
        assert!(Curve::read(&section(1, 0, &[])).is_none());
        assert!(Curve::read(&section(0, 0, &keys)).is_none());
    }

    #[test]
    fn values_start_as_the_resources_and_only_animated_ones_change() {
        let mut raw = vec![0u8; 0xac0];
        let mut put = |at: usize, v: f32| raw[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put(0x7a0, 2.0);
        put(0x800, 5.0);
        put(0x968, 7.0);
        raw[0x8b8..0x8bc].copy_from_slice(&30i32.to_le_bytes());
        let mut values = Values::of(&raw);
        assert_eq!(values.get(Slot::Scale).x, 2.0);
        assert_eq!(values.x(Slot::Life), 30.0);
        let anims = Anims::of(&[
            (
                *b"EATR",
                section(1, 0, &[[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 10.0]]),
            ),
            (*b"EAPL", section(1, 0, &[[90.0, 0.0, 0.0, 0.0]])),
        ]);
        assert!(!anims.moves_emitter());
        values.run(&anims, 5.0, &raw);
        assert_eq!(values.x(Slot::Rate), 5.0);
        values.run(&anims, 2.0, &raw);
        assert_eq!(values.x(Slot::Rate), 2.0);
        // Never past the resource's life.
        assert_eq!(values.x(Slot::Life), 30.0);
        assert_eq!(values.x(Slot::Directed), 7.0);
    }

    #[test]
    fn an_animated_turn_is_in_degrees_and_the_resources_in_radians() {
        let mut raw = vec![0u8; 0xac0];
        for at in [0x7a0, 0x7a4, 0x7a8] {
            raw[at..at + 4].copy_from_slice(&1f32.to_le_bytes());
        }
        let quarter = std::f32::consts::FRAC_PI_2;
        raw[0x790..0x794].copy_from_slice(&quarter.to_le_bytes());
        let turned = |anims: &Anims| {
            let mut values = Values::of(&raw);
            values.run(anims, 0.0, &raw);
            values.matrix(anims).0.transform_point3(Vec3::X)
        };
        // Only the place animated: the turn is the resource's quarter, in radians.
        let moved = Anims::of(&[(*b"EAET", section(1, 0, &[[5.0, 0.0, 0.0, 0.0]]))]);
        assert!(turned(&moved).abs_diff_eq(Vec3::new(5.0, 1.0, 0.0), 1e-5));
        let turn = Anims::of(&[(*b"EAER", section(1, 0, &[[0.0, 0.0, 90.0, 0.0]]))]);
        assert!(turned(&turn).abs_diff_eq(Vec3::Y, 1e-5));
    }
}
