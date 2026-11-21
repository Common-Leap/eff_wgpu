//! The runtime's random numbers (nn::vfx::detail::Random): where each spawn of an effect gets
//! its seeds, the generator every emitter draws its particles from, and the two tables of random
//! vectors they read alongside it.

use std::sync::OnceLock;

use glam::Vec3;

/// The runtime's xorshift128 (Random::GetGlobalRandom): the global generator effect sets and
/// emitters take their seeds from, and the one that filled the vector tables.
#[derive(Debug, Clone)]
pub struct Xorshift128(pub [u32; 4]);

// Steps of the runtime's own generators, not iterators.
#[allow(clippy::should_implement_trait)]
impl Xorshift128 {
    pub fn next(&mut self) -> u32 {
        let [x, y, z, w] = self.0;
        let t = x ^ (x << 11);
        let n = w ^ (w >> 19) ^ t ^ (t >> 8);
        self.0 = [y, z, w, n];
        n
    }
}

/// The emitter's LCG (emitter+0xac): value * 2^-32, then step.
#[derive(Debug, Clone)]
pub struct Lcg(pub u32);

#[allow(clippy::should_implement_trait)]
impl Lcg {
    pub fn next(&mut self) -> f32 {
        let value = self.0;
        self.0 = self.0.wrapping_mul(0x41c6_4e6d).wrapping_add(0x3039);
        value as f32 * (1.0 / 4_294_967_296.0)
    }
    pub fn signed(&mut self) -> f32 {
        self.next() * 2.0 - 1.0
    }
    /// A whole number below `spread` (`(state * spread) >> 32`), then step.
    pub fn below(&mut self, spread: i32) -> f32 {
        let draw = ((self.0 as u64 * spread.max(0) as u64) >> 32) as f32;
        self.next();
        draw
    }
}

/// The 512 entries of each random vector table (Random::Initialize, Smash 0x13800).
const TABLE: usize = 512;

struct Tables {
    /// Points in the cube [-1, 1)^3.
    cube: Vec<Vec3>,
    /// Directions.
    unit: Vec<Vec3>,
}

/// Random::Initialize fills both tables once from fixed seeds: each entry takes six draws, the
/// first three a point in the cube, the next three a direction (normalised; zero if all three
/// were zero). A draw is turned into [-1, 1) through its top 23 bits as a float mantissa.
fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut g = Xorshift128([0x178e_ab2c, 0xe318_145e, 0x45f0_cdb4, 0x720a_056d]);
        let mut draw = || (f32::from_bits(g.next() >> 9 | 0x3f80_0000) - 1.0) * 2.0 - 1.0;
        let (mut cube, mut unit) = (Vec::with_capacity(TABLE), Vec::with_capacity(TABLE));
        for _ in 0..TABLE {
            cube.push(Vec3::new(draw(), draw(), draw()));
            unit.push(Vec3::new(draw(), draw(), draw()).normalize_or_zero());
        }
        Tables { cube, unit }
    })
}

/// An emitter's random state (detail::Random at emitter+0xa8, set by Random::SetSeed): the LCG
/// at the seed, and the two table indices at its low and high halves. Each read of a table
/// moves its index on.
#[derive(Debug, Clone)]
pub struct EmitterRandom {
    pub lcg: Lcg,
    /// Into the cube table (emitter+0xa8): a particle's random velocity (res+0x980) and the
    /// simple random field (FRN1).
    pub cube_at: u16,
    /// Into the direction table (emitter+0xaa): a point emitter's directions and the position
    /// random (res+0x810).
    pub unit_at: u16,
}

#[allow(clippy::should_implement_trait)]
impl EmitterRandom {
    pub fn seeded(seed: u32) -> Self {
        Self {
            lcg: Lcg(seed),
            cube_at: seed as u16,
            unit_at: (seed >> 16) as u16,
        }
    }
    pub fn next(&mut self) -> f32 {
        self.lcg.next()
    }
    pub fn signed(&mut self) -> f32 {
        self.lcg.signed()
    }
    pub fn below(&mut self, spread: i32) -> f32 {
        self.lcg.below(spread)
    }
    pub fn cube(&mut self) -> Vec3 {
        let v = tables().cube[self.cube_at as usize % TABLE];
        self.cube_at = self.cube_at.wrapping_add(1);
        v
    }
    pub fn unit(&mut self) -> Vec3 {
        let v = tables().unit[self.unit_at as usize % TABLE];
        self.unit_at = self.unit_at.wrapping_add(1);
        v
    }
}

/// An emitter's fixed seed (seed type 2): res+0x760 times 0xdfdc1c35 (Emitter::Initialize).
pub fn fixed_seed(raw: &[u8]) -> u32 {
    let value = raw
        .get(0x760..0x764)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .unwrap_or(0);
    value.wrapping_mul(0xdfdc_1c35)
}

/// The seeds of one spawn of an effect set (EmitterSet::ResetCommon, Smash 0x3cb0, and
/// Emitter::Initialize, 0x1cc0). The set draws a seed from the global generator when it is
/// spawned; then each emitter, in the order the set creates them (each emitter, then its child
/// emitters), takes by its seed type (res+0x757) a draw of its own (0), the set's seed (1) or its
/// fixed seed (2). So every spawn differs in game except the emitters with a fixed seed, and the
/// emitters of a set that take its seed share one sequence.
///
/// `spawn` stands in for the global generator's state, which in game is wherever earlier spawns
/// left it: the same number gives the same spawn again.
#[derive(Debug, Clone)]
pub struct SpawnSeeds {
    global: Xorshift128,
    set_seed: u32,
}

impl SpawnSeeds {
    pub fn new(spawn: u32) -> Self {
        // splitmix64 spreads the number over a state that is never all zero.
        let mut z = u64::from(spawn) ^ 0x6a09_e667_f3bc_c908;
        let mut word = || {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut r = z;
            r = (r ^ (r >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            r = (r ^ (r >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (r ^ (r >> 31)) as u32
        };
        let mut global = Xorshift128([word(), word(), word(), word() | 1]);
        let set_seed = global.next();
        Self { global, set_seed }
    }

    /// The next emitter's seed, by its seed type.
    pub fn emitter(&mut self, raw: &[u8]) -> u32 {
        match raw.get(0x757).copied().unwrap_or(2) {
            0 => self.global.next(),
            1 => self.set_seed,
            _ => fixed_seed(raw),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tables_hold_cube_points_and_directions() {
        let t = tables();
        assert_eq!((t.cube.len(), t.unit.len()), (512, 512));
        for v in &t.cube {
            assert!(
                v.abs().max_element() < 1.0 && v.min_element() >= -1.0,
                "{v}"
            );
        }
        for v in &t.unit {
            assert!((v.length() - 1.0).abs() < 1e-5, "{v}");
        }
        // Spread over the whole cube and every way round.
        let mean = t.cube.iter().sum::<Vec3>() / 512.0;
        assert!(mean.abs().max_element() < 0.1, "{mean}");
        let mean = t.unit.iter().sum::<Vec3>() / 512.0;
        assert!(mean.abs().max_element() < 0.1, "{mean}");
        // The first entry is the first six draws of the generator from the game's seeds.
        let mut g = Xorshift128([0x178e_ab2c, 0xe318_145e, 0x45f0_cdb4, 0x720a_056d]);
        let first = (f32::from_bits(g.next() >> 9 | 0x3f80_0000) - 1.0) * 2.0 - 1.0;
        assert_eq!(t.cube[0].x, first);
    }

    #[test]
    fn spawns_differ_and_seed_types_say_which_emitters_share() {
        let raw = |kind: u8, fixed: u32| {
            let mut r = vec![0u8; 0x770];
            r[0x757] = kind;
            r[0x760..0x764].copy_from_slice(&fixed.to_le_bytes());
            r
        };
        let (own, set, fixed) = (raw(0, 0), raw(1, 0), raw(2, 7));
        let seeds = |spawn| {
            let mut s = SpawnSeeds::new(spawn);
            [
                s.emitter(&own),
                s.emitter(&own),
                s.emitter(&set),
                s.emitter(&set),
                s.emitter(&fixed),
            ]
        };
        let (a, b) = (seeds(1), seeds(2));
        assert_eq!(a, seeds(1));
        assert_ne!(a[0], a[1]);
        assert_eq!(a[2], a[3]);
        assert_eq!(a[4], 7u32.wrapping_mul(0xdfdc_1c35));
        assert_eq!(a[4], b[4]);
        for k in 0..4 {
            assert_ne!(a[k], b[k]);
        }
    }
}
