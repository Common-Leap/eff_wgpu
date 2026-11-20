//! The field sections of an emitter (`ptcl::Emitter::attributes`): forces the game applies to
//! each particle on the CPU after its own motion (EmitterCalculator::CalculateParticleBehavior,
//! Smash 0xf9f0). In the game's order: spin (FSPN), convergence (FCOV), curl noise (FCLN) and
//! position add (FPAD). Magnet (FMAG), collision (FCOL) and the two random fields (FRND,
//! FRN1) are not run.

use std::sync::OnceLock;

use glam::{Mat4, Vec3};

fn f32_at(b: &[u8], at: usize) -> f32 {
    b.get(at..at + 4)
        .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
        .unwrap_or(0.0)
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .unwrap_or(0)
}

/// A field value animated over each particle's life (ResAnim8KeyParamSet: enabled, loop,
/// random start, key count, loop length, then up to eight keys of x, y, z and time).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anim8 {
    looped: bool,
    random_start: bool,
    count: usize,
    loop_length: f32,
    keys: [[f32; 4]; 8],
}

impl Anim8 {
    /// The set at `at` in `d`, when its enabled word is set.
    fn read(d: &[u8], at: usize) -> Option<Self> {
        if u32_at(d, at) == 0 || d.len() < at + 0x14 {
            return None;
        }
        let mut keys = [[0.0; 4]; 8];
        for (k, key) in keys.iter_mut().enumerate() {
            for (c, v) in key.iter_mut().enumerate() {
                *v = f32_at(d, at + 0x14 + k * 0x10 + c * 4);
            }
        }
        Some(Anim8 {
            looped: u32_at(d, at + 4) != 0,
            random_start: u32_at(d, at + 8) != 0,
            count: (u32_at(d, at + 0xc) as usize).min(8),
            loop_length: u32_at(d, at + 0x10) as f32,
            keys,
        })
    }

    /// The value at a particle's `age` (CalculateField8KeyAnim, Smash 0xe280): over the
    /// emitter's life `life`, or over the loop length from a start the particle's random value
    /// picks; held at the first key before it and the last after it, linear between.
    pub fn at(&self, age: f32, life: f32, random: f32) -> Vec3 {
        let value = |k: usize| Vec3::new(self.keys[k][0], self.keys[k][1], self.keys[k][2]);
        if self.count == 0 {
            return Vec3::ZERO;
        }
        if self.count == 1 {
            return value(0);
        }
        let t = if self.looped && self.loop_length > 0.0 {
            let start = if self.random_start {
                random * self.loop_length
            } else {
                0.0
            };
            ((start + age) % self.loop_length) / self.loop_length
        } else {
            age / life
        };
        let last = self.count - 1;
        if t >= self.keys[last][3] {
            return value(last);
        }
        if t < self.keys[0][3] {
            return value(0);
        }
        for k in 0..last {
            let (a, b) = (self.keys[k][3], self.keys[k + 1][3]);
            if a <= t && t < b {
                return value(k) + (value(k + 1) - value(k)) * ((t - a) / (b - a));
            }
        }
        Vec3::ZERO
    }
}

/// FCOV: each frame a particle moves `strength` of the way to `target`. With `relative`, the
/// target is taken from where the emitter is now (FieldConvergence's second type).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Convergence {
    pub relative: bool,
    pub target: Vec3,
    pub strength: f32,
    /// The strength over the particle's life, replacing `strength` when set.
    pub strength_anim: Option<Anim8>,
}

/// FSPN: each frame a particle turns `speed` radians about one of the emitter's axes and moves
/// `outward` units away from that axis (Smash 0xea80).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spin {
    /// 0, 1 or 2 for X, Y or Z.
    pub axis: u32,
    pub speed: f32,
    pub outward: f32,
    speed_anim: Option<Anim8>,
    outward_anim: Option<Anim8>,
}

/// FCLN: each frame a particle's velocity gains `power` times the game's curl noise table at
/// its position times `scale`, offset by `base` and by `speed` times the emitter's time
/// (Smash 0x12b90).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurlNoise {
    /// Blend the eight table cells around the point instead of taking the one it is in.
    pub interpolate: bool,
    /// `base` is scaled by the particle's first random value.
    pub random_base: bool,
    /// Sample at the particle's world position and turn the noise back into emitter space.
    pub world: bool,
    pub speed: Vec3,
    pub power: Vec3,
    pub scale: f32,
    pub base: f32,
}

/// FPAD: each frame a particle moves by `amount`, in emitter space or, with `world`, in the
/// world's axes (Smash 0xf690).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionAdd {
    pub world: bool,
    pub amount: Vec3,
    amount_anim: Option<Anim8>,
}

/// The field sections that move an emitter's particles on the CPU. Read with [`Fields::of`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Fields {
    pub convergence: Option<Convergence>,
    pub spin: Option<Spin>,
    pub curl_noise: Option<CurlNoise>,
    pub position_add: Option<PositionAdd>,
}

impl Fields {
    /// The fields an emitter's attribute sections hold.
    pub fn of(attributes: &[([u8; 4], Vec<u8>)]) -> Self {
        let section = |magic: &[u8; 4]| {
            attributes
                .iter()
                .find(|(m, _)| m == magic)
                .map(|(_, d)| d.as_slice())
        };
        let vec3 =
            |d: &[u8], at: usize| Vec3::new(f32_at(d, at), f32_at(d, at + 4), f32_at(d, at + 8));
        // FCOV: type byte, target (+4), strength (+0x10), its animation (+0x14).
        let convergence = section(b"FCOV")
            .filter(|d| d.len() >= 0x14)
            .map(|d| Convergence {
                relative: d[0] != 0,
                target: vec3(d, 4),
                strength: f32_at(d, 0x10),
                strength_anim: Anim8::read(d, 0x14),
            });
        // FSPN: speed (+0), axis (+4), outward speed (+8), their animations (+0xc, +0xa0).
        let spin = section(b"FSPN").filter(|d| d.len() >= 0xc).map(|d| Spin {
            speed: f32_at(d, 0),
            axis: u32_at(d, 4),
            outward: f32_at(d, 8),
            speed_anim: Anim8::read(d, 0xc),
            outward_anim: Anim8::read(d, 0xa0),
        });
        // FCLN: flag bytes (+0..+2), speed (+4), power (+0x10), scale (+0x1c), base (+0x20).
        let curl_noise = section(b"FCLN")
            .filter(|d| d.len() >= 0x24)
            .map(|d| CurlNoise {
                interpolate: d[0] & 1 != 0,
                random_base: d[1] & 1 != 0,
                world: d[2] != 0,
                speed: vec3(d, 4),
                power: vec3(d, 0x10),
                scale: f32_at(d, 0x1c),
                base: f32_at(d, 0x20),
            });
        // FPAD: world byte (+0), amount (+4), its animation (+0x10).
        let position_add = section(b"FPAD")
            .filter(|d| d.len() >= 0x10)
            .map(|d| PositionAdd {
                world: d[0] != 0,
                amount: vec3(d, 4),
                amount_anim: Anim8::read(d, 0x10),
            });
        Fields {
            convergence,
            spin,
            curl_noise,
            position_add,
        }
    }

    pub fn is_empty(&self) -> bool {
        *self == Fields::default()
    }
}

/// What the fields read of one particle and its emitter this frame.
pub(crate) struct Particle<'a> {
    pub position: &'a mut Vec3,
    pub velocity: &'a mut Vec3,
    /// Frames since the particle was born.
    pub age: f32,
    /// The emitter's life (res+0x8b8), over which animated values run.
    pub life: f32,
    /// The particle's first random value.
    pub random: f32,
    /// The particle's motion scale (its scale stream's w) times the frame rate.
    pub motion: f32,
    /// The frame rate alone (curl noise is not scaled by the particle's motion scale).
    pub frame_rate: f32,
    /// The emitter's time.
    pub time: f32,
    /// Emitter space to world for this particle: the emitter's matrix now when particles
    /// follow it, else the one the particle was born with.
    pub matrix: Mat4,
    /// Where the emitter is now, in this particle's emitter space.
    pub emitter_here: Vec3,
}

impl Fields {
    /// One frame of every field on one particle, in the game's order.
    pub(crate) fn apply(&self, p: Particle) {
        let rate = p.motion;
        if let Some(spin) = self.spin {
            let degrees = std::f32::consts::PI / 180.0;
            let speed = spin
                .speed_anim
                .map_or(spin.speed, |a| a.at(p.age, p.life, p.random).x * degrees);
            let outward = spin
                .outward_anim
                .map_or(spin.outward, |a| a.at(p.age, p.life, p.random).x * degrees);
            let (s, c) = (rate * speed).sin_cos();
            let v = &mut *p.position;
            // The two coordinates across the axis, turned the way Smash 0xea80 turns them.
            let (a, b) = match spin.axis {
                0 => (c * v.y + s * v.z, c * v.z - s * v.y),
                1 => (c * v.x - s * v.z, c * v.z + s * v.x),
                2 => (c * v.x + s * v.y, c * v.y - s * v.x),
                _ => (f32::NAN, f32::NAN),
            };
            if !a.is_nan() {
                let length = (a * a + b * b).sqrt();
                let push = if outward != 0.0 && length > 0.0 {
                    rate * outward / length
                } else {
                    0.0
                };
                let (a, b) = (a + a * push, b + b * push);
                match spin.axis {
                    0 => (v.y, v.z) = (a, b),
                    1 => (v.x, v.z) = (a, b),
                    _ => (v.x, v.y) = (a, b),
                }
            }
        }
        if let Some(c) = self.convergence {
            let strength = c
                .strength_anim
                .map_or(c.strength, |a| a.at(p.age, p.life, p.random).x);
            let target = if c.relative {
                p.emitter_here + c.target
            } else {
                c.target
            };
            *p.position += (target - *p.position) * strength * rate;
        }
        if let Some(n) = self.curl_noise {
            let base = if n.random_base {
                n.base * p.random
            } else {
                n.base
            };
            let at = if n.world {
                p.matrix.transform_point3(*p.position)
            } else {
                *p.position
            };
            let point = Vec3::splat(base) + n.speed * p.time + at * n.scale;
            let mut noise = curl_noise(point, n.interpolate);
            if n.world {
                noise = to_local(&p.matrix, noise);
            }
            // Each axis of the power scales its own axis of the noise; the frame rate is the
            // only other factor, not the particle's motion scale.
            *p.velocity += n.power * noise * p.frame_rate;
        }
        if let Some(add) = self.position_add {
            let amount = add
                .amount_anim
                .map_or(add.amount, |a| a.at(p.age, p.life, p.random))
                * rate;
            *p.position += if add.world {
                to_local(&p.matrix, amount)
            } else {
                amount
            };
        }
    }
}

/// A world direction in the space of `matrix`, by its rotation alone.
fn to_local(matrix: &Mat4, v: Vec3) -> Vec3 {
    let (_, rotation, _) = matrix.to_scale_rotation_translation();
    rotation.inverse() * v
}

/// The side of the curl noise table (CurlNoiseTable: 32 cells a side).
const CURL_SIDE: i32 = 32;

/// The curl noise table: three signed bytes a cell, x the slowest axis, read as byte / 127.
///
/// The game's own table is data in its executable, so this one is made here instead: a smooth
/// random field on a lattice of four cells, with the game table's spread per axis (standard
/// deviation 0.25, 0.20 and 0.23) and about its correlation between neighbouring cells (0.75
/// at one cell, near none past eight). Particles move with the same strength and grain as in
/// the game; their exact paths differ, as they do in the game from one spawn to the next.
fn curl_table() -> &'static [[i8; 3]] {
    static TABLE: OnceLock<Vec<[i8; 3]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        const LATTICE: usize = 8;
        const CELL: usize = (CURL_SIDE as usize) / LATTICE;
        let mut state: u32 = 0x1234_5678;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        let points: Vec<[f32; 3]> = (0..LATTICE.pow(3))
            .map(|_| [next(), next(), next()])
            .collect();
        let point = |x: usize, y: usize, z: usize| {
            points[(x % LATTICE) * LATTICE * LATTICE + (y % LATTICE) * LATTICE + z % LATTICE]
        };
        let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
        let side = CURL_SIDE as usize;
        let mut field = vec![[0f32; 3]; side * side * side];
        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    let (lx, ly, lz) = (x / CELL, y / CELL, z / CELL);
                    let f = |c: usize| smooth((c % CELL) as f32 / CELL as f32);
                    let (fx, fy, fz) = (f(x), f(y), f(z));
                    let out = &mut field[(x * side + y) * side + z];
                    for (axis, value) in out.iter_mut().enumerate() {
                        let at = |dx, dy, dz| point(lx + dx, ly + dy, lz + dz)[axis];
                        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
                        let x0 = lerp(
                            lerp(at(0, 0, 0), at(0, 0, 1), fz),
                            lerp(at(0, 1, 0), at(0, 1, 1), fz),
                            fy,
                        );
                        let x1 = lerp(
                            lerp(at(1, 0, 0), at(1, 0, 1), fz),
                            lerp(at(1, 1, 0), at(1, 1, 1), fz),
                            fy,
                        );
                        *value = lerp(x0, x1, fx);
                    }
                }
            }
        }
        let spread = [0.25f32, 0.20, 0.23];
        let mut out = vec![[0i8; 3]; field.len()];
        for axis in 0..3 {
            let n = field.len() as f32;
            let mean = field.iter().map(|v| v[axis]).sum::<f32>() / n;
            let deviation =
                (field.iter().map(|v| (v[axis] - mean).powi(2)).sum::<f32>() / n).sqrt();
            for (o, v) in out.iter_mut().zip(&field) {
                let value = (v[axis] - mean) / deviation * spread[axis];
                o[axis] = (value * 127.0).round().clamp(-127.0, 127.0) as i8;
            }
        }
        out
    })
}

/// One cell of the table by whole coordinates, each taken as its absolute value modulo the
/// side (CalculateParticleBehavior_FieldCurlNoise).
fn curl_cell(x: i32, y: i32, z: i32) -> Vec3 {
    let wrap = |c: i32| c.unsigned_abs() as usize % CURL_SIDE as usize;
    let side = CURL_SIDE as usize;
    let cell = curl_table()[(wrap(x) * side + wrap(y)) * side + wrap(z)];
    Vec3::new(cell[0] as f32, cell[1] as f32, cell[2] as f32) / 127.0
}

/// The curl noise at `point`: the cell it is in, or (GetCurlNoiseS8Interpolate) a blend of
/// that cell and the next one along each axis by the point's fraction, both taken towards
/// zero as the game's integer conversion does.
pub(crate) fn curl_noise(point: Vec3, interpolate: bool) -> Vec3 {
    let (x, y, z) = (point.x as i32, point.y as i32, point.z as i32);
    if !interpolate {
        return curl_cell(x, y, z);
    }
    // The next cell: one on, or for a coordinate below -1 its bitwise complement.
    let next = |c: i32| if c >= -1 { c + 1 } else { !c };
    let (fx, fy, fz) = (point.x - x as f32, point.y - y as f32, point.z - z as f32);
    let (nx, ny, nz) = (next(x), next(y), next(z));
    let lerp = |a: Vec3, b: Vec3, t: f32| a + (b - a) * t;
    let plane = |zc: i32| {
        lerp(
            lerp(curl_cell(x, y, zc), curl_cell(nx, y, zc), fx),
            lerp(curl_cell(x, ny, zc), curl_cell(nx, ny, zc), fx),
            fy,
        )
    };
    lerp(plane(z), plane(nz), fz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_curl_table_has_the_games_spread_and_grain() {
        let table = curl_table();
        assert_eq!(table.len(), 32 * 32 * 32);
        let n = table.len() as f32;
        for (axis, want) in [0.25f32, 0.20, 0.23].iter().enumerate() {
            let values: Vec<f32> = table.iter().map(|c| c[axis] as f32 / 127.0).collect();
            let mean = values.iter().sum::<f32>() / n;
            let deviation = (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
            assert!((deviation - want).abs() < 0.01, "{axis}: {deviation}");
            // Neighbours along x agree as much as the game's do (0.68 to 0.90).
            let neighbour = |i: usize| values[(i + 32 * 32) % values.len()];
            let covariance = values
                .iter()
                .enumerate()
                .map(|(i, v)| (v - mean) * (neighbour(i) - mean))
                .sum::<f32>()
                / n;
            let correlation = covariance / (deviation * deviation);
            assert!((0.6..0.95).contains(&correlation), "{axis}: {correlation}");
        }
    }

    #[test]
    fn interpolated_noise_meets_the_cells_at_whole_coordinates() {
        for p in [Vec3::new(3.0, 7.0, 11.0), Vec3::new(-5.0, 2.0, -9.0)] {
            assert_eq!(
                curl_noise(p, true),
                curl_cell(p.x as i32, p.y as i32, p.z as i32)
            );
        }
        let a = curl_noise(Vec3::new(3.0, 7.0, 11.0), true);
        let b = curl_noise(Vec3::new(4.0, 7.0, 11.0), true);
        let mid = curl_noise(Vec3::new(3.5, 7.0, 11.0), true);
        assert!((mid - (a + b) * 0.5).length() < 1e-6);
    }

    #[test]
    fn an_eight_key_value_holds_its_ends_and_runs_between_keys() {
        let mut keys = [[0.0; 4]; 8];
        keys[0] = [1.0, 0.0, 0.0, 0.2];
        keys[1] = [3.0, 0.0, 0.0, 0.6];
        let anim = Anim8 {
            looped: false,
            random_start: false,
            count: 2,
            loop_length: 0.0,
            keys,
        };
        assert_eq!(anim.at(0.0, 10.0, 0.0).x, 1.0);
        assert_eq!(anim.at(9.0, 10.0, 0.0).x, 3.0);
        assert!((anim.at(4.0, 10.0, 0.0).x - 2.0).abs() < 1e-6);
    }
}
