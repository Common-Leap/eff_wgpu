//! The CPU half of an emitter: emission, particle birth and motion, and what is handed to the
//! shaders each frame (per-particle streams and the dynamic block).
//!
//! Stream contents follow Smash's InitializeParticle (0xa2d0): LocalPos = position and life,
//! LocalVec = velocity and the emitter time at birth, Scale, four Random values, InitRotate.
//! The shaders do the rest (colour, alpha, scale curves, billboarding) from the static block.

use std::collections::HashMap;

use effect_library::emitter::EmitterData;
use glam::{Mat4, Vec3};

/// The runtime's emitter LCG (emitter+0xac): value * 2^-32, then step.
#[derive(Debug, Clone)]
pub struct Lcg(pub u32);

impl Lcg {
    pub fn next(&mut self) -> f32 {
        let value = self.0;
        self.0 = self.0.wrapping_mul(0x41c6_4e6d).wrapping_add(0x3039);
        value as f32 * (1.0 / 4_294_967_296.0)
    }
    pub fn signed(&mut self) -> f32 {
        self.next() * 2.0 - 1.0
    }
    fn unit_vector(&mut self) -> Vec3 {
        loop {
            let v = Vec3::new(self.signed(), self.signed(), self.signed());
            let length = v.length();
            if length > 1e-4 && length <= 1.0 {
                return v / length;
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Live {
    position: Vec3,
    velocity: Vec3,
    /// Movement over the last frame (LocalDiff).
    diff: Vec3,
    life: f32,
    birth: f32,
    scale: [f32; 4],
    random: [f32; 4],
    rotate: [f32; 4],
}

fn i32_at(b: &[u8], at: usize) -> i32 {
    b.get(at..at + 4)
        .map(|s| i32::from_le_bytes(s.try_into().unwrap()))
        .unwrap_or(0)
}

fn f32_at(b: &[u8], at: usize) -> f32 {
    b.get(at..at + 4)
        .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
        .unwrap_or(0.0)
}

/// One emitter running in the effect's space.
pub struct EmitterSim<'a> {
    data: &'a EmitterData,
    /// The emitter's data after load-time fixes (`static_block::fixed_emitter_data`).
    raw: &'a [u8],
    rng: Lcg,
    /// Emitter time (emitter+0x44).
    pub time: f32,
    /// Fade-in (emitter+0x74): 0 to 1 over res+0x76c frames when res+0x75b (alpha) or
    /// res+0x75c (scale) asks for it, else 1 (Emitter::Initialize, EmitterCalculator::Calculate).
    fade_in: f32,
    /// The emitter's own transform in its set (emitter+0x2c0, CreateResMatrix): translation,
    /// rotation and scale with their random spreads.
    local: Mat4,
    /// The same without scale (emitter+0x300).
    local_rt: Mat4,
    timer: f32,
    interval: f32,
    accumulated: f32,
    particles: Vec<Live>,
}

impl<'a> EmitterSim<'a> {
    pub fn new(data: &'a EmitterData, raw: &'a [u8]) -> Self {
        let mut rng = Lcg(data.random_seed);
        let (local, local_rt) = res_matrix(raw, &mut rng);
        Self {
            data,
            raw,
            rng,
            local,
            local_rt,
            time: 0.0,
            fade_in: if raw.get(0x75b).copied().unwrap_or(0) != 0
                || raw.get(0x75c).copied().unwrap_or(0) != 0
            {
                0.0
            } else {
                1.0
            },
            timer: f32::MAX,
            interval: 0.0,
            accumulated: 0.0,
            particles: Vec::new(),
        }
    }

    /// The emitter's matrix (emitter+0x340, UpdateEmitterMatrix): the set's matrix times the
    /// emitter's own.
    pub fn emitter_srt(&self, set: &Mat4) -> Mat4 {
        *set * self.local
    }

    /// The emitter's matrix without scale (emitter+0x380): the set's, without scale, times
    /// the emitter's own without scale.
    pub fn emitter_rt(&self, set_rt: &Mat4) -> Mat4 {
        *set_rt * self.local_rt
    }

    pub fn count(&self) -> usize {
        self.particles.len()
    }

    fn spawn(&mut self, count: usize) {
        let data = self.data;
        let velocity = &data.particle_velocity;
        for index in 0..count {
            let (mut position, normal) = spawn_shape(data, &mut self.rng, index, count);
            if data.emission.position_random > 0.0 {
                let jitter =
                    Vec3::new(self.rng.signed(), 0.0, self.rng.signed()).normalize_or_zero();
                position += jitter * data.emission.position_random;
            }
            let mut direction = Vec3::new(
                velocity.designated_dir_x,
                velocity.designated_dir_y,
                velocity.designated_dir_z,
            )
            .normalize_or_zero();
            if velocity.diffusion_dir_angle > 0.0 && direction != Vec3::ZERO {
                let spread = velocity.diffusion_dir_angle.to_radians();
                direction = (direction + self.rng.unit_vector() * spread.sin()).normalize_or_zero();
            }
            let mut vel =
                normal * velocity.all_direction + direction * velocity.designated_dir_scale;
            vel *= 1.0 - self.rng.next() * velocity.vel_random / 100.0;
            // Life (InitializeParticle): life * (1 - randint(0..=life_random) / 100), or
            // 2^28 frames when infinite.
            let raw = self.raw;
            let life = if raw.get(0x8a8).copied().unwrap_or(0) != 0 {
                268_435_456.0
            } else {
                let spread = i32_at(raw, 0x8bc);
                let draw = ((self.rng.0 as u64 * spread.max(0) as u64) >> 32) as f32;
                self.rng.next();
                i32_at(raw, 0x8b8) as f32 * (1.0 - draw * 0.01)
            };
            // Scale: base (0x130..) * emitter scale (0x9c0..) * (1 - random% / 100 * r); one draw
            // for every axis when the x and y percentages match.
            let (px, py, pz) = (f32_at(raw, 0x9cc), f32_at(raw, 0x9d0), f32_at(raw, 0x9d4));
            let factors = if px == py {
                let f = 1.0 - px / 100.0 * self.rng.next();
                [f, f, f]
            } else {
                [
                    1.0 - px / 100.0 * self.rng.next(),
                    1.0 - py / 100.0 * self.rng.next(),
                    1.0 - pz / 100.0 * self.rng.next(),
                ]
            };
            let spread = f32_at(raw, 0x8c0);
            let w = 1.0 + spread * (1.0 - 2.0 * self.rng.next());
            let scale4 = [
                f32_at(raw, 0x9c0) * factors[0],
                f32_at(raw, 0x9c4) * factors[1],
                f32_at(raw, 0x9c8) * factors[2],
                w,
            ];
            let random = [
                self.rng.next(),
                self.rng.next(),
                self.rng.next(),
                self.rng.next(),
            ];
            self.particles.push(Live {
                position,
                velocity: vel,
                diff: vel,
                life,
                birth: self.time,
                scale: scale4,
                random,
                rotate: [
                    f32_at(self.raw, 0x320),
                    f32_at(self.raw, 0x324),
                    f32_at(self.raw, 0x328),
                    0.0,
                ],
            });
        }
    }

    /// One frame: emission, then motion; the emitter time advances by one.
    pub fn step(&mut self) {
        if self.fade_in < 1.0 {
            let frames = i32_at(self.raw, 0x76c);
            self.fade_in = if frames < 1 {
                1.0
            } else {
                (self.fade_in + 1.0 / frames as f32).min(1.0)
            };
        }
        let frame = self.time;
        let emission = &self.data.emission;
        let start = emission.start as f32;
        let active = frame >= start
            && (emission.duration == 0
                || frame < start + emission.duration as f32
                || emission.is_one_time);
        if emission.is_one_time {
            if frame == start {
                let n = emission.rate * (1.0 - emission.rate_random / 100.0 * self.rng.next());
                self.spawn(n.max(0.0) as usize);
            }
        } else if active {
            if self.timer >= self.interval {
                let rate = emission.rate * (1.0 + emission.rate_random / 100.0 * self.rng.next());
                self.accumulated = (self.accumulated + rate).max(0.0);
                let n = self.accumulated.floor();
                if n > 0.0 {
                    self.spawn(n as usize);
                    self.accumulated -= n;
                    self.timer = 0.0;
                    let jitter = emission.interval_random * self.rng.next();
                    self.interval = (emission.interval as f32 + jitter).max(0.0);
                } else {
                    self.timer = 0.0;
                }
            } else {
                self.timer += 1.0;
            }
        }
        let statics = &self.data.emitter_static;
        let gravity = Vec3::new(
            statics.gravity_dir_x,
            statics.gravity_dir_y,
            statics.gravity_dir_z,
        ) * statics.gravity_scale;
        let air = statics.air_res;
        for particle in &mut self.particles {
            particle.position += particle.velocity;
            particle.diff = particle.velocity;
            if air != 1.0 {
                particle.velocity *= air;
            }
            particle.velocity += gravity;
        }
        self.time += 1.0;
        let now = self.time;
        self.particles.retain(|p| now - p.birth < p.life);
    }

    /// The per-particle streams the shaders read, by attribute name.
    pub fn streams(&self, emitter_matrix: &Mat4) -> HashMap<&'static str, Vec<[f32; 4]>> {
        let mut out: HashMap<&'static str, Vec<[f32; 4]>> = HashMap::new();
        let rows = emitter_matrix.transpose().to_cols_array_2d();
        for p in &self.particles {
            out.entry("sysLocalPosAttr").or_default().push([
                p.position.x,
                p.position.y,
                p.position.z,
                p.life,
            ]);
            out.entry("sysLocalVecAttr").or_default().push([
                p.velocity.x,
                p.velocity.y,
                p.velocity.z,
                p.birth,
            ]);
            out.entry("sysLocalDiffAttr")
                .or_default()
                .push([p.diff.x, p.diff.y, p.diff.z, 0.0]);
            out.entry("sysScaleAttr").or_default().push(p.scale);
            out.entry("sysRandomAttr").or_default().push(p.random);
            out.entry("sysInitRotateAttr").or_default().push(p.rotate);
            out.entry("sysColor0Attr").or_default().push([1.0; 4]);
            out.entry("sysColor1Attr").or_default().push([1.0; 4]);
            out.entry("sysEmtMat0Attr").or_default().push(rows[0]);
            out.entry("sysEmtMat1Attr").or_default().push(rows[1]);
            out.entry("sysEmtMat2Attr").or_default().push(rows[2]);
        }
        out
    }

    /// The dynamic block (Smash 0x9770) for this frame. `srt` is the emitter's matrix with
    /// scale, `rt` without; `colour` is the user/set colour multiplier (1 by default).
    pub fn dynamic_block(&self, srt: &Mat4, rt: &Mat4, colour: [f32; 4]) -> Vec<u8> {
        let r = self.raw;
        let mut v = [0f32; 48];
        v[0] = colour[0] * f32_at(r, 0x7ac);
        v[1] = colour[1] * f32_at(r, 0x7b0);
        v[2] = colour[2] * f32_at(r, 0x7b4);
        v[3] = colour[3] * f32_at(r, 0x7b8);
        v[4] = colour[0] * f32_at(r, 0x7bc);
        v[5] = colour[1] * f32_at(r, 0x7c0);
        v[6] = colour[2] * f32_at(r, 0x7c4);
        v[7] = colour[3] * f32_at(r, 0x7c8);
        v[8] = self.time;
        v[9] = self.particles.len() as f32;
        v[10] = 0.0;
        v[11] = 1.0;
        // Alpha and scale multipliers: the set's own (1 here) times the fade-in when the
        // emitter asks for it. The fade-out on kill (emitter+0x70) is not modelled: it stays 1.
        let fade = |flag: usize| {
            if r.get(flag).copied().unwrap_or(0) != 0 {
                self.fade_in
            } else {
                1.0
            }
        };
        v[12] = fade(0x75b);
        let scale = fade(0x75c);
        v[13] = scale;
        v[14] = scale;
        v[15] = scale;
        let rows = |m: &Mat4| m.transpose().to_cols_array();
        v[16..32].copy_from_slice(&rows(srt));
        v[32..48].copy_from_slice(&rows(rt));
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}

/// Emitter::CreateResMatrix (Smash 0x2c80): rotation res+0x788 + res+0x794 * (2r - 1) per axis,
/// translation res+0x770 + res+0x77c * (2r - 1), scale res+0x7a0; six draws, rotation first.
/// The rows it writes are the columns of T * Rz * Ry * Rx * S.
fn res_matrix(raw: &[u8], rng: &mut Lcg) -> (Mat4, Mat4) {
    let mut jitter =
        |base: usize, spread: usize| f32_at(raw, base) + f32_at(raw, spread) * rng.signed();
    let rotation = [
        jitter(0x788, 0x794),
        jitter(0x78c, 0x798),
        jitter(0x790, 0x79c),
    ];
    let translation = Vec3::new(
        jitter(0x770, 0x77c),
        jitter(0x774, 0x780),
        jitter(0x778, 0x784),
    );
    let scale = Vec3::new(f32_at(raw, 0x7a0), f32_at(raw, 0x7a4), f32_at(raw, 0x7a8));
    let rotate = Mat4::from_euler(glam::EulerRot::ZYX, rotation[2], rotation[1], rotation[0]);
    let rt = Mat4::from_translation(translation) * rotate;
    (rt * Mat4::from_scale(scale), rt)
}

/// Spawn position and outward direction by volume type (g_EmitFunctions order).
fn spawn_shape(data: &EmitterData, rng: &mut Lcg, index: usize, count: usize) -> (Vec3, Vec3) {
    let shape = &data.shape_info;
    let radius = Vec3::new(
        shape.volume_radius_x * shape.volume_form_scale_x.max(0.0),
        shape.volume_radius_y * shape.volume_form_scale_y.max(0.0),
        shape.volume_radius_z * shape.volume_form_scale_z.max(0.0),
    );
    let sweep = if shape.sweep_longitude > 0.0 {
        shape.sweep_longitude
    } else {
        std::f32::consts::TAU
    };
    let circle = |angle: f32, fill: f32| {
        let dir = Vec3::new(angle.sin(), 0.0, angle.cos());
        (dir * radius * fill, dir)
    };
    match shape.volume_type {
        1 | 3 => {
            let angle = shape.sweep_start + rng.next() * sweep;
            let fill = if shape.volume_type == 3 {
                1.0 - shape.caliber_ratio * rng.next()
            } else {
                1.0
            };
            circle(angle, fill)
        }
        2 => circle(
            shape.sweep_start + sweep * index as f32 / count.max(1) as f32,
            1.0,
        ),
        4..=7 => {
            let dir = rng.unit_vector();
            let fill = if shape.volume_type == 7 {
                1.0 - shape.caliber_ratio * rng.next()
            } else {
                1.0
            };
            (dir * radius * fill, dir)
        }
        8 | 9 => {
            let angle = shape.sweep_start + rng.next() * sweep;
            let (mut position, dir) = circle(angle, 1.0);
            position.y = rng.signed() * radius.y;
            (position, dir)
        }
        10 | 11 => {
            let position = Vec3::new(rng.signed(), rng.signed(), rng.signed()) * radius;
            (position, position.normalize_or_zero())
        }
        12 | 13 => {
            let t = if shape.volume_type == 13 {
                index as f32 / count.max(1) as f32 - 0.5
            } else {
                rng.signed() * 0.5
            };
            (
                Vec3::new(0.0, 0.0, shape.line_center + t * shape.line_length),
                Vec3::Z,
            )
        }
        14 => (
            Vec3::new(rng.signed() * radius.x, 0.0, rng.signed() * radius.z),
            Vec3::Y,
        ),
        _ => (Vec3::ZERO, rng.unit_vector()),
    }
}
