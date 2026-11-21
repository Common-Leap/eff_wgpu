//! The CPU half of an emitter: emission, particle birth and motion, and what is handed to the
//! shaders each frame (per-particle streams and the dynamic block).
//!
//! Stream contents follow Smash's InitializeParticle (0xa2d0): LocalPos = position and life,
//! LocalVec = velocity and the emitter time at birth, Scale, four Random values, InitRotate.
//! The shaders do the rest (colour, alpha, scale curves, billboarding) from the static block.

use std::collections::HashMap;

use effect_library::emitter::EmitterData;
use glam::{Mat4, Quat, Vec2, Vec3};

pub use crate::random::{EmitterRandom, Lcg, SpawnSeeds};

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
    /// The emitter's matrix when this particle was born (EmtMat0-2): the shader places
    /// particles that do not follow the emitter with it.
    birth_matrix: Mat4,
    /// Colour 0 and 1 (sysColor0Attr / sysColor1Attr): the script's particle colour, or what a
    /// child takes from its parent particle.
    colours: [[f32; 4]; 2],
    /// Each child emitter's emission from this particle: timer, interval and the amount not
    /// yet emitted.
    child_emission: Vec<[f32; 3]>,
}

fn i32_at(b: &[u8], at: usize) -> i32 {
    b.get(at..at + 4)
        .map(|s| i32::from_le_bytes(s.try_into().unwrap()))
        .unwrap_or(0)
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .unwrap_or(0)
}

fn f32_at(b: &[u8], at: usize) -> f32 {
    b.get(at..at + 4)
        .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
        .unwrap_or(0.0)
}

pub use crate::field::{Convergence, CurlNoise, Fields, PositionAdd, Spin};

/// Live particles one emitter keeps, the most the renderer draws in one call.
pub const MAX_PARTICLES: usize = 4096;

/// One emitter running in the effect's space.
pub struct EmitterSim<'a> {
    data: &'a EmitterData,
    /// The emitter's data after load-time fixes (`static_block::fixed_emitter_data`).
    raw: &'a [u8],
    rng: EmitterRandom,
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
    /// The emitter's matrix this frame (emitter+0x340) and its translation last frame
    /// (emitter+0x3c0), for distance emission and birth matrices.
    current: Mat4,
    previous_translation: Option<Vec3>,
    /// The emitter set's scale this frame (the scale an `EFFECT` call asks for). The game
    /// passes it to the shaders as the particle scale, beside the fade (live probe: Dark Samus's
    /// up tilt arc, scale 1.65 at her 0.96, reads 1.584 there).
    set_scale: Vec3,
    /// The particle colour a script sets on the effect (`LAST_PARTICLE_SET_COLOR`), which the
    /// shaders read per particle; white unless set.
    particle_colour: [f32; 4],
    /// Distance travelled and not yet emitted for (distance emission).
    distance: f32,
    /// How far the emitter moved over the last frame, in its own space (emitter+0x3d0): a new
    /// particle takes res+0x990 times it as extra velocity, and distance emission measures it.
    /// A child emitter's is its parent particle's last move, set only when the child emits by
    /// distance (Smash 0x8d70).
    emitter_velocity: Vec3,
    /// The next division an equally divided shape emits from when res+0x874 is 2
    /// (emitter+0x38).
    divide_counter: i32,
    /// Child emitters: each emits from every live particle of this one.
    children: Vec<EmitterSim<'a>>,
    /// Where a child's next particles start, in its parent emitter's space, and the colours
    /// they take from the parent particle.
    spawn_offset: Vec3,
    spawn_velocity: Vec3,
    spawn_colours: Option<[[f32; 4]; 2]>,
    /// Set when the effect is killed (`EFFECT_OFF_KIND`; the game's fade request).
    killed: bool,
    fields: Fields,
    /// Fade-out after a kill (emitter+0x70): 1 to 0 over res+0x768 frames when res+0x755
    /// (alpha) or res+0x756 (scale) asks for it.
    fade_out: f32,
    /// The emitter is gone: its fade-out ran out (Calculate returned false).
    finished: bool,
    /// Effect frames each step advances (emitter+0x4c, the frame step Calculate is handed):
    /// 1 at normal speed, a quarter in training mode's quarter-speed slow motion.
    frame_rate: f32,
}

impl<'a> EmitterSim<'a> {
    /// The emitter with its fixed seed (res+0x760, as seed type 2 gives it), whatever its seed
    /// type: the same particles every time. A preview that plays an effect more than once wants
    /// [`EmitterSim::with_seed`] and a [`SpawnSeeds`] for each spawn, as the game seeds them.
    pub fn new(data: &'a EmitterData, raw: &'a [u8]) -> Self {
        Self::with_seed(data, raw, crate::random::fixed_seed(raw))
    }

    /// The emitter as Emitter::Initialize starts it from `seed` (Random::SetSeed): its
    /// transform takes the first draws, its particles the rest.
    pub fn with_seed(data: &'a EmitterData, raw: &'a [u8], seed: u32) -> Self {
        let mut rng = EmitterRandom::seeded(seed);
        let (local, local_rt) = res_matrix(raw, &mut rng.lcg);
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
            timer: 0.0,
            interval: 0.0,
            accumulated: 0.0,
            particles: Vec::new(),
            current: local,
            previous_translation: None,
            set_scale: Vec3::ONE,
            particle_colour: [1.0; 4],
            distance: 0.0,
            emitter_velocity: Vec3::ZERO,
            divide_counter: 0,
            children: Vec::new(),
            spawn_offset: Vec3::ZERO,
            spawn_velocity: Vec3::ZERO,
            spawn_colours: None,
            killed: false,
            fields: Fields::default(),
            fade_out: 1.0,
            finished: false,
            frame_rate: 1.0,
        }
    }

    /// Kill the effect, as `EFFECT_OFF_KIND` does (EmitterCalculator::Calculate's fade path):
    /// from the next step the emitter stops emitting if res+0x754 says so, and fades out over
    /// res+0x768 frames if res+0x755 (alpha) or res+0x756 (scale) says so, then is gone.
    /// Without either fade flag its particles live out their lives.
    pub fn kill(&mut self) {
        self.killed = true;
    }

    /// Add a child emitter (`data` parsed, `raw` after load-time fixes): it emits from every
    /// live particle of this emitter, as the game's child emitters do (Smash 0x8d70).
    pub fn add_child(&mut self, data: &'a EmitterData, raw: &'a [u8]) -> &mut EmitterSim<'a> {
        self.add_child_with_seed(data, raw, crate::random::fixed_seed(raw))
    }

    /// [`EmitterSim::add_child`] with the child's seed (see [`SpawnSeeds`]).
    pub fn add_child_with_seed(
        &mut self,
        data: &'a EmitterData,
        raw: &'a [u8],
        seed: u32,
    ) -> &mut EmitterSim<'a> {
        let mut child = EmitterSim::with_seed(data, raw, seed);
        child.time = self.time;
        child.frame_rate = self.frame_rate;
        self.children.push(child);
        self.children.last_mut().unwrap()
    }

    /// Advance `rate` effect frames a step instead of one (the frame step the game hands
    /// EmitterCalculator::Calculate, Smash 0x8450): time, emission timers, fades and lives
    /// count by it, and particles move that share of their velocity and fields. The game's
    /// slow motion runs effects so: at a quarter, a particle's move in one step (what the
    /// shaders stretch sparks by) and what a child inherits from it are a quarter as long.
    /// Child emitters added later take the rate too.
    pub fn set_frame_rate(&mut self, rate: f32) {
        if rate.is_finite() && rate > 0.0 {
            self.frame_rate = rate;
            for child in &mut self.children {
                child.set_frame_rate(rate);
            }
        }
    }

    /// Run the emitter's field sections on its particles (see [`Fields`]). Without this they
    /// move only by their velocity, gravity and air resistance.
    pub fn set_fields(&mut self, fields: Fields) {
        self.fields = fields;
    }

    /// The child emitters, each drawn as its own emitter.
    pub fn children(&self) -> &[EmitterSim<'a>] {
        &self.children
    }

    /// Whether the emitter is gone and draws nothing.
    pub fn finished(&self) -> bool {
        self.finished
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

    /// Spawn `count` particles with the generator at `seed`, as the game's InitializeParticle
    /// does at emitter time `time`, and return the generator state afterwards. For checking the
    /// spawn port against generator states read from the running game.
    pub fn spawn_from_seed(&mut self, seed: u32, time: f32, count: usize) -> u32 {
        self.rng = EmitterRandom::seeded(seed);
        self.time = time;
        self.spawn(count);
        self.rng.lcg.0
    }

    /// Set the particle colour (`LAST_PARTICLE_SET_COLOR`): red, green, blue and opacity.
    pub fn set_particle_colour(&mut self, colour: [f32; 4]) {
        self.particle_colour = colour;
    }

    pub fn count(&self) -> usize {
        if self.finished {
            0
        } else {
            self.particles.len()
        }
    }

    /// One emission of `count` (Smash 0xb1c0): a draw shared by the whole emission first, then
    /// each particle. An equally divided circle or line whose divisions are not picked by
    /// res+0x874 emits every division `count` times (res+0x880 or +0x888, less a random
    /// percentage res+0x884 or +0x88c of them).
    fn spawn(&mut self, count: usize) {
        let batch = self.rng.next();
        let shape = &self.data.shape_info;
        let divisions = match shape.volume_type {
            2 => Some((0x880, 0x884)),
            13 => Some((0x888, 0x88c)),
            _ => None,
        };
        let count = match divisions {
            Some((n, random)) if i32_at(self.raw, 0x874) == 0 => {
                let n = u32_at(self.raw, n);
                let cut = (batch * u32_at(self.raw, random) as f32 * 0.01 * n as f32) as i64;
                count * (n as i64 - cut).max(0) as usize
            }
            _ => count,
        };
        // The renderer draws at most this many; a rate past it (or a corrupt one) must not
        // grow the particle list without bound.
        let count = count.min(MAX_PARTICLES.saturating_sub(self.particles.len()));
        let data = self.data;
        let raw = self.raw;
        let velocity = &data.particle_velocity;
        let vec3_at =
            |at: usize| Vec3::new(f32_at(raw, at), f32_at(raw, at + 4), f32_at(raw, at + 8));
        for index in 0..count {
            // InitializeParticle (Smash 0xa2d0), its draws in the game's order. The shape's
            // place and outward direction, the direction at the all-direction speed.
            let (mut position, normal) = spawn_shape(
                data,
                raw,
                &mut self.rng,
                &mut self.divide_counter,
                batch,
                index,
            );
            let mut outward = normal * velocity.all_direction;
            // XZ diffusion (res+0x97c): outward across the emitter's Y axis from its centre, a
            // random way for a particle on the axis.
            let xz = f32_at(raw, 0x97c);
            if xz != 0.0 {
                let mut flat = Vec2::new(position.x, position.z);
                if flat.length_squared() <= f32::EPSILON {
                    flat = Vec2::new(self.rng.signed(), self.rng.signed());
                }
                outward += Vec3::new(flat.x, 0.0, flat.y).normalize_or_zero() * xz;
            }
            // The speed's random share (res+0x98c), drawn here whether or not it is used.
            let speed = 1.0 - self.rng.next() * velocity.vel_random / 100.0;
            // Position random (res+0x810): that far out a direction of the runtime's table.
            let scatter = f32_at(raw, 0x810);
            if scatter != 0.0 {
                position += self.rng.unit() * scatter;
            }
            position += self.spawn_offset;
            // The designated direction (res+0x96c), spread by the diffusion angle (res+0x978,
            // degrees) over a cap about it: two draws, the turn about it and the height on the
            // cap, which runs linearly from 1 - angle / 90 to 1 (a whole sphere at 180).
            let designated = vec3_at(0x96c);
            let angle = f32_at(raw, 0x978);
            let along = if angle == 0.0 {
                designated
            } else {
                let around = self.rng.next() * std::f32::consts::TAU;
                let low = 1.0 - angle / 90.0;
                let height = low + (1.0 - low) * self.rng.next();
                let ring = (1.0 - height * height).max(0.0).sqrt();
                from_y(designated) * Vec3::new(ring * around.cos(), height, ring * around.sin())
            };
            let mut vel = (outward + along * velocity.designated_dir_scale) * speed;
            // Random velocity per axis (res+0x980) from the cube table, then the emitter's own
            // movement (emitter+0x3d0) times res+0x990: Sephiroth's charge sparks drop gold
            // dots that drift in after them (0.2), so each trail stretches as it goes.
            vel += self.rng.cube() * vec3_at(0x980);
            vel += self.emitter_velocity * f32_at(raw, 0x990);
            vel += self.spawn_velocity;
            // Scale: base (0x9c0..) * (1 - random% / 100 * r); one draw for every axis when the
            // x and y percentages match.
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
            // Life: life * (1 - randint(0..life_random) / 100), or 2^28 frames when infinite.
            let life = if raw.get(0x8a8).copied().unwrap_or(0) != 0 {
                268_435_456.0
            } else {
                let draw = self.rng.below(i32_at(raw, 0x8bc));
                i32_at(raw, 0x8b8) as f32 * (1.0 - draw * 0.01)
            };
            let random = [
                self.rng.next(),
                self.rng.next(),
                self.rng.next(),
                self.rng.next(),
            ];
            // InitRotate: the initial angle (0x700..), each axis cleared at load when that axis
            // does not rotate. No draw: the shaders add the random range (static block 0x710)
            // times the particle's random values. These offsets were 0x320.., the second
            // sampler's texture scale (1, 1), which turned every mesh particle 57 degrees about
            // X and Y: Sephiroth's charge ball was seen side on.
            let rotate = [
                f32_at(raw, 0x700),
                f32_at(raw, 0x704),
                f32_at(raw, 0x708),
                0.0,
            ];
            self.particles.push(Live {
                position,
                velocity: vel,
                diff: vel,
                life,
                birth: self.time,
                scale: scale4,
                random,
                rotate,
                birth_matrix: self.current,
                colours: self
                    .spawn_colours
                    .unwrap_or([self.particle_colour, self.particle_colour]),
                child_emission: vec![[0.0; 3]; self.children.len()],
            });
        }
    }

    /// One frame with the set where it was: emission, then motion; time advances by one.
    pub fn step(&mut self) {
        self.step_at(&Mat4::IDENTITY);
    }

    /// One frame with the set's matrix for this frame (EmitterCalculator::Calculate). A set
    /// that moves between frames is what distance emission measures, and each new particle
    /// keeps the matrix it was born with.
    pub fn step_at(&mut self, set: &Mat4) {
        self.current = *set * self.local;
        self.set_scale = set.to_scale_rotation_translation().0;
        let translation = self.current.w_axis.truncate();
        let moved = self
            .previous_translation
            .map_or(Vec3::ZERO, |previous| translation - previous);
        let previous = self.previous_translation.unwrap_or(translation);
        // EmitterCalculator::Calculate: the move since last frame through the inverse of the
        // emitter's matrix (emitter+0x3d0).
        self.emitter_velocity = if self.current.determinant().abs() > 1e-12 {
            self.current.inverse().transform_vector3(moved)
        } else {
            Vec3::ZERO
        };
        if self.fade_in < 1.0 {
            let frames = i32_at(self.raw, 0x76c);
            self.fade_in = if frames < 1 {
                1.0
            } else {
                (self.fade_in + self.frame_rate / frames as f32).min(1.0)
            };
        }
        let raw = self.raw;
        let flag = |at: usize| raw.get(at).copied().unwrap_or(0) != 0;
        let mut may_emit = true;
        if self.killed && !self.finished {
            may_emit = !flag(0x754);
            if flag(0x755) || flag(0x756) {
                let frames = i32_at(self.raw, 0x768);
                self.fade_out = if frames < 1 {
                    -1.0
                } else {
                    self.fade_out - self.frame_rate / frames as f32
                };
                if self.fade_out < 0.0 {
                    self.fade_out = 0.0;
                    self.finished = true;
                    self.particles.clear();
                }
            }
        }
        let step = self.frame_rate;
        if self.finished {
            self.time += step;
            self.step_children();
            return;
        }
        let frame = self.time;
        let emission = &self.data.emission;
        let start = emission.start as f32;
        // EmitterCalculator::Calculate: emission begins at res+0x7f4. A one-time emitter
        // (res+0x7f0) emits for res+0x7fc frames from there and then stops: sys_atk_smoke's
        // impact3 (rate 1.5 for 3 frames) has 1, 3 and then 4 particles in game, read live
        // from the emitter. Any other emitter keeps emitting until it is killed; the duration
        // does not stop it (only a child emitter's). Measured in game, Mario's fireball trail
        // (duration 4) keeps adding flames long after its fourth frame.
        let active = may_emit
            && frame >= start
            && (!emission.is_one_time || frame < start + emission.duration.max(1) as f32);
        // Rate randomness and interval spread are whole percentages and frames in the file
        // (res+0x804, +0x80c). The random share takes from the rate: TimeEmit divides it by
        // -100, so a 100% random rate of 1 emits between none and one a check.
        let rate_random = i32_at(self.raw, 0x804) as f32 / 100.0;
        if active && self.raw.get(0x7f2).copied().unwrap_or(0) != 0 {
            self.emit_by_distance(moved, previous);
        } else if active {
            // Time emission (Smash 0x7c30, the res+0x7f2 == 0 branch).
            if self.timer >= self.interval {
                let r = self.rng.next();
                let rate = emission.rate - emission.rate * rate_random * r;
                self.accumulated = (self.accumulated + rate).max(0.0);
                let n = self.accumulated as i32;
                if n == 0 {
                    self.timer = 0.0;
                } else {
                    let interval = self.interval;
                    self.spawn(n as usize);
                    self.interval = self.next_interval();
                    self.update_matrix_by_emit();
                    self.accumulated -= n as f32;
                    self.timer = (self.timer - interval) + step;
                }
            } else {
                self.timer += step;
            }
        }
        self.previous_translation = Some(translation);
        let statics = &self.data.emitter_static;
        let gravity = Vec3::new(
            statics.gravity_dir_x,
            statics.gravity_dir_y,
            statics.gravity_dir_z,
        ) * statics.gravity_scale;
        let air = statics.air_res;
        self.move_particles(gravity, air);
        self.time += step;
        // A particle draws `life` times, the frame it is born included: the game ages and
        // removes particles before it emits, so a new one always reaches the screen (live
        // probe: sys_attack_arc_b's arccylinder1, life 1, draws once; arccylinder2, life 3,
        // three times). The draw after this step reads time `now - step`.
        let now = self.time;
        self.particles.retain(|p| now - step - p.birth < p.life);
        self.step_children();
    }

    /// One frame of motion for every particle (CalculateParticleBehavior, Smash 0xf9f0):
    /// position by velocity times the particle's motion scale (its scale stream's w, which
    /// res+0x8c0 spreads), then air resistance and gravity on the velocity, then the field
    /// sections.
    fn move_particles(&mut self, gravity: Vec3, air: f32) {
        let follows = self.raw.get(0x753).copied().unwrap_or(0) != 1;
        let here = self.current.w_axis.truncate();
        let life = i32_at(self.raw, 0x8b8) as f32;
        let fields = self.fields;
        let run_fields = !fields.is_empty();
        let rate = self.frame_rate;
        // Air resistance per frame, compounded over the share of a frame a step is.
        let air = if rate == 1.0 { air } else { air.powf(rate) };
        for particle in &mut self.particles {
            let before = particle.position;
            let motion = particle.scale[3] * rate;
            particle.position += particle.velocity * motion;
            if air != 1.0 {
                particle.velocity *= air;
            }
            particle.velocity += gravity * rate;
            if run_fields {
                let matrix = if follows {
                    self.current
                } else {
                    particle.birth_matrix
                };
                // FieldConvergence's second type aims at where the emitter is now; a particle
                // that follows the emitter is already in its space, where that is the origin.
                let emitter_here = if follows {
                    Vec3::ZERO
                } else {
                    particle.birth_matrix.inverse().transform_point3(here)
                };
                fields.apply(crate::field::Particle {
                    position: &mut particle.position,
                    velocity: &mut particle.velocity,
                    age: self.time - particle.birth,
                    life,
                    random: particle.random[0],
                    randoms: particle.random,
                    rng: &mut self.rng,
                    air: f32_at(self.raw, 0xc0),
                    motion,
                    frame_rate: rate,
                    time: self.time,
                    matrix,
                    emitter_here,
                });
            }
            particle.diff = particle.position - before;
        }
    }

    /// Move a child emitter's particles one frame, without emitting.
    fn age(&mut self) {
        let statics = &self.data.emitter_static;
        let gravity = Vec3::new(
            statics.gravity_dir_x,
            statics.gravity_dir_y,
            statics.gravity_dir_z,
        ) * statics.gravity_scale;
        let air = statics.air_res;
        self.move_particles(gravity, air);
        let rate = self.frame_rate;
        self.time += rate;
        let now = self.time;
        self.particles.retain(|p| now - rate - p.birth < p.life);
    }

    /// Each child emitter after this emitter's frame (Smash 0x8d70): its particles age, then it
    /// emits from every live parent particle once the particle is res+0x7f8 percent of its
    /// life old, for res+0x7fc frames when it is a one-time emitter and else for the rest of
    /// the particle's life, by its own rate and interval, or by the distance the particle moved
    /// when it emits by distance (res+0x7f2). A child particle starts at its parent
    /// particle, in the parent's space (where the parent particle was born, when it does not
    /// follow the emitter), and takes its velocity (res+0x7d8, times res+0x7e8) and its
    /// colours and opacities (res+0x7dc..0x7df) as they are at that age.
    fn step_children(&mut self) {
        if self.children.is_empty() {
            return;
        }
        let step = self.frame_rate;
        let now = self.time - step;
        let follows = self.raw.get(0x753).copied().unwrap_or(0) != 1;
        let mut children = std::mem::take(&mut self.children);
        for (index, child) in children.iter_mut().enumerate() {
            child.age();
            let raw = child.raw;
            let flag = |at: usize| raw.get(at).copied().unwrap_or(0) != 0;
            let timing = u32_at(raw, 0x7f8) as f32 / 100.0;
            let one_time = flag(0x7f0);
            let duration = u32_at(raw, 0x7fc) as f32;
            let rate = child.data.emission.rate;
            let rate_random = i32_at(raw, 0x804) as f32 / 100.0;
            let by_distance = flag(0x7f2);
            for p in 0..self.particles.len() {
                let age = now - self.particles[p].birth;
                let start = self.particles[p].life * timing;
                if age < start || (one_time && age >= start + duration) {
                    continue;
                }
                let [timer, interval, accumulated] = self.particles[p].child_emission[index];
                let (n, state) = if by_distance {
                    // Smash 0x8d70 hands a child that emits by distance its parent particle's
                    // last move as the emitter's (emitter+0x3d0), from the particle's second
                    // frame on, and 0x7c30 emits one particle every res+0x824 of it.
                    if age == 0.0 {
                        (0, [timer, interval, accumulated])
                    } else {
                        child.emitter_velocity = self.particles[p].diff;
                        let (n, left) = child.distance_count(accumulated);
                        (n, [timer, interval, left])
                    }
                } else if timer >= interval {
                    let r = child.rng.next();
                    let amount = (accumulated + rate - rate * rate_random * r).max(0.0);
                    let n = amount as usize;
                    if n > 0 {
                        (n, [timer - interval + step, interval, amount - n as f32])
                    } else {
                        (0, [0.0, interval, amount])
                    }
                } else {
                    (0, [timer + step, interval, accumulated])
                };
                let mut state = state;
                if n > 0 {
                    // A time emission takes its next interval after the particles (Smash 0x30e0).
                    let next_interval = !by_distance;
                    let particle = &self.particles[p];
                    child.current = if follows {
                        self.current
                    } else {
                        particle.birth_matrix
                    };
                    child.spawn_offset = particle.position;
                    child.spawn_velocity = if flag(0x7d8) {
                        particle.velocity * f32_at(raw, 0x7e8)
                    } else {
                        Vec3::ZERO
                    };
                    let mut colours = [child.particle_colour; 2];
                    for (k, (rgb, alpha)) in [(0x7dc, 0x7de), (0x7dd, 0x7df)].iter().enumerate() {
                        let parent = self.colour_at(k, particle, age);
                        if flag(*rgb) {
                            colours[k][..3].copy_from_slice(&parent[..3]);
                        }
                        if flag(*alpha) {
                            colours[k][3] = parent[3];
                        }
                    }
                    child.spawn_colours = Some(colours);
                    child.spawn(n);
                    child.spawn_offset = Vec3::ZERO;
                    child.spawn_velocity = Vec3::ZERO;
                    child.spawn_colours = None;
                    if next_interval {
                        state[1] = child.next_interval();
                        child.update_matrix_by_emit();
                    }
                }
                child.emitter_velocity = Vec3::ZERO;
                self.particles[p].child_emission[index] = state;
            }
        }
        self.children = children;
    }

    /// A particle's colour `which` (0 or 1) and its opacity at `age` (Smash's
    /// CalculateParticleColor0/1VecFromTime): constant, a key picked by the particle's random
    /// value, or the 8-key curve over its life; times the emitter's colour (res+0x7ac / 0x7bc).
    fn colour_at(&self, which: usize, particle: &Live, age: f32) -> [f32; 4] {
        use effect_library::enums::ColorType;
        let c = &self.data.particle_color;
        let s = &self.data.emitter_static;
        let (kind, alpha_kind, constant, keys, alpha_keys, count, alpha_count, emitter) =
            if which == 0 {
                (
                    &c.color0_type,
                    &c.alpha0_type,
                    [c.color0_r, c.color0_g, c.color0_b, c.alpha0],
                    &s.color0,
                    &s.alpha0,
                    s.num_color0_keys,
                    s.num_alpha0_keys,
                    0x7ac,
                )
            } else {
                (
                    &c.color1_type,
                    &c.alpha1_type,
                    [c.color1_r, c.color1_g, c.color1_b, c.alpha1],
                    &s.color1,
                    &s.alpha1,
                    s.num_color1_keys,
                    s.num_alpha1_keys,
                    0x7bc,
                )
            };
        let t = age / particle.life.max(1.0);
        let pick = |kind: &ColorType,
                    table: &effect_library::emitter::AnimationKeyTable,
                    count: u32|
         -> Option<[f32; 3]> {
            let keys = &table.keys[..(count as usize).min(8)];
            let at = |k: &effect_library::emitter::AnimationKey| [k.x, k.y, k.z];
            match kind {
                ColorType::Animated8Key if !keys.is_empty() => {
                    if keys.len() == 1 || t <= keys[0].time {
                        return Some(at(&keys[0]));
                    }
                    let last = keys[keys.len() - 1];
                    if t >= last.time {
                        return Some(at(&last));
                    }
                    let i = keys.windows(2).position(|w| t < w[1].time).unwrap_or(0);
                    let (a, b) = (keys[i], keys[i + 1]);
                    let f = (t - a.time) / (b.time - a.time).max(1e-6);
                    Some([
                        a.x + (b.x - a.x) * f,
                        a.y + (b.y - a.y) * f,
                        a.z + (b.z - a.z) * f,
                    ])
                }
                // Type 3 picks a key by the particle's first random value. effect_library calls
                // type 1 Random, which no game file uses; 3 reads as Unknown(3).
                ColorType::Unknown(3) | ColorType::Random if !keys.is_empty() => {
                    let i = ((particle.random[0] * keys.len() as f32) as usize).min(keys.len() - 1);
                    Some(at(&keys[i]))
                }
                _ => None,
            }
        };
        let rgb = pick(kind, keys, count).unwrap_or([constant[0], constant[1], constant[2]]);
        let alpha = pick(alpha_kind, alpha_keys, alpha_count).map_or(constant[3], |a| a[0]);
        [
            rgb[0] * f32_at(self.raw, emitter),
            rgb[1] * f32_at(self.raw, emitter + 4),
            rgb[2] * f32_at(self.raw, emitter + 8),
            alpha * f32_at(self.raw, emitter + 12),
        ]
    }

    /// The next emission interval after an emission (Smash 0x30e0): res+0x808 frames plus one
    /// plus a random whole number below res+0x80c.
    fn next_interval(&mut self) -> f32 {
        let draw = self.rng.below(i32_at(self.raw, 0x80c));
        i32_at(self.raw, 0x808) as f32 + 1.0 + draw
    }

    /// After a time emission's interval (Smash 0x30e0): when res+0x758 asks, the emitter's own
    /// transform is drawn again with its random spreads (six draws, as CreateResMatrix), so
    /// each emission comes from a new place and angle.
    fn update_matrix_by_emit(&mut self) {
        if self.raw.get(0x758).copied().unwrap_or(0) != 0 {
            (self.local, self.local_rt) = res_matrix(self.raw, &mut self.rng.lcg);
        }
    }

    /// A move's length as distance emission counts it (Smash 0x7c30): zero under the margin
    /// res+0x830, else clamped to res+0x828..res+0x82c.
    fn distance_step(&self, length: f32) -> f32 {
        let step = if length >= f32_at(self.raw, 0x830) {
            length
        } else {
            0.0
        };
        let (min, max) = (f32_at(self.raw, 0x828), f32_at(self.raw, 0x82c));
        if step != 0.0 && (step < min || step > max) {
            step.clamp(min, max)
        } else {
            step
        }
    }

    /// Distance emission's count from the distance not yet emitted for and the emitter's move
    /// this frame (emitter+0x3d0): every res+0x824 of the total is one particle. Returns the
    /// count and what is left over.
    fn distance_count(&self, carried: f32) -> (usize, f32) {
        let step = self.distance_step(self.emitter_velocity.length());
        let unit = f32_at(self.raw, 0x824);
        let total = carried + step;
        if unit == 0.0 {
            return (0, total);
        }
        let n = ((total / unit) as i32).max(0);
        (n as usize, total - n as f32 * unit)
    }

    /// Distance emission (Smash 0x7c30, the res+0x7f2 branch): the distance the emitter moved
    /// this frame (zero under the margin res+0x830, else clamped to res+0x828..res+0x82c) adds
    /// to a running total, and every res+0x824 of it is one particle, placed along the path
    /// from last frame's position to this one.
    fn emit_by_distance(&mut self, moved: Vec3, previous: Vec3) {
        let step = self.distance_step(moved.length());
        let unit = f32_at(self.raw, 0x824);
        let mut total = self.distance + step;
        if unit != 0.0 {
            let n = (total / unit) as i32;
            let current = self.current;
            let here = current.w_axis.truncate();
            for _ in 0..n.max(0) {
                total -= unit;
                let t = if step == 0.0 { 0.0 } else { total / step };
                let at = here + (previous - here) * t;
                self.current.w_axis = at.extend(1.0);
                self.spawn(1);
            }
            self.current = current;
        }
        self.distance = total;
    }

    /// The per-particle streams the shaders read, by attribute name.
    pub fn streams(&self) -> HashMap<&'static str, Vec<[f32; 4]>> {
        let mut out: HashMap<&'static str, Vec<[f32; 4]>> = HashMap::new();
        for p in &self.particles {
            let rows = p.birth_matrix.transpose().to_cols_array_2d();
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
            out.entry("sysColor0Attr").or_default().push(p.colours[0]);
            out.entry("sysColor1Attr").or_default().push(p.colours[1]);
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
        // The game draws a frame before its time advances: after the spawn frame's Calculate
        // the dynamic block holds time 0 and the fade-in has already taken its first step
        // (live capture: time 4 with fade 5/6). `self.time` counts steps taken, one ahead.
        v[8] = (self.time - self.frame_rate).max(0.0);
        v[9] = self.particles.len() as f32;
        // MakeDynamicConstantBuffer's param_2 and frame step: 0 and 1 for an ordinary draw
        // (captured: the game's fireball disc block reads (time, count, 0, 1)).
        v[10] = 0.0;
        v[11] = self.frame_rate;
        // MakeDynamicConstantBuffer: the fade-in where res+0x75b / 0x75c asks for it, times
        // the fade-out after a kill where res+0x755 / 0x756 does.
        let on = |flag: usize| r.get(flag).copied().unwrap_or(0) != 0;
        let fade = |fade_in: usize, fade_out: usize| {
            (if on(fade_in) { self.fade_in } else { 1.0 })
                * (if on(fade_out) { self.fade_out } else { 1.0 })
        };
        v[12] = fade(0x75b, 0x755);
        // The particle scale: the set's scale times the scale fade. Particles are drawn with
        // the unscaled matrix (`rt`) and this, so leaving the set's scale out drew every
        // particle of a scaled effect at size 1 while its emitter moved as scaled.
        let fade_scale = fade(0x75c, 0x756);
        v[13] = fade_scale * self.set_scale.x;
        v[14] = fade_scale * self.set_scale.y;
        v[15] = fade_scale * self.set_scale.z;
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

/// The rotation that turns +Y to `to` (the shortest one; half a turn about X when `to` points
/// down), as InitializeParticle and the sphere shapes build it.
fn from_y(to: Vec3) -> Quat {
    let to = to.normalize_or_zero();
    if to == Vec3::ZERO {
        Quat::IDENTITY
    } else if to.y <= -1.0 + 1e-6 {
        Quat::from_rotation_x(std::f32::consts::PI)
    } else {
        Quat::from_rotation_arc(Vec3::Y, to)
    }
}

/// Spawn position and outward direction by volume type (g_EmitFunctions order). `batch` is the
/// draw the whole emission shares and `index` the particle's place in it.
fn spawn_shape(
    data: &EmitterData,
    raw: &[u8],
    rng: &mut EmitterRandom,
    divide_counter: &mut i32,
    batch: f32,
    index: usize,
) -> (Vec3, Vec3) {
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
    // The sweep is centred on its start angle: sys_atk_smoke's impact3 (start pi, sweep 163
    // degrees) throws its particles backward to both sides in game, read live from the
    // emitter, where a sweep starting at pi would throw half of them forward.
    // res+0x839: every emission starts its sweep at a random angle (the shared draw).
    let start = if shape.sweep_start_random != 0 {
        std::f32::consts::TAU * batch
    } else {
        shape.sweep_start
    };
    let first = start - sweep * 0.5;
    let circle = |angle: f32, fill: f32| {
        let dir = Vec3::new(angle.sin(), 0.0, angle.cos());
        (dir * radius * fill, dir)
    };
    // Circle fill (Smash 0xb6f0): a draw for the angle, then one for the distance out, the
    // square root of a uniform share of the ring's area between 1 - caliber and 1; the
    // direction leans to the longer axis of an oval.
    let circle_fill = |rng: &mut EmitterRandom| {
        let angle = first + rng.next() * sweep;
        let inner = 1.0 - shape.caliber_ratio;
        let r = rng.next();
        let fill = (r + (1.0 - r) * inner * inner).max(0.0).sqrt();
        let (sin, cos) = angle.sin_cos();
        let (mut x, mut z) = (sin * fill, cos * fill);
        if radius.x <= radius.z {
            x *= if radius.z != 0.0 {
                radius.x / radius.z
            } else {
                0.0
            };
        } else {
            z *= radius.z / radius.x;
        }
        let dir = Vec3::new(x, 0.0, z).try_normalize().unwrap_or(Vec3::Z);
        (
            Vec3::new(sin * fill * radius.x, 0.0, cos * fill * radius.z),
            dir,
        )
    };
    // Sphere and sphere fill (Smash 0xb9d0, 0xc890): a draw for the angle about the axis (the
    // swept longitude, or a whole turn with a latitude, Smash 0xd810), one for the height
    // (all of -1..1, or from cos(latitude) to 1 about the latitude axis res+0x83e).
    let sphere = |rng: &mut EmitterRandom| {
        // The runtime reads res+0x83a for this (effect_library calls it arc_type).
        let latitude = raw.get(0x83a).copied() == Some(1);
        let angle = if latitude {
            rng.next() * std::f32::consts::TAU
        } else {
            first + rng.next() * sweep
        };
        let height = if latitude {
            1.0 - rng.next() * (1.0 - f32_at(raw, 0x844).cos())
        } else {
            rng.signed()
        };
        let ring = (1.0 - height * height).max(0.0).sqrt();
        let mut dir = Vec3::new(ring * angle.sin(), height, ring * angle.cos());
        if latitude {
            let axis = match shape.volume_latitude_dir {
                0 => Vec3::X,
                1 => Vec3::NEG_X,
                3 => Vec3::NEG_Y,
                4 => Vec3::Z,
                5 => Vec3::NEG_Z,
                _ => Vec3::Y,
            };
            dir = from_y(axis) * dir;
        }
        dir
    };
    match shape.volume_type {
        // Point (Smash 0xb1d0): no place, a direction from the runtime's table.
        0 => (Vec3::ZERO, rng.unit()),
        1 => circle(first + rng.next() * sweep, 1.0),
        3 => circle_fill(rng),
        2 => {
            // Smash 0xb3f0: res+0x880 divisions of the sweep (one less on an open arc, so both
            // ends get one). Which division: the place in the emission (res+0x874 = 0), a random
            // one (1) or the next in turn (2). Each particle then turns by up to res+0x84c
            // either way.
            let sweep = shape.sweep_longitude;
            let full = sweep == std::f32::consts::TAU;
            let n = u32_at(raw, 0x880) as i64;
            let (n, division) = match i32_at(raw, 0x874) {
                0 => {
                    let cut = (u32_at(raw, 0x884) as f32 * batch * 0.01 * n as f32) as i64;
                    let n = n - cut;
                    (n - i64::from(!full && n > 1), index as i64)
                }
                kind => {
                    let n = n - i64::from(n > 1 && !full);
                    let division = match kind {
                        2 => {
                            let at = *divide_counter as i64;
                            *divide_counter = if at + 1 < n { at as i32 + 1 } else { 0 };
                            if n == 0 {
                                0
                            } else {
                                at % n
                            }
                        }
                        1 => (rng.next() * n as f32) as i64,
                        _ => index as i64,
                    };
                    (n, division)
                }
            };
            let n = n.max(1) as f32;
            let jitter = rng.next() - 0.5;
            let angle = start
                + (sweep / n * division as f32 - sweep * 0.5)
                + shape.volume_surface_pos_rand * 2.0 * jitter;
            circle(angle, 1.0)
        }
        // The equally divided spheres (5, 6) take their directions from the runtime's tables
        // of 32 and 64; these draw them as a sphere does.
        4..=6 => {
            let dir = sphere(rng);
            (dir * radius, dir)
        }
        7 => {
            let dir = sphere(rng);
            let fill = rng.next().sqrt() * shape.caliber_ratio + 1.0 - shape.caliber_ratio;
            (dir * radius * fill, dir)
        }
        // Cylinder and its fill (Smash 0xce80, 0xcf10): the circle, then a height.
        8 | 9 => {
            let (mut position, dir) = if shape.volume_type == 9 {
                circle_fill(rng)
            } else {
                circle(first + rng.next() * sweep, 1.0)
            };
            position.y = rng.signed() * radius.y;
            (position, dir)
        }
        10 => {
            let position = Vec3::new(rng.signed(), rng.signed(), rng.signed()) * radius;
            (position, position.normalize_or_zero())
        }
        // Box fill (Smash 0xd140): a draw picks which of the hollow box's slabs, three place
        // the point in it and three more flip each axis.
        11 => {
            let inner = 1.0 - shape.caliber_ratio;
            let inner = if inner == 1.0 { 0.999 } else { inner };
            let pick = rng.next() * (1.0 - inner * inner * inner);
            let mut p = Vec3::new(rng.next(), rng.next(), rng.next());
            let outer = 1.0 - inner;
            if pick < outer {
                p.y = inner + outer * p.y;
            } else if pick < outer + inner * outer {
                p.x = inner + outer * p.x;
                p.y *= inner;
            } else {
                p.x *= inner;
                p.y *= inner;
                p.z = inner + outer * p.z;
            }
            for axis in 0..3 {
                if rng.next() < 0.5 {
                    p[axis] = -p[axis];
                }
            }
            let position = p * radius;
            (position, position.normalize_or_zero())
        }
        // Smash 0xd360 and 0xd3f0: along z, centred less half the centre ratio of the length;
        // the divided line puts its res+0x888 divisions end to end.
        12 | 13 => {
            let length = shape.line_length;
            let t = if shape.volume_type == 13 {
                let n = u32_at(raw, 0x888) as i64;
                let (n, division) = match i32_at(raw, 0x874) {
                    2 => {
                        let at = *divide_counter as i64;
                        *divide_counter = if at + 1 < n { at as i32 + 1 } else { 0 };
                        (n, if n == 0 { 0 } else { at % n })
                    }
                    1 => (n, (rng.next() * n as f32) as i64),
                    0 => {
                        let cut = (u32_at(raw, 0x88c) as f32 * batch * 0.01 * n as f32) as i64;
                        (n - cut, index as i64)
                    }
                    _ => (n, index as i64),
                };
                if n - 1 == 0 {
                    0.5
                } else {
                    division as f32 / (n - 1) as f32
                }
            } else {
                rng.next()
            };
            (
                Vec3::new(
                    0.0,
                    0.0,
                    length * t - (length + length * shape.line_center) * 0.5,
                ),
                Vec3::Z,
            )
        }
        14 => (
            Vec3::new(rng.signed() * radius.x, 0.0, rng.signed() * radius.z),
            Vec3::Y,
        ),
        _ => (Vec3::ZERO, rng.unit()),
    }
}
