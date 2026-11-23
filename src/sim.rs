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
    /// Which of its emitter's particles this is, for the children that follow it.
    id: u64,
    /// For a particle of a child emitted once per parent particle (res+0x7e4): the parent
    /// particle it rides with and where that particle was last frame.
    anchor: Option<(u64, Vec3)>,
}

/// What a child particle takes from its parent particle (Smash 0x9ab0, run after
/// InitializeParticle for each particle a child emitter emits).
#[derive(Debug, Clone, Copy, Default)]
struct Inherited {
    /// The parent particle's position: added to the particle's own.
    offset: Vec3,
    /// The parent particle's velocity times res+0x7e8, when res+0x7d8 asks for it.
    velocity: Vec3,
    /// The parent particle's scale at its age times res+0x7ec (res+0x7d9): replaces the
    /// particle's own base scale; its own scale curve still runs on top in the shader.
    scale: Option<Vec3>,
    /// The parent particle's rotation at its age (res+0x7da): replaces the initial rotation.
    rotate: Option<Vec3>,
    /// Colour 0 and 1: the parent's colour (res+0x7dc, 0x7dd) and opacity (res+0x7de, 0x7df).
    colours: [[Option<f32>; 4]; 2],
    anchor: Option<(u64, Vec3)>,
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
    /// For a child emitter, its parent's own transform in the set (with scale and without) and
    /// whether the parent's particles keep their birth matrix (parent res+0x753 = 1), which
    /// leaves the child in the world on its own transform alone (Smash 0x8090).
    parent: Option<(Mat4, Mat4, bool)>,
    /// What a child's next particles take from the parent particle they are emitted from.
    inherit: Option<Inherited>,
    /// For a child emitter: its parent's emitter opacities (res+0x7b8, 0x7c8) and fade this
    /// frame, which its own multiply by when it takes the parent's opacity every frame
    /// (res+0x7e2, 0x7e3; MakeDynamicConstantBuffer, Smash 0x9770).
    parent_alpha: [f32; 2],
    parent_fade: f32,
    /// The id the next particle takes.
    next_id: u64,
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
            parent: None,
            inherit: None,
            parent_alpha: [1.0; 2],
            parent_fade: 1.0,
            next_id: 0,
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

    /// The emitter's matrix (emitter+0x340, UpdateEmitterMatrix, Smash 0x8090): the set's
    /// matrix times the emitter's own. A child emitter's is its parent's matrix times its own,
    /// or its own alone when its parent's particles keep their birth matrix, since those hand
    /// it their positions in the world. Sephiroth's Scintilla wall (P_EdgeSenkouShield) is
    /// seven child rows of hexagons, each row placed by its own translation (9.9 down to -13.8
    /// in Y); without it all seven rows drew on one line.
    pub fn emitter_srt(&self, set: &Mat4) -> Mat4 {
        match self.parent {
            None => *set * self.local,
            Some((parent, _, false)) => *set * parent * self.local,
            Some((_, _, true)) => self.local,
        }
    }

    /// The emitter's matrix without scale (emitter+0x380), built as [`EmitterSim::emitter_srt`]
    /// is from the set's matrix without scale.
    pub fn emitter_rt(&self, set_rt: &Mat4) -> Mat4 {
        match self.parent {
            None => *set_rt * self.local_rt,
            Some((_, parent, false)) => *set_rt * parent * self.local_rt,
            Some((_, _, true)) => self.local_rt,
        }
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
        let inherit = self.inherit.unwrap_or_default();
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
            position += inherit.offset;
            // The designated direction (res+0x96c), spread by the diffusion angle (res+0x978,
            // degrees) over a cap about it: two draws, the turn about it and the height on the
            // cap, which runs linearly from 1 - angle / 90 to 1 (a whole sphere at 180).
            // A world-oriented velocity (res+0x7f3) names its direction in the world: the
            // game takes it through the inverse of the emitter's matrix, so the shaders, which
            // turn every velocity by that matrix, send the particles that way however the
            // effect is turned.
            let designated = if raw.get(0x7f3).copied().unwrap_or(0) != 0 {
                world_to_emitter(&self.current, vec3_at(0x96c))
            } else {
                vec3_at(0x96c)
            };
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
            vel += inherit.velocity;
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
            let mut scale4 = [
                f32_at(raw, 0x9c0) * factors[0],
                f32_at(raw, 0x9c4) * factors[1],
                f32_at(raw, 0x9c8) * factors[2],
                w,
            ];
            if let Some(s) = inherit.scale {
                scale4[..3].copy_from_slice(&s.to_array());
            }
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
            let mut rotate = [
                f32_at(raw, 0x700),
                f32_at(raw, 0x704),
                f32_at(raw, 0x708),
                0.0,
            ];
            if let Some(r) = inherit.rotate {
                rotate[..3].copy_from_slice(&r.to_array());
            }
            let mut colours = [self.particle_colour; 2];
            for (colour, taken) in colours.iter_mut().zip(inherit.colours) {
                for (channel, value) in colour.iter_mut().zip(taken) {
                    if let Some(value) = value {
                        *channel = value;
                    }
                }
            }
            let id = self.next_id;
            self.next_id += 1;
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
                colours,
                child_emission: vec![[0.0; 3]; self.children.len()],
                id,
                anchor: inherit.anchor,
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
        // World gravity (res+0x7f1) pulls along the world's axes: the game turns it into the
        // emitter's space through the emitter's rotation (its matrix without scale, or the one
        // the particle was born with when particles do not follow the emitter). Otherwise it
        // pulls along the emitter's own axes, and turns with the effect.
        let world_gravity = self.raw.get(0x7f1).copied().unwrap_or(0) != 0;
        let current_gravity = world_gravity_in(&self.current, gravity);
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
            let pull = match (world_gravity, follows) {
                (false, _) => gravity,
                (true, true) => current_gravity,
                (true, false) => world_gravity_in(&particle.birth_matrix, gravity),
            };
            particle.velocity += pull * rate;
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
    /// when it emits by distance (res+0x7f2). The child's matrix is this emitter's times the
    /// child's own (Smash 0x8090); a child particle starts at its shape's place plus the parent
    /// particle's position (Smash 0x9ab0), both in the child's space. When this emitter's
    /// particles keep their birth matrix (res+0x753 = 1), the parent particle's position and
    /// velocity are first taken into the world through it, and the child's matrix is its own
    /// alone. What else a child particle takes from its parent particle is in
    /// [`EmitterSim::inherited`]. The child draws at the set's scale, as every emitter of the
    /// set does (MakeDynamicConstantBuffer, Smash 0x9770, reads it from emitter+0x78, the set).
    ///
    /// A child that is one emitter per parent particle (res+0x7e4; the game makes it when the
    /// particle is born, Smash 0xa2d0) is placed at that particle every frame (Smash 0x8d70's
    /// second loop writes emitter+0x5c0, and 0x8090 adds it to the emitter's translation), so
    /// its particles that follow their emitter ride with the parent particle.
    fn step_children(&mut self) {
        if self.children.is_empty() {
            return;
        }
        let step = self.frame_rate;
        let now = self.time - step;
        let follows = self.raw.get(0x753).copied().unwrap_or(0) != 1;
        let parent_alpha = [f32_at(self.raw, 0x7b8), f32_at(self.raw, 0x7c8)];
        let parent_fade = self.fade(0x75b, 0x755);
        let mut children = std::mem::take(&mut self.children);
        let mut anchors: Option<HashMap<u64, Vec3>> = None;
        for (index, child) in children.iter_mut().enumerate() {
            child.parent = Some((self.local, self.local_rt, !follows));
            child.set_scale = self.set_scale;
            child.parent_alpha = parent_alpha;
            child.parent_fade = parent_fade;
            child.current = if follows {
                self.current * child.local
            } else {
                child.local
            };
            child.age();
            let raw = child.raw;
            let flag = |at: usize| raw.get(at).copied().unwrap_or(0) != 0;
            let per_particle = flag(0x7e4);
            if per_particle && raw.get(0x753).copied().unwrap_or(0) != 1 {
                let anchors = anchors.get_or_insert_with(|| {
                    self.particles
                        .iter()
                        .map(|p| (p.id, anchor_of(p, follows)))
                        .collect()
                });
                child.follow(anchors);
            }
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
                    child.inherit = Some(self.inherited(child, &self.particles[p], age, follows));
                    child.spawn(n);
                    child.inherit = None;
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

    /// What `child`'s particles emitted now from `particle` (this emitter's, `age` frames old)
    /// take from it (Smash 0x9ab0). Its position and, when res+0x7d8 asks, its velocity times
    /// res+0x7e8; its scale at its age (res+0x7d9) times res+0x7ec; its rotation at its age
    /// (res+0x7da); its colour and opacity 0 and 1 at its age (res+0x7dc..0x7df). The game
    /// reads that age one frame step back for a child shared by every particle, and as it is
    /// for a child that is one emitter per particle (emitter+0x564, written by Smash 0x8d70).
    /// A child per particle takes the particle's velocity turned by this emitter's matrix
    /// without scale (emitter+0x5d0) and sits where the particle is: its translation, in this
    /// emitter's space, is the particle's position (emitter+0x5c0), so in the child's own
    /// space that place is turned back through the child's transform. Res+0x7db (colour
    /// scale) and res+0x7e0 (draw path) have no reader in the effect library's code
    /// (Smash 0x0..0x80000).
    fn inherited(&self, child: &EmitterSim, particle: &Live, age: f32, follows: bool) -> Inherited {
        let raw = child.raw;
        let flag = |at: usize| raw.get(at).copied().unwrap_or(0) != 0;
        let per_particle = flag(0x7e4);
        let at = if per_particle {
            age
        } else {
            age - self.frame_rate
        };
        let (position, velocity) = if follows {
            (particle.position, particle.velocity)
        } else {
            (
                particle.birth_matrix.transform_point3(particle.position),
                particle.birth_matrix.transform_vector3(particle.velocity),
            )
        };
        let (offset, velocity) = if per_particle {
            let own = child.local;
            let offset = if own.determinant().abs() > 1e-12 {
                own.inverse().transform_vector3(position)
            } else {
                position
            };
            (offset, unscaled(&self.current) * particle.velocity)
        } else {
            (position, velocity)
        };
        let mut colours = [[None; 4]; 2];
        for (k, (rgb, alpha)) in [(0x7dc, 0x7de), (0x7dd, 0x7df)].into_iter().enumerate() {
            if flag(rgb) || flag(alpha) {
                let parent = self.colour_at(k, particle, at);
                if flag(rgb) {
                    for c in 0..3 {
                        colours[k][c] = Some(parent[c]);
                    }
                }
                if flag(alpha) {
                    colours[k][3] = Some(parent[3]);
                }
            }
        }
        Inherited {
            offset,
            velocity: if flag(0x7d8) {
                velocity * f32_at(raw, 0x7e8)
            } else {
                Vec3::ZERO
            },
            scale: flag(0x7d9).then(|| self.scale_at(particle, at) * f32_at(raw, 0x7ec)),
            rotate: flag(0x7da).then(|| self.rotation_at(particle, at)),
            colours,
            anchor: per_particle.then_some((particle.id, position)),
        }
    }

    /// Move the particles of a child that is one emitter per parent particle by how far their
    /// parent particle moved since last frame (`anchors`: each live parent particle's place),
    /// in the child's own space. A particle whose parent particle is gone stays where it is:
    /// the game stops placing that emitter (Smash 0x8d70 only places emitters whose particle
    /// is alive).
    fn follow(&mut self, anchors: &HashMap<u64, Vec3>) {
        let own = self.local;
        let to_own = |v: Vec3| {
            if own.determinant().abs() > 1e-12 {
                own.inverse().transform_vector3(v)
            } else {
                v
            }
        };
        for particle in &mut self.particles {
            let Some((id, last)) = particle.anchor.as_mut() else {
                continue;
            };
            if let Some(&now) = anchors.get(id) {
                particle.position += to_own(now - *last);
                *last = now;
            }
        }
    }

    /// The fade-in where `fade_in` (res+0x75b or 0x75c) asks for it times the fade-out after a
    /// kill where `fade_out` (res+0x755 or 0x756) does (MakeDynamicConstantBuffer).
    fn fade(&self, fade_in: usize, fade_out: usize) -> f32 {
        let on = |flag: usize| self.raw.get(flag).copied().unwrap_or(0) != 0;
        (if on(fade_in) { self.fade_in } else { 1.0 })
            * (if on(fade_out) { self.fade_out } else { 1.0 })
    }

    /// A particle's scale at `age` (CalculateParticleScaleVecFromTime, Smash 0xffe0): its
    /// scale stream times the scale curve (res+0x600, res+0x70 keys; the first key alone under
    /// two), then, when res+0x9e5 asks, a wave on X and Y (see [`wave`]).
    fn scale_at(&self, particle: &Live, age: f32) -> Vec3 {
        let raw = self.raw;
        let flag = |at: usize| raw.get(at).copied().unwrap_or(0) != 0;
        let count = i32_at(raw, 0x70);
        let curve = if count < 2 {
            key(raw, 0x600, 0).0
        } else {
            let rate = if flag(0x8dc) {
                i32_at(raw, 0x8f4) as f32
            } else {
                0.0
            };
            let random_loop = raw.get(0x8e1).copied().unwrap_or(0) as f32;
            let k = Keys {
                table: 0x600,
                count,
                loop_rate: rate,
                random_loop,
            };
            k.at(raw, particle.random[0], age, particle.life)
        };
        let mut scale = Vec3::from_slice(&particle.scale[..3]) * curve;
        if flag(0x9e5) {
            let x = wave(raw, particle.random[0], age, 0);
            let y = if flag(0x9e6) {
                wave(raw, particle.random[0], age, 1)
            } else {
                x
            };
            scale.x *= x;
            scale.y *= y;
        }
        scale
    }

    /// A particle's rotation at `age` (CalculateRotationMatrix, Smash 0x114f0): its initial
    /// rotation plus the random range (res+0x710) times its random values, turning at the
    /// rotation speed (res+0x720) plus its random share (res+0x730) slowed by the rotation's
    /// air resistance (res+0x72c). Each axis turns the other way, initial angle included,
    /// when its flag (res+0x8ad..0x8af) is set and the particle's next random value is at
    /// least a half.
    fn rotation_at(&self, particle: &Live, age: f32) -> Vec3 {
        let raw = self.raw;
        let v3 = |at: usize| Vec3::new(f32_at(raw, at), f32_at(raw, at + 4), f32_at(raw, at + 8));
        let (range, speed, speed_random) = (v3(0x710), v3(0x720), v3(0x730));
        let r = particle.random;
        let mut out = Vec3::ZERO;
        for axis in 0..3 {
            let next = r[(axis + 1) % 3];
            let flip = raw.get(0x8ad + axis).copied().unwrap_or(0) != 0 && next >= 0.5;
            let sign = if flip { -1.0 } else { 1.0 };
            let base = range[axis] * r[axis] + sign * particle.rotate[axis];
            let turn = sign * (speed[axis] + speed_random[axis] * (r[axis] + next) * 0.5);
            out[axis] = base + turn * resisted(f32_at(raw, 0x72c), age);
        }
        out
    }

    /// A particle's colour `which` (0 or 1) and its opacity at `age` (Smash's
    /// CalculateParticleColor0VecFromTime 0x10490 and Color1 0x10cc0): the constant colour
    /// (res+0x9a0 / 0x9b0), or for colour type 3 a key picked by the particle's first random
    /// value, or for type 2 the 8-key curve; the opacity's curve for opacity type 2. Then
    /// times the colour scale (res+0x3b0) and the emitter's colour and opacity, which the game
    /// passes from colour 0 for both (emitter+0x624 and +0x654, from res+0x7ac and 0x7b8), and,
    /// when res+0x9e4 asks, the opacity times a wave (see [`wave`]), kept between 0 and 1.
    fn colour_at(&self, which: usize, particle: &Live, age: f32) -> [f32; 4] {
        let raw = self.raw;
        let byte = |at: usize| raw.get(at).copied().unwrap_or(0);
        let w = which;
        let constant = 0x9a0 + 0x10 * w;
        let mut out = [
            f32_at(raw, constant),
            f32_at(raw, constant + 4),
            f32_at(raw, constant + 8),
            f32_at(raw, constant + 12),
        ];
        let random = particle.random[0];
        // Colour then opacity: type byte, key table, key count, loop flag, loop rate and
        // random loop flag.
        let colour = (0x99c + w, 0x3c0 + 0x100 * w, 0x60 + 8 * w, 0x8d8 + 2 * w);
        let alpha = (0x99e + w, 0x440 + 0x100 * w, 0x64 + 8 * w, 0x8d9 + 2 * w);
        let keys = |(_, table, count, looped): (usize, usize, usize, usize)| Keys {
            table,
            count: i32_at(raw, count),
            loop_rate: if byte(looped) != 0 {
                i32_at(raw, 0x8e4 + 4 * (looped - 0x8d8)) as f32
            } else {
                0.0
            },
            random_loop: byte(0x8dd + (looped - 0x8d8)) as f32,
        };
        match byte(colour.0) {
            3 => {
                let n = i32_at(raw, colour.2);
                let i = (random * n as f32) as i32;
                out[..3].copy_from_slice(&key(raw, colour.1, i).0.to_array());
            }
            2 if i32_at(raw, colour.2) >= 1 => {
                let rgb = keys(colour).at(raw, random, age, particle.life);
                out[..3].copy_from_slice(&rgb.to_array());
            }
            _ => {}
        }
        if byte(alpha.0) == 2 && i32_at(raw, alpha.2) >= 1 {
            out[3] = keys(alpha).at(raw, random, age, particle.life).x;
        }
        let scale = f32_at(raw, 0x3b0);
        for (c, value) in out.iter_mut().take(3).enumerate() {
            *value *= f32_at(raw, 0x7ac + 4 * c) * scale;
        }
        out[3] *= f32_at(raw, 0x7b8);
        if byte(0x9e4) != 0 {
            out[3] *= wave(raw, random, age, 0);
            out[3] = out[3].clamp(0.0, 1.0);
        }
        out
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
        // the fade-out after a kill where res+0x755 / 0x756 does. A child that takes its
        // parent particle's opacity 0 or 1 every frame (res+0x7de with 0x7e2, res+0x7df with
        // 0x7e3) multiplies that opacity by its parent emitter's, and its fade by its parent's.
        let on = |flag: usize| r.get(flag).copied().unwrap_or(0) != 0;
        let each = |alpha: usize, every: usize| self.parent.is_some() && on(alpha) && on(every);
        if each(0x7de, 0x7e2) {
            v[3] *= self.parent_alpha[0];
        }
        if each(0x7df, 0x7e3) {
            v[7] *= self.parent_alpha[1];
        }
        v[12] = self.fade(0x75b, 0x755);
        if each(0x7de, 0x7e2) || each(0x7df, 0x7e3) {
            v[12] *= self.parent_fade;
        }
        // The particle scale: the set's scale times the scale fade. Particles are drawn with
        // the unscaled matrix (`rt`) and this, so leaving the set's scale out drew every
        // particle of a scaled effect at size 1 while its emitter moved as scaled.
        let fade_scale = self.fade(0x75c, 0x756);
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

/// Where a particle is for the children emitted from it: in its emitter's space, or in the
/// world through its birth matrix when its emitter's particles keep it (`follows` false).
fn anchor_of(particle: &Live, follows: bool) -> Vec3 {
    if follows {
        particle.position
    } else {
        particle.birth_matrix.transform_point3(particle.position)
    }
}

/// A matrix's turn without its scale: each axis cut to length 1 (an emitter's matrix without
/// scale, emitter+0x380).
fn unscaled(m: &Mat4) -> glam::Mat3 {
    glam::Mat3::from_cols(
        m.x_axis.truncate().normalize_or_zero(),
        m.y_axis.truncate().normalize_or_zero(),
        m.z_axis.truncate().normalize_or_zero(),
    )
}

/// Key `index` of the 8-key table at res+`table`: its value and time.
fn key(raw: &[u8], table: usize, index: i32) -> (Vec3, f32) {
    let at = table + 16 * index.clamp(0, 7) as usize;
    (
        Vec3::new(f32_at(raw, at), f32_at(raw, at + 4), f32_at(raw, at + 8)),
        f32_at(raw, at + 12),
    )
}

/// An 8-key curve as the CPU reads it (Smash 0xe400).
struct Keys {
    table: usize,
    count: i32,
    /// Frames one loop of the curve takes (0: the curve runs once over the life).
    loop_rate: f32,
    /// 1 when each particle starts its loop at a random point (its first random value).
    random_loop: f32,
}

impl Keys {
    /// The curve's value at `age` for a particle with first random value `random` and life
    /// `life`: 1 with no keys, the first key with one; else the curve's place is age over life,
    /// or with a loop rate the age (plus the random start) wrapped to the loop over the rate;
    /// the first key before its time, the last from its time on, and between them a straight
    /// line between the two keys around it.
    fn at(&self, raw: &[u8], random: f32, age: f32, life: f32) -> Vec3 {
        let n = self.count.min(8);
        if n <= 0 {
            return Vec3::ONE;
        }
        if n == 1 {
            return key(raw, self.table, 0).0;
        }
        let t = if self.loop_rate > 0.0 {
            (random * self.random_loop * self.loop_rate + age) % self.loop_rate / self.loop_rate
        } else {
            age / life
        };
        let last = key(raw, self.table, n - 1);
        if last.1 <= t {
            return last.0;
        }
        let first = key(raw, self.table, 0);
        if t < first.1 {
            return first.0;
        }
        for i in 0..n - 1 {
            let (a, b) = (key(raw, self.table, i), key(raw, self.table, i + 1));
            if a.1 <= t && t < b.1 {
                return a.0 + (b.0 - a.0) * ((t - a.1) / (b.1 - a.1));
            }
        }
        Vec3::ZERO
    }
}

/// The wave res+0x9e4 puts on opacity and res+0x9e5 on scale (Smash 0xffe0, 0x10490): with
/// `axis` 0 the X settings, 1 the Y ones (res+0xe0 amplitude, 0xe8 period, 0xf0 random phase,
/// 0xf8 phase, each +4 for Y). Its place is (phase + age) / period plus the random phase times
/// the particle's first random value; the kind (res+0x9e7, high four bits) makes it a cosine
/// that dips to 1 - amplitude (0), a saw (1) or a square (2).
fn wave(raw: &[u8], random: f32, age: f32, axis: usize) -> f32 {
    let at = |base: usize| f32_at(raw, base + 4 * axis);
    let (amplitude, period, random_phase, phase) = (at(0xe0), at(0xe8), at(0xf0), at(0xf8));
    let place = (phase + age) / period + random_phase * random;
    let part = place - place.floor();
    match raw.get(0x9e7).copied().unwrap_or(0) >> 4 {
        0 => 1.0 - amplitude * ((std::f32::consts::TAU * place).cos() + 1.0) * 0.5,
        1 => (1.0 - amplitude * part).abs(),
        2 => (1.0 - amplitude * if part < 0.5 { 1.0 } else { 0.0 }).abs(),
        _ => 1.0,
    }
}

/// How far a rotation turning at speed 1 has turned after `age` frames under the per-frame
/// resistance `air`: the age itself at 1, else (1 - air^age) / (1 - air).
fn resisted(air: f32, age: f32) -> f32 {
    if air == 1.0 {
        age
    } else {
        (1.0 - air.powf(age)) / (1.0 - air)
    }
}

/// A world direction in the space of an emitter whose matrix is `m` (scale included): through
/// the matrix's inverse, as InitializeParticle turns a world-oriented velocity (Smash 0xa2d0).
fn world_to_emitter(m: &Mat4, world: Vec3) -> Vec3 {
    if m.determinant().abs() > 1e-12 {
        m.inverse().transform_vector3(world)
    } else {
        world
    }
}

/// World gravity in the space of an emitter whose matrix is `m`: its component along each of
/// the matrix's axes, scale taken out (CalculateParticleBehavior, Smash 0xf9f0).
fn world_gravity_in(m: &Mat4, gravity: Vec3) -> Vec3 {
    let along = |axis: glam::Vec4| axis.truncate().normalize_or_zero().dot(gravity);
    Vec3::new(along(m.x_axis), along(m.y_axis), along(m.z_axis))
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

/// Child inheritance against the game's own effects. Point VFX_GAME_EFFECTS at the game's
/// `effect` folder (skipped when unset).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ptcl, static_block};

    /// An emitter of a game effect with its children: parsed, and raw after load-time fixes.
    struct Part {
        data: EmitterData,
        fixed: Vec<u8>,
        children: Vec<Part>,
    }

    fn part(parsed: &effect_library::structs::Emitter, raw: &ptcl::Emitter) -> Part {
        Part {
            data: parsed.data.clone(),
            fixed: static_block::fixed_emitter_data(raw, &|_| None, None),
            children: parsed
                .children
                .iter()
                .zip(&raw.children)
                .map(|(p, r)| part(p, r))
                .collect(),
        }
    }

    /// Emitter `emitter` of the set named `set` in `file` (under the effect folder).
    fn game_part(file: &str, set: &str, emitter: usize) -> Option<Part> {
        let root = std::path::PathBuf::from(std::env::var_os("VFX_GAME_EFFECTS")?);
        let data = std::fs::read(root.join(file)).unwrap();
        let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
        let index = eff
            .entry_names
            .iter()
            .zip(&eff.entries)
            .find(|(n, _)| n.as_str() == set)
            .map(|(_, e)| e.emitter_set_id as usize - 1)
            .unwrap();
        let p = eff.ptcl_file.as_ref().unwrap();
        let raw = &ptcl::emitter_sets(&p.base_bytes)[index].emitters[emitter];
        Some(part(
            &p.emitter_list.emitter_sets[index].emitters[emitter],
            raw,
        ))
    }

    fn sim(part: &Part) -> EmitterSim<'_> {
        let mut s = EmitterSim::new(&part.data, &part.fixed);
        for child in &part.children {
            s.add_child(&child.data, &child.fixed);
        }
        s
    }

    fn put_f32(raw: &mut [u8], at: usize, value: f32) {
        raw[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn skipped() -> bool {
        let unset = std::env::var_os("VFX_GAME_EFFECTS").is_none();
        if unset {
            eprintln!("VFX_GAME_EFFECTS not set; skipped");
        }
        unset
    }

    /// Steps until the first child has particles, then a few more.
    fn run_until_child(s: &mut EmitterSim, child: usize, extra: usize) {
        for _ in 0..200 {
            s.step();
            if s.children[child].count() > 0 {
                break;
            }
        }
        assert!(s.children[child].count() > 0, "the child never emitted");
        for _ in 0..extra {
            s.step();
        }
    }

    /// The tester's case: Bayonetta's Witch Time clock hand (clock1, scale 5.4 x 30.3 x 5.4)
    /// leaves a trail (trace1, scale 1) that takes its scale (res+0x7d9 is set in the game's
    /// own file). The trail drew tiny because nothing took the hand's scale.
    #[test]
    fn the_clock_trail_takes_the_hands_scale() {
        if skipped() {
            return;
        }
        let file = "fighter/bayonetta/ef_bayonetta.eff";
        let part = game_part(file, "BAYONETTA_WITCHTIME_CLOCK", 0).unwrap();
        assert_eq!(part.children[0].fixed[0x7d9], 1);
        let mut s = sim(&part);
        run_until_child(&mut s, 0, 3);
        let hand = s.particles[0].scale;
        assert!((hand[1] - 30.28696).abs() < 1e-3, "{hand:?}");
        for p in &s.children[0].particles {
            assert_eq!(p.scale[..3], hand[..3]);
        }
        // Paired: with the flag cleared the trail keeps its own scale of 1.
        let mut part = part;
        part.children[0].fixed[0x7d9] = 0;
        let mut s = sim(&part);
        run_until_child(&mut s, 0, 3);
        for p in &s.children[0].particles {
            assert_eq!(p.scale[..3], [1.0; 3]);
        }
    }

    /// The scale rate (res+0x7ec) multiplies what is taken.
    #[test]
    fn the_scale_rate_multiplies_the_taken_scale() {
        if skipped() {
            return;
        }
        let file = "fighter/bayonetta/ef_bayonetta.eff";
        let mut part = game_part(file, "BAYONETTA_WITCHTIME_CLOCK", 0).unwrap();
        put_f32(&mut part.children[0].fixed, 0x7ec, 0.5);
        let mut s = sim(&part);
        run_until_child(&mut s, 0, 1);
        let hand = s.particles[0].scale;
        let trail = s.children[0].particles[0].scale;
        for axis in 0..3 {
            assert!(
                (trail[axis] - hand[axis] * 0.5).abs() < 1e-4,
                "{trail:?} {hand:?}"
            );
        }
    }

    /// Roy's Blazer eruption: each flare2 spark turns by a random share of half a turn about Z
    /// (res+0x718 = pi, no initial angle, no spin), and the smoke3 puffs it leaves take that
    /// angle (res+0x7da), where they had their own initial angle of 0.
    #[test]
    fn the_eruption_smoke_takes_its_sparks_angle() {
        if skipped() {
            return;
        }
        let file = "fighter/roy/ef_roy.eff";
        let part = game_part(file, "ROY_ERUPTION", 5).unwrap();
        let child = &part.children[0];
        assert_eq!(child.fixed[0x7da], 1);
        assert_eq!(f32_at(&child.fixed, 0x708), 0.0);
        let mut s = sim(&part);
        run_until_child(&mut s, 0, 2);
        let angles: Vec<f32> = s
            .particles
            .iter()
            .map(|p| std::f32::consts::PI * p.random[2])
            .collect();
        let children = &s.children[0].particles;
        assert!(!children.is_empty());
        for p in children {
            assert!(p.rotate[2] != 0.0);
            assert!(
                angles.iter().any(|a| (a - p.rotate[2]).abs() < 1e-5),
                "{} not one of {angles:?}",
                p.rotate[2]
            );
        }
        // Paired: with the flag cleared the puffs keep their own angle.
        let mut part = part;
        part.children[0].fixed[0x7da] = 0;
        let mut s = sim(&part);
        run_until_child(&mut s, 0, 2);
        assert!(s.children[0].particles.iter().all(|p| p.rotate[2] == 0.0));
    }

    /// Sephiroth's charged fire sparks (sparks4) hand their colours to Emitter1. The game
    /// multiplies what it hands by the spark's colour scale (res+0x3b0, 2 here), and passes
    /// colour 1 the emitter's colour 0, not its colour 1 (Smash 0x9ab0 hands both
    /// CalculateParticleColor0/1VecFromTime emitter+0x624 and +0x654).
    #[test]
    fn colours_taken_carry_the_colour_scale_and_the_emitters_colour_0() {
        if skipped() {
            return;
        }
        let file = "fighter/edge/ef_edge.eff";
        let base = game_part(file, "EDGE_FIRE2_HOLD", 11).unwrap();
        assert_eq!(base.children[0].fixed[0x7dc..0x7e0], [1, 1, 1, 1]);
        let colours = |edit: &dyn Fn(&mut Vec<u8>)| {
            let mut part = game_part(file, "EDGE_FIRE2_HOLD", 11).unwrap();
            edit(&mut part.fixed);
            let mut s = sim(&part);
            run_until_child(&mut s, 0, 2);
            s.children[0].particles[0].colours
        };
        let plain = colours(&|_| {});
        let scaled = colours(&|raw| {
            let scale = f32_at(raw, 0x3b0);
            put_f32(raw, 0x3b0, scale * 2.0);
        });
        let red0 = colours(&|raw| put_f32(raw, 0x7ac, 0.5));
        let red1 = colours(&|raw| put_f32(raw, 0x7bc, 0.5));
        for k in 0..2 {
            for c in 0..3 {
                assert!((scaled[k][c] - plain[k][c] * 2.0).abs() < 1e-4);
            }
            assert_eq!(scaled[k][3], plain[k][3]);
            assert!((red0[k][0] - plain[k][0] * 0.5).abs() < 1e-4);
            assert_eq!(red1[k][0], plain[k][0]);
        }
        assert!(plain[0][0] > 0.0 && plain[1][0] > 0.0);
    }

    /// Marshadow's Shadow Steel winds (wind1) take their shadow's opacities every frame
    /// (res+0x7e2, 0x7e3): the wind's opacities and fade multiply by its parent emitter's.
    #[test]
    fn opacity_taken_every_frame_follows_the_parent_emitter() {
        if skipped() {
            return;
        }
        let file = "pokemon/marshadow/ef_marshadow.eff";
        let mut part = game_part(file, "MARSHADOW_SHADOW_STEEL", 2).unwrap();
        assert_eq!(part.children[0].fixed[0x7de..0x7e4], [1, 1, 1, 1, 1, 1]);
        put_f32(&mut part.fixed, 0x7b8, 0.5);
        put_f32(&mut part.fixed, 0x7c8, 0.25);
        let row = |s: &EmitterSim| {
            let block = s.dynamic_block(&Mat4::IDENTITY, &Mat4::IDENTITY, [1.0; 4]);
            let f = |at: usize| f32::from_le_bytes(block[at..at + 4].try_into().unwrap());
            [f(0x0c), f(0x1c), f(0x30)]
        };
        let mut s = sim(&part);
        run_until_child(&mut s, 0, 1);
        let own = [
            f32_at(&part.children[0].fixed, 0x7b8),
            f32_at(&part.children[0].fixed, 0x7c8),
        ];
        let [a0, a1, fade] = row(&s.children[0]);
        assert_eq!([a0, a1, fade], [own[0] * 0.5, own[1] * 0.25, 1.0]);
        // The shadow fades out over res+0x768 = 4 frames once killed; so does the wind.
        assert_eq!(part.fixed[0x755], 1);
        s.kill();
        s.step();
        s.step();
        let parent_fade = row(&s)[2];
        assert!(parent_fade > 0.0 && parent_fade < 1.0);
        assert_eq!(
            row(&s.children[0])[2],
            s.children[0].fade(0x75b, 0x755) * parent_fade
        );
        // Paired: without the every-frame flags the wind keeps its own.
        part.children[0].fixed[0x7e2] = 0;
        part.children[0].fixed[0x7e3] = 0;
        let mut s = sim(&part);
        run_until_child(&mut s, 0, 1);
        assert_eq!(row(&s.children[0]), [own[0], own[1], 1.0]);
    }

    /// What the running game (13.0.4) handed child particles, read at the end of Smash 0x9ab0
    /// beside the parent particle's streams: Sephiroth's full-charge Gigaflare (Emitter1 from
    /// sparks4: scale and colours), his charge sparks (one emitter per particle: colours) and
    /// Roy's full Flare Blade (rotation). All 1,820 records of that session agree; these are
    /// six of them, at ages where one step either way gives something else.
    #[test]
    fn what_a_child_takes_matches_the_running_game() {
        if skipped() {
            return;
        }
        struct Record {
            /// Parent age: emitter time less birth, or emitter+0x564 for one per particle.
            age: f32,
            step: f32,
            life: f32,
            scale: [f32; 3],
            random: [f32; 4],
            taken_scale: Option<[f32; 3]>,
            taken_turn: Option<f32>,
            colours: Option<[[f32; 4]; 2]>,
        }
        let check = |file: &str, set: &str, emitter: usize, child: usize, record: Record| {
            let part = game_part(file, set, emitter).unwrap();
            let mut parent = EmitterSim::new(&part.data, &part.fixed);
            parent.frame_rate = record.step;
            let of = &part.children[child];
            let child = EmitterSim::new(&of.data, &of.fixed);
            let [x, y, z] = record.scale;
            let particle = Live {
                position: Vec3::ZERO,
                velocity: Vec3::ZERO,
                diff: Vec3::ZERO,
                life: record.life,
                birth: 0.0,
                scale: [x, y, z, 1.0],
                random: record.random,
                rotate: [0.0; 4],
                birth_matrix: Mat4::IDENTITY,
                colours: [[1.0; 4]; 2],
                child_emission: Vec::new(),
                id: 0,
                anchor: None,
            };
            let got = parent.inherited(&child, &particle, record.age, true);
            let near = |a: f32, b: f32| (a - b).abs() < 1e-4;
            assert_eq!(got.scale.is_some(), record.taken_scale.is_some());
            if let (Some(got), Some(want)) = (got.scale, record.taken_scale) {
                assert!((0..3).all(|k| near(got[k], want[k])), "{got} {want:?}");
            }
            assert_eq!(got.rotate.is_some(), record.taken_turn.is_some());
            if let (Some(got), Some(want)) = (got.rotate, record.taken_turn) {
                assert!(near(got.z, want) && got.x == 0.0 && got.y == 0.0, "{got}");
            }
            if let Some(want) = record.colours {
                for k in 0..2 {
                    for c in 0..4 {
                        let got = got.colours[k][c].unwrap();
                        assert!(near(got, want[k][c]), "colour {k} {c}: {got} {want:?}");
                    }
                }
            }
        };
        let gigaflare = |age, life, scale, random, taken_scale, colours| {
            let record = Record {
                age,
                step: 1.0,
                life,
                scale,
                random,
                taken_scale: Some(taken_scale),
                taken_turn: None,
                colours: Some(colours),
            };
            check(
                "fighter/edge/ef_edge.eff",
                "EDGE_FIRE3_BURST2",
                9,
                0,
                record,
            );
        };
        gigaflare(
            2.0,
            17.400002,
            [6.3827963, 12.7890625, 11.152418],
            [0.5392321, 0.7159593, 0.9102671, 0.74755865],
            [6.1923285, 12.407426, 10.819621],
            [[2.0, 2.0, 2.0, 1.0], [2.0, 0.0, 0.08673049, 1.0]],
        );
        gigaflare(
            9.0,
            24.0,
            [7.236662, 12.477024, 14.6442995],
            [0.9251717, 0.78743845, 0.11956772, 0.84290457],
            [5.984163, 10.317539, 12.109709],
            [
                [2.0, 1.9927348, 1.9982266, 1.0],
                [2.0, 0.0, 0.049493782, 1.0],
            ],
        );
        gigaflare(
            21.0,
            27.0,
            [6.6923494, 12.730158, 13.341118],
            [0.34320498, 0.7480073, 0.111577585, 0.2764631],
            [3.069599, 5.838978, 6.1192083],
            [[2.0, 1.1047585, 1.781477, 1.0], [2.0, 0.0, 0.0, 1.0]],
        );
        let charge = |age, life, size: f32, random, colours| {
            let record = Record {
                age,
                step: 1.0,
                life,
                scale: [size; 3],
                random,
                taken_scale: None,
                taken_turn: None,
                colours: Some(colours),
            };
            check("fighter/edge/ef_edge.eff", "EDGE_FIRE1_HOLD", 7, 0, record);
        };
        charge(
            3.0,
            21.84,
            3.6898875,
            [0.36322674, 0.4026642, 0.9192214, 0.29741958],
            [
                [5.0, 1.1765441, 1.1315339, 2.0],
                [5.0, 0.19841285, 0.19841285, 0.9299687],
            ],
        );
        charge(
            17.0,
            20.4,
            3.2105603,
            [0.4671461, 0.91357005, 0.34266183, 0.40876815],
            [
                [5.0, 4.4665318, 3.777467, 2.0],
                [5.0, 0.19841285, 0.19841285, 0.1190476],
            ],
        );
        let record = Record {
            age: 0.0,
            step: 0.7,
            life: 45.6,
            scale: [0.9337501; 3],
            random: [0.37254477, 0.92193, 0.0077153067, 0.57483727],
            taken_scale: None,
            taken_turn: Some(0.048476703),
            colours: None,
        };
        check("fighter/roy/ef_roy.eff", "ROY_ERUPTION_MAX", 4, 0, record);
    }

    /// Yoshi's Final Smash stars (star1) each carry their own starline1 emitter (res+0x7e4):
    /// the game places that emitter at its star every frame, so the line's particles, which
    /// follow their emitter, move with the star.
    #[test]
    fn a_child_per_particle_rides_with_its_parent_particle() {
        if skipped() {
            return;
        }
        let file = "fighter/yoshi/ef_yoshi.eff";
        let part = game_part(file, "YOSHI_FINAL_HIT", 8).unwrap();
        let child = &part.children[0];
        assert_eq!((child.fixed[0x7e4], child.fixed[0x753]), (1, 0));
        let mut s = sim(&part);
        run_until_child(&mut s, 0, 2);
        let own = s.children[0].local;
        let before: HashMap<u64, Vec3> = s.children[0]
            .particles
            .iter()
            .map(|p| (p.id, p.position))
            .collect();
        let stars: HashMap<u64, Vec3> = s.particles.iter().map(|p| (p.id, p.position)).collect();
        s.step();
        let mut checked = 0;
        for p in &s.children[0].particles {
            let (Some(was), Some((star, _))) = (before.get(&p.id), p.anchor) else {
                continue;
            };
            let Some(now) = s.particles.iter().find(|q| q.id == star) else {
                continue;
            };
            let moved = own.inverse().transform_vector3(now.position - stars[&star]);
            assert!(moved.length() > 1e-3, "the star stood still");
            let expected = *was + p.diff + moved;
            assert!(
                (p.position - expected).length() < 1e-3,
                "{} {expected}",
                p.position
            );
            checked += 1;
        }
        assert!(checked > 0);
        // Paired: one shared starline emitter leaves its particles where they were made.
        let mut part = part;
        part.children[0].fixed[0x7e4] = 0;
        let mut s = sim(&part);
        run_until_child(&mut s, 0, 2);
        let before: HashMap<u64, Vec3> = s.children[0]
            .particles
            .iter()
            .map(|p| (p.id, p.position))
            .collect();
        s.step();
        for p in &s.children[0].particles {
            if let Some(was) = before.get(&p.id) {
                assert!((p.position - (*was + p.diff)).length() < 1e-5);
            }
        }
    }
}
