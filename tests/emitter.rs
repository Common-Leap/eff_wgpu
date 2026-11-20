//! Emitter state against a real effect. Point VFX_GAME_EFFECTS at the game's `effect` folder
//! (skipped when unset).
use eff_wgpu::{ptcl, sim, static_block};

fn with_emitter<T>(emitter: usize, f: impl FnOnce(&mut sim::EmitterSim) -> T) -> T {
    let root = std::path::PathBuf::from(std::env::var_os("VFX_GAME_EFFECTS").unwrap());
    let data = std::fs::read(root.join("fighter/mario/ef_mario.eff")).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let set = eff
        .entry_names
        .iter()
        .zip(&eff.entries)
        .find(|(n, _)| n.as_str() == "MARIO_FB_BULLET_R")
        .map(|(_, e)| e.emitter_set_id as usize - 1)
        .unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let raw = &ptcl::emitter_sets(&p.base_bytes)[set].emitters[emitter];
    let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
    let parsed = &p.emitter_list.emitter_sets[set].emitters[emitter].data;
    let mut s = sim::EmitterSim::new(parsed, &fixed);
    f(&mut s)
}

fn dynamic_scale_and_alpha(steps: u32, emitter: usize) -> (f32, f32) {
    with_emitter(emitter, |s| {
        for _ in 0..steps {
            s.step();
        }
        let block = s.dynamic_block(&glam::Mat4::IDENTITY, &glam::Mat4::IDENTITY, [1.0; 4]);
        let f = |at: usize| f32::from_le_bytes(block[at..at + 4].try_into().unwrap());
        (f(0x34), f(0x30))
    })
}

#[test]
fn mario_fireball_disc_grows_in_over_six_frames() {
    if std::env::var_os("VFX_GAME_EFFECTS").is_none() {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    }
    // Emitter 1 ("ciecle3") sets res+0x75c with res+0x76c = 6: its scale fades in, alpha doesn't.
    assert_eq!(dynamic_scale_and_alpha(3, 1), (0.5, 1.0));
    assert_eq!(dynamic_scale_and_alpha(6, 1), (1.0, 1.0));
    assert_eq!(dynamic_scale_and_alpha(20, 1), (1.0, 1.0));
    // Emitter 0 ("fire1") sets neither flag and is full size at once.
    assert_eq!(dynamic_scale_and_alpha(1, 0), (1.0, 1.0));
}

#[test]
fn mario_fireball_disc_is_turned_by_its_own_rotation() {
    if std::env::var_os("VFX_GAME_EFFECTS").is_none() {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    }
    // "ciecle3" stores a -90 degree turn about Y (res+0x78c) with no spread: its X axis ends
    // up along +Z in the set, and nothing moves or scales it.
    let m = with_emitter(1, |s| s.emitter_srt(&glam::Mat4::IDENTITY));
    let x = m.transform_vector3(glam::Vec3::X);
    assert!((x - glam::Vec3::Z).length() < 1e-5, "{x}");
    assert!(m.transform_point3(glam::Vec3::ZERO).length() < 1e-6);
    // "fire1" has no turn of its own.
    let m = with_emitter(0, |s| s.emitter_srt(&glam::Mat4::IDENTITY));
    assert!((m.transform_vector3(glam::Vec3::X) - glam::Vec3::X).length() < 1e-6);
}

#[test]
fn dynamic_block_row_two_matches_a_captured_game_draw() {
    if std::env::var_os("VFX_GAME_EFFECTS").is_none() {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    }
    // The game's block for the fireball's ciecle1 disc, captured at emitter time 8 (Ryujinx +
    // RenderDoc): row 2 is (time, particle count, 0, 1).
    let row = with_emitter(2, |s| {
        for _ in 0..9 {
            s.step();
        }
        let block = s.dynamic_block(&glam::Mat4::IDENTITY, &glam::Mat4::IDENTITY, [1.0; 4]);
        let f = |at: usize| f32::from_le_bytes(block[at..at + 4].try_into().unwrap());
        [f(0x20), f(0x24), f(0x28), f(0x2c)]
    });
    assert_eq!(row, [8.0, 1.0, 0.0, 1.0]);
}

#[test]
fn distance_emission_follows_the_path_the_emitter_moved() {
    if std::env::var_os("VFX_GAME_EFFECTS").is_none() {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    }
    // fire1 (emitter 0): distance emission, one particle every 2.0 units, from frame 2. Its
    // duration (4) does not stop it: in game the fireball trail keeps adding flames.
    let counts = |speed: f32| {
        with_emitter(0, |s| {
            for frame in 0..8 {
                s.step_at(&glam::Mat4::from_translation(glam::Vec3::new(
                    speed * frame as f32,
                    0.0,
                    0.0,
                )));
            }
            s.count()
        })
    };
    // Standing still emits nothing, as in the game.
    assert_eq!(counts(0.0), 0);
    // 1.4 a frame over frames 2-7: 8.4 units, four particles.
    assert_eq!(counts(1.4), 4);
    // 3.0 a frame: 18 units, nine particles.
    assert_eq!(counts(3.0), 9);
}

#[test]
fn bank_one_literals_match_a_captured_game_draw() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let data =
        std::fs::read(std::path::Path::new(&root).join("fighter/mario/ef_mario.eff")).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let bin = eff
        .ptcl_file
        .unwrap()
        .shader_info
        .unwrap()
        .binary_data
        .unwrap();
    let file = eff_wgpu::bnsh::ShaderFile::read(&bin).unwrap();
    // Program 6 is the fireball disc ciecle3; the game bound these 32 bytes as its vertex bank 1.
    let captured: Vec<u8> = [
        1.0f32.to_bits(),
        1 << 19,
        1 << 20,
        1 << 21,
        1 << 22,
        1 << 23,
        1 << 24,
        (-0.5f32).to_bits(),
    ]
    .iter()
    .flat_map(|w| w.to_le_bytes())
    .collect();
    assert_eq!(
        file.programs[6].stages[0].as_ref().unwrap().literals(),
        captured.as_slice()
    );
}

/// sys_atk_smoke's last emitter, impact3: one-time, rate 1.5 for 3 frames from frame 2, thrown
/// from a 163 degree circle sweep centred on pi. Read live from the game (13.0.4, Mario's
/// forward smash): 1, then 3, then 4 particles, every one moving backward (-z).
#[test]
fn ground_dust_spray_emits_over_its_duration_and_backward() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let data =
        std::fs::read(std::path::Path::new(&root).join("system/common/ef_common.eff")).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let set = eff
        .entry_names
        .iter()
        .zip(&eff.entries)
        .find(|(n, _)| n.eq_ignore_ascii_case("sys_atk_smoke"))
        .map(|(_, e)| e.emitter_set_id as usize - 1)
        .unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let index = p.emitter_list.emitter_sets[set]
        .emitters
        .iter()
        .position(|e| e.data.name.as_deref() == Some("impact3"))
        .unwrap();
    let raw = &ptcl::emitter_sets(&p.base_bytes)[set].emitters[index];
    let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
    let parsed = &p.emitter_list.emitter_sets[set].emitters[index].data;
    let mut s = sim::EmitterSim::new(parsed, &fixed);
    let mut counts = Vec::new();
    for _ in 0..6 {
        s.step();
        counts.push(s.count());
    }
    // The game's draw at emitter time T follows T + 1 updates.
    assert_eq!(counts, [0, 0, 1, 3, 4, 4]);
    let streams = s.streams();
    for v in &streams["sysLocalVecAttr"] {
        assert!(v[2] < 0.0, "particle thrown forward: {v:?}");
    }
}

/// The kill flags are the parsed emitter info's is_fade_emit, is_fade_alpha_fade and
/// is_scale_fade at res+0x754..0x756, and a killed emitter stops emitting when the first is set.
#[test]
fn a_killed_emitter_stops_emitting_and_fades_as_its_flags_say() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let data =
        std::fs::read(std::path::Path::new(&root).join("system/common/ef_common.eff")).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let raws = ptcl::emitter_sets(&p.base_bytes);
    let (mut checked, mut fading) = (0, 0);
    for (set, parsed) in p.emitter_list.emitter_sets.iter().enumerate() {
        for (index, emitter) in parsed.emitters.iter().enumerate() {
            let raw = &raws[set].emitters[index];
            let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
            let info = &emitter.data.emitter_info;
            assert_eq!(
                (fixed[0x754] != 0, fixed[0x755] != 0, fixed[0x756] != 0),
                (
                    info.is_fade_emit != 0,
                    info.is_fade_alpha_fade != 0,
                    info.is_scale_fade != 0
                ),
                "set {set} emitter {index}"
            );
            checked += 1;
            // An emitter that keeps emitting, stops when killed, and fades its alpha out.
            if info.is_fade_emit != 0
                && info.is_fade_alpha_fade != 0
                && !emitter.data.emission.is_one_time
                && emitter.data.emission.rate >= 1.0
                && emitter.data.emission.start == 0
                && i32::from_le_bytes(fixed[0x768..0x76c].try_into().unwrap()) >= 2
                && fading == 0
            {
                fading += 1;
                let mut s = sim::EmitterSim::new(&emitter.data, &fixed);
                for _ in 0..5 {
                    s.step();
                }
                assert!(s.count() > 0);
                s.kill();
                s.step();
                let alpha = |s: &sim::EmitterSim| {
                    let block =
                        s.dynamic_block(&glam::Mat4::IDENTITY, &glam::Mat4::IDENTITY, [1.0; 4]);
                    f32::from_le_bytes(block[0x30..0x34].try_into().unwrap())
                };
                assert!(alpha(&s) < 1.0, "fades after a kill");
                for _ in 0..200 {
                    s.step();
                }
                assert!(s.finished(), "gone once the fade runs out");
                assert_eq!(s.count(), 0);
            }
        }
    }
    assert!(checked > 1000, "{checked}");
    assert_eq!(fading, 1, "no fading emitter found to test");
}

#[test]
fn the_particle_scale_carries_the_set_scale() {
    if std::env::var_os("VFX_GAME_EFFECTS").is_none() {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    }
    // Live probe: Dark Samus's up tilt spawns sys_attack_arc_b at scale 1.65 (1.584 at her
    // 0.96) and the game's dynamic block holds 1.584 at 0x34..0x40, beside the scale fade.
    let particle_scale = |steps: u32, emitter: usize, scale: f32| {
        with_emitter(emitter, |s| {
            let set = glam::Mat4::from_scale(glam::Vec3::splat(scale));
            for _ in 0..steps {
                s.step_at(&set);
            }
            let block = s.dynamic_block(&set, &glam::Mat4::IDENTITY, [1.0; 4]);
            let f = |at: usize| f32::from_le_bytes(block[at..at + 4].try_into().unwrap());
            [f(0x34), f(0x38), f(0x3c)]
        })
    };
    // fire1 has no scale fade: the set's scale as it is.
    assert_eq!(particle_scale(6, 0, 1.584), [1.584; 3]);
    // ciecle3 fades its scale in over six frames: half way, half the set's scale.
    assert_eq!(particle_scale(3, 1, 2.0), [1.0; 3]);
    // Unscaled, as before.
    assert_eq!(particle_scale(6, 0, 1.0), [1.0; 3]);
}

/// Particle counts per step for one emitter of the shared file's sys_attack_arc_b.
fn arc_b_counts(emitter: &str, steps: usize) -> Vec<usize> {
    let root = std::path::PathBuf::from(std::env::var_os("VFX_GAME_EFFECTS").unwrap());
    let data = std::fs::read(root.join("system/common/ef_common.eff")).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let set = eff
        .entry_names
        .iter()
        .zip(&eff.entries)
        .find(|(n, _)| n.as_str() == "SYS_ATTACK_ARC_B")
        .map(|(_, e)| e.emitter_set_id as usize - 1)
        .unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let index = p.emitter_list.emitter_sets[set]
        .emitters
        .iter()
        .position(|e| e.data.name.as_deref() == Some(emitter))
        .unwrap();
    let raw = &ptcl::emitter_sets(&p.base_bytes)[set].emitters[index];
    let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
    let parsed = &p.emitter_list.emitter_sets[set].emitters[index].data;
    let mut s = sim::EmitterSim::new(parsed, &fixed);
    (0..steps)
        .map(|_| {
            s.step();
            s.count()
        })
        .collect()
}

#[test]
fn a_particle_draws_on_the_frame_it_is_born_and_for_its_whole_life() {
    if std::env::var_os("VFX_GAME_EFFECTS").is_none() {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    }
    // Live probe, Dark Samus's up tilt: arccylinder1 (life 1) draws at emitter time 0 only,
    // arccylinder2 (life 3, from time 1) at times 1, 2 and 3. The draw after step n reads
    // time n - 1.
    assert_eq!(arc_b_counts("arccylinder1", 3), vec![1, 0, 0]);
    assert_eq!(arc_b_counts("arccylinder2", 5), vec![0, 1, 1, 1, 0]);
}

/// Sephiroth's blue charge (edge_fire2_hold): an emitter's initial rotation comes from its own
/// rotation fields. It was read from the second sampler's texture scale (1, 1), which turned
/// every mesh particle 57 degrees about X and Y, so the charge ball was seen side on.
#[test]
fn particles_start_at_their_own_initial_rotation() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let data = std::fs::read(std::path::Path::new(&root).join("fighter/edge/ef_edge.eff")).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let set = p
        .emitter_list
        .emitter_sets
        .iter()
        .position(|s| s.name == "P_EdgeFire2Hold")
        .unwrap();
    let rotation = |name: &str| {
        let index = p.emitter_list.emitter_sets[set]
            .emitters
            .iter()
            .position(|e| e.data.name.as_deref() == Some(name))
            .unwrap();
        let raw = &ptcl::emitter_sets(&p.base_bytes)[set].emitters[index];
        let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
        let mut s = sim::EmitterSim::new(
            &p.emitter_list.emitter_sets[set].emitters[index].data,
            &fixed,
        );
        for _ in 0..30 {
            s.step();
        }
        s.streams()["sysInitRotateAttr"].clone()
    };
    // No initial angle and no random range: no rotation.
    for particle in rotation("sphere1") {
        assert_eq!(particle, [0.0; 4]);
    }
    // Random ranges of 50 degrees about X and a full turn about Y and Z: within them, and not
    // all alike.
    let ring = rotation("lightning_ring1_bk");
    assert!(!ring.is_empty());
    for [x, y, z, _] in &ring {
        assert!((0.0..=0.873).contains(x), "{x}");
        assert!(
            (0.0..=std::f32::consts::TAU).contains(y) && (0.0..=std::f32::consts::TAU).contains(z)
        );
    }
}

/// Runs `f` on a new sim of the emitter `name` in the set `set` of `file` under the game's
/// effect folder, or returns None when VFX_GAME_EFFECTS is unset.
fn with_named<T>(
    file: &str,
    set: &str,
    name: &str,
    f: impl FnOnce(&mut sim::EmitterSim) -> T,
) -> Option<T> {
    let root = std::env::var_os("VFX_GAME_EFFECTS")?;
    let data = std::fs::read(std::path::Path::new(&root).join(file)).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let set = p
        .emitter_list
        .emitter_sets
        .iter()
        .position(|s| s.name == set)
        .unwrap();
    let index = p.emitter_list.emitter_sets[set]
        .emitters
        .iter()
        .position(|e| e.data.name.as_deref() == Some(name))
        .unwrap();
    let raw = &ptcl::emitter_sets(&p.base_bytes)[set].emitters[index];
    let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
    let parsed = &p.emitter_list.emitter_sets[set].emitters[index].data;
    let mut s = sim::EmitterSim::new(parsed, &fixed);
    Some(f(&mut s))
}

/// Directions (radians, about Y in emitter space) the live particles move in.
fn headings(s: &sim::EmitterSim) -> Vec<f32> {
    s.streams()
        .get("sysLocalVecAttr")
        .map(|v| v.iter().map(|v| v[0].atan2(v[2])).collect())
        .unwrap_or_default()
}

/// The ice hit's crystals (P_CmnHitIce crystal1): an equally divided circle of 6 whose
/// divisions are not picked one at a time (res+0x874 = 0), so one emission of rate 1 throws a
/// crystal from every division, 60 degrees apart. The preview threw one.
#[test]
fn an_equally_divided_circle_emits_every_division() {
    let Some(mut angles) = with_named(
        "system/common/ef_common.eff",
        "P_CmnHitIce",
        "crystal1",
        |s| {
            s.step();
            headings(s)
        },
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert_eq!(angles.len(), 6);
    angles.sort_by(f32::total_cmp);
    for pair in angles.windows(2) {
        let step = (pair[1] - pair[0]).to_degrees();
        assert!((step - 60.0).abs() < 0.01, "{angles:?}");
    }
}

/// Sephiroth's charge rays (edge_fire2_hold light2): one ray an emission from a random one of 5
/// divisions (res+0x874 = 1) of a circle whose start is random each emission (res+0x839),
/// turned up to 30 degrees either way. Every ray came out of the same division, so the rays
/// stacked into one upright beam.
#[test]
fn equally_divided_rays_point_different_ways() {
    let Some(angles) = with_named(
        "fighter/edge/ef_edge.eff",
        "P_EdgeFire2Hold",
        "light2",
        |s| {
            let mut all = Vec::new();
            for _ in 0..80 {
                s.step();
                all.extend(headings(s));
            }
            all
        },
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert!(angles.len() > 20);
    let spread = angles
        .iter()
        .map(|a| (a - angles[0]).sin().abs())
        .fold(0.0f32, f32::max);
    assert!(spread > 0.5, "{angles:?}");
}

/// Sephiroth's charge sparks (edge_fire2_hold sparks4) have two child emitters that leave a
/// trail of dots behind every spark, coloured as the spark is: the game's red and gold streaks.
/// Child emitters were not run at all, so the sparks drew without their trails.
#[test]
fn child_emitters_trail_their_parent_particles() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let data = std::fs::read(std::path::Path::new(&root).join("fighter/edge/ef_edge.eff")).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let set = p
        .emitter_list
        .emitter_sets
        .iter()
        .position(|s| s.name == "P_EdgeFire2Hold")
        .unwrap();
    let index = p.emitter_list.emitter_sets[set]
        .emitters
        .iter()
        .position(|e| e.data.name.as_deref() == Some("sparks4"))
        .unwrap();
    let parsed = &p.emitter_list.emitter_sets[set].emitters[index];
    let raw = &ptcl::emitter_sets(&p.base_bytes)[set].emitters[index];
    let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
    let children: Vec<Vec<u8>> = raw
        .children
        .iter()
        .map(|c| static_block::fixed_emitter_data(c, &|_| None, None))
        .collect();
    assert_eq!(parsed.children.len(), 2);
    let mut s = sim::EmitterSim::new(&parsed.data, &fixed);
    for (child, fixed) in parsed.children.iter().zip(&children) {
        s.add_child(&child.data, fixed);
    }
    for _ in 0..30 {
        s.step();
    }
    assert!(s.count() > 0);
    // The sparks start 6 to 30 from the centre and fly in; every dot is where a spark was.
    for child in s.children() {
        assert!(child.count() > s.count(), "{} dots", child.count());
        let streams = child.streams();
        let radii: Vec<f32> = streams["sysLocalPosAttr"]
            .iter()
            .map(|pos| glam::Vec3::new(pos[0], pos[1], pos[2]).length())
            .collect();
        assert!(radii.iter().all(|r| *r <= 30.5), "{radii:?}");
        assert!(radii.iter().any(|r| *r > 3.0), "{radii:?}");
        // Coloured as the spark is at that age, not the plain white of a new particle.
        assert!(streams["sysColor0Attr"]
            .iter()
            .any(|c| c[..3] != [1.0, 1.0, 1.0]));
    }
}

/// The flames Mario's fireball leaves (MARIO_FB_BULLET_R fire1) take a tenth of the
/// fireball's move backward (res+0x990 = -0.1, InitializeParticle adds it to a new particle's
/// velocity), so the trail drifts behind the ball. The preview's flames ignored the emitter's
/// move and stood still where they were dropped.
#[test]
fn new_particles_take_part_of_the_emitters_move() {
    if std::env::var_os("VFX_GAME_EFFECTS").is_none() {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    }
    let velocities = with_emitter(0, |s| {
        for frame in 0..16 {
            s.step_at(&glam::Mat4::from_translation(glam::Vec3::new(
                3.0 * frame as f32,
                0.0,
                0.0,
            )));
        }
        s.streams()["sysLocalVecAttr"].clone()
    });
    // Moving 3 a frame along x adds -0.3 to every flame's x velocity. Its own velocity is
    // 0.3 in a random direction, which averages out over the trail.
    let n = velocities.len() as f32;
    assert!(n >= 10.0, "{velocities:?}");
    let mean = velocities.iter().fold(glam::Vec3::ZERO, |sum, v| {
        sum + glam::Vec3::new(v[0], v[1], v[2])
    }) / n;
    assert!(mean.x < -0.2 && mean.x > -0.4, "{mean}");
}

/// Sephiroth's charge sparks (edge_fire2_hold sparks4) drop gold dots through Emitter1, a child
/// that emits by distance (one dot every 1.5 of the spark's travel) and takes a fifth of the
/// spark's move (res+0x990 = 0.2), so the dots creep in after the spark and each trail draws
/// out into a long dotted line. Emitter2 emits by time, one a frame, and is handed no move.
/// Children emitted one dot a frame each, standing still, which drew short clumps.
#[test]
fn a_child_that_emits_by_distance_follows_its_parents_move() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let data = std::fs::read(std::path::Path::new(&root).join("fighter/edge/ef_edge.eff")).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let set = p
        .emitter_list
        .emitter_sets
        .iter()
        .position(|s| s.name == "P_EdgeFire2Hold")
        .unwrap();
    let index = p.emitter_list.emitter_sets[set]
        .emitters
        .iter()
        .position(|e| e.data.name.as_deref() == Some("sparks4"))
        .unwrap();
    let parsed = &p.emitter_list.emitter_sets[set].emitters[index];
    let raw = &ptcl::emitter_sets(&p.base_bytes)[set].emitters[index];
    let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
    let children: Vec<Vec<u8>> = raw
        .children
        .iter()
        .map(|c| static_block::fixed_emitter_data(c, &|_| None, None))
        .collect();
    let mut s = sim::EmitterSim::new(&parsed.data, &fixed);
    for (child, fixed) in parsed.children.iter().zip(&children) {
        s.add_child(&child.data, fixed);
    }
    for _ in 0..40 {
        s.step();
    }
    let by_distance = &s.children()[0];
    let by_time = &s.children()[1];
    // A spark flies in at up to 1 a frame: by distance it drops at most one dot every 1.5
    // frames, by time one every frame.
    assert!(by_distance.count() > 0);
    assert!(
        (by_distance.count() as f32) < by_time.count() as f32 * 0.8,
        "{} by distance, {} by time",
        by_distance.count(),
        by_time.count()
    );
    // The distance child's dots move inward, a fifth as fast as the sparks.
    let streams = by_distance.streams();
    for (pos, vel) in streams["sysLocalPosAttr"]
        .iter()
        .zip(&streams["sysLocalVecAttr"])
    {
        let pos = glam::Vec3::new(pos[0], pos[1], pos[2]);
        let vel = glam::Vec3::new(vel[0], vel[1], vel[2]);
        assert!(vel.length() > 0.05 && vel.length() < 0.21, "{vel}");
        assert!(pos.dot(vel) < 0.0, "{pos} {vel}");
    }
    // The time child is handed no move: its dots only fall by its tiny gravity.
    for vel in &by_time.streams()["sysLocalVecAttr"] {
        assert!(glam::Vec2::new(vel[0], vel[2]).length() < 1e-6, "{vel:?}");
    }
}

/// Sephiroth's charge sparks (edge_fire2_hold sparks4) carry a convergence field (FCOV: 0.11 of
/// the way to the centre a frame) on top of flying in at up to 1 a frame. The game draws long
/// gold lines that reach the ball within their life; run without the field they crept a third
/// of the way and drew short dashes.
#[test]
fn a_convergence_field_pulls_particles_to_its_target() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let data = std::fs::read(std::path::Path::new(&root).join("fighter/edge/ef_edge.eff")).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let set = p
        .emitter_list
        .emitter_sets
        .iter()
        .position(|s| s.name == "P_EdgeFire2Hold")
        .unwrap();
    let index = p.emitter_list.emitter_sets[set]
        .emitters
        .iter()
        .position(|e| e.data.name.as_deref() == Some("sparks4"))
        .unwrap();
    let parsed = &p.emitter_list.emitter_sets[set].emitters[index];
    let raw = &ptcl::emitter_sets(&p.base_bytes)[set].emitters[index];
    let fields = sim::Fields::of(&raw.attributes);
    let convergence = fields.convergence.expect("sparks4 has a convergence field");
    assert_eq!(convergence.target, glam::Vec3::ZERO);
    assert!((convergence.strength - 0.11).abs() < 1e-6);
    let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
    // The first sparks (frame 18) six frames on: how far from the centre each still is,
    // against where it started.
    let radii = |with_fields: bool| {
        let mut s = sim::EmitterSim::new(&parsed.data, &fixed);
        if with_fields {
            // The convergence alone: sparks4's curl noise bends the same paths.
            s.set_fields(sim::Fields {
                convergence: fields.convergence,
                ..Default::default()
            });
        }
        for _ in 0..25 {
            s.step();
        }
        s.streams()["sysLocalPosAttr"]
            .iter()
            .map(|p| glam::Vec3::new(p[0], p[1], p[2]).length())
            .collect::<Vec<_>>()
    };
    let (plain, pulled) = (radii(false), radii(true));
    assert_eq!(plain.len(), pulled.len());
    assert!(!plain.is_empty());
    // Same sparks, same seed: every one is nearer the centre with the field, and by a lot.
    let mean = |r: &[f32]| r.iter().sum::<f32>() / r.len() as f32;
    for (a, b) in plain.iter().zip(&pulled) {
        assert!(b < a, "{plain:?} {pulled:?}");
    }
    assert!(mean(&pulled) < mean(&plain) * 0.7, "{plain:?} {pulled:?}");
}

/// The raw attribute sections of the emitter `name` in `set` of `file`, beside a sim of it.
fn with_fields<T>(
    file: &str,
    set: &str,
    name: &str,
    f: impl FnOnce(&effect_library::emitter::EmitterData, &[u8], sim::Fields) -> T,
) -> Option<T> {
    let root = std::env::var_os("VFX_GAME_EFFECTS")?;
    let data = std::fs::read(std::path::Path::new(&root).join(file)).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let set = p
        .emitter_list
        .emitter_sets
        .iter()
        .position(|s| s.name == set)
        .unwrap();
    let index = p.emitter_list.emitter_sets[set]
        .emitters
        .iter()
        .position(|e| e.data.name.as_deref() == Some(name))
        .unwrap();
    let raw = &ptcl::emitter_sets(&p.base_bytes)[set].emitters[index];
    let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
    let parsed = &p.emitter_list.emitter_sets[set].emitters[index].data;
    Some(f(parsed, &fixed, sim::Fields::of(&raw.attributes)))
}

/// Positions and births of the live particles after `steps` frames with `fields`.
fn run_with(
    data: &effect_library::emitter::EmitterData,
    raw: &[u8],
    fields: sim::Fields,
    steps: usize,
) -> Vec<(glam::Vec3, f32)> {
    let mut s = sim::EmitterSim::new(data, raw);
    s.set_fields(fields);
    for _ in 0..steps {
        s.step();
    }
    let streams = s.streams();
    streams["sysLocalPosAttr"]
        .iter()
        .zip(&streams["sysLocalVecAttr"])
        .map(|(p, v)| (glam::Vec3::new(p[0], p[1], p[2]), v[3]))
        .collect()
}

/// Sephiroth's charge dots (edge_fire2_hold sparks2) carry a spin field (FSPN): 3 degrees a
/// frame about the emitter's Y axis, drawing in half a unit a frame. In game a dot 21 units out
/// was 13 units out six frames later, curving as it went; without the field it barely moved.
#[test]
fn a_spin_field_turns_particles_about_its_axis_and_draws_them_in() {
    let Some((fields, plain, spun)) = with_fields(
        "fighter/edge/ef_edge.eff",
        "P_EdgeFire2Hold",
        "sparks2",
        |data, raw, fields| {
            let only_spin = sim::Fields {
                spin: fields.spin,
                ..Default::default()
            };
            (
                fields,
                run_with(data, raw, sim::Fields::default(), 12),
                run_with(data, raw, only_spin, 12),
            )
        },
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let spin = fields.spin.expect("sparks2 has a spin field");
    assert_eq!(spin.axis, 1);
    assert!((spin.speed + 0.05236).abs() < 1e-4, "{}", spin.speed);
    assert!((spin.outward + 0.5).abs() < 1e-6);
    let noise = fields.curl_noise.expect("sparks2 has a curl noise field");
    assert_eq!(noise.power, glam::Vec3::splat(0.2));
    assert_eq!(plain.len(), spun.len());
    assert!(plain.len() > 5);
    // The same particles: each has turned by 3 degrees and come in by half a unit for every
    // frame it has lived, give or take its own small drift.
    let mut checked = 0;
    for ((a, birth), (b, _)) in plain.iter().zip(&spun) {
        let frames = 12.0 - birth;
        let (ra, rb) = (a.x.hypot(a.z), b.x.hypot(b.z));
        if frames < 3.0 || ra < 10.0 {
            continue;
        }
        let turned = (b.z.atan2(b.x) - a.z.atan2(a.x) + std::f32::consts::PI)
            .rem_euclid(std::f32::consts::TAU)
            - std::f32::consts::PI;
        assert!(
            (turned - spin.speed * frames).abs() < 0.03 * frames,
            "{frames} frames turned {turned}"
        );
        assert!(
            (ra - rb - 0.5 * frames).abs() < 0.25 * frames,
            "{frames} frames: {ra} -> {rb}"
        );
        checked += 1;
    }
    assert!(checked >= 3, "{checked}");
}

/// Sephiroth's charge sparks (sparks4) carry a curl noise field (FCLN) of power 1: each frame a
/// spark's velocity gains the noise at its position, so its path bends; it does not jump.
#[test]
fn a_curl_noise_field_bends_velocities_by_its_power() {
    let Some((plain, bent)) = with_fields(
        "fighter/edge/ef_edge.eff",
        "P_EdgeFire2Hold",
        "sparks4",
        |data, raw, fields| {
            let only_noise = sim::Fields {
                curl_noise: fields.curl_noise,
                ..Default::default()
            };
            let velocities = |fields| {
                let mut s = sim::EmitterSim::new(data, raw);
                s.set_fields(fields);
                for _ in 0..22 {
                    s.step();
                }
                s.streams()["sysLocalVecAttr"]
                    .iter()
                    .map(|v| (glam::Vec3::new(v[0], v[1], v[2]), v[3]))
                    .collect::<Vec<_>>()
            };
            (velocities(sim::Fields::default()), velocities(only_noise))
        },
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert_eq!(plain.len(), bent.len());
    assert!(!plain.is_empty());
    let mut changed = 0;
    for ((a, birth), (b, _)) in plain.iter().zip(&bent) {
        let frames = 22.0 - birth;
        let gained = (*b - *a).length();
        // At most the table's largest cell (1.3) a frame.
        assert!(gained <= 1.3 * frames + 1e-4, "{frames}: {gained}");
        if gained > 0.01 {
            changed += 1;
        }
    }
    assert!(changed * 2 >= plain.len(), "{changed} of {}", plain.len());
}

/// A particle moves by its velocity times its own motion scale (Smash 0xf9f0), the scale
/// stream's w, which res+0x8c0 spreads: Bomberman's finishing sparks (spread 0.5) fly at half
/// to one and a half times their speed. 971 game emitters spread it; the preview moved every
/// particle at the same speed.
#[test]
fn particles_move_by_their_own_motion_scale() {
    let Some(rows) = with_named(
        "assist/bomberman/ef_bomberman.eff",
        "P_BombermanBombFinish",
        "spark2_small",
        |s| {
            s.step();
            let streams = s.streams();
            streams["sysLocalDiffAttr"]
                .iter()
                .zip(&streams["sysLocalVecAttr"])
                .zip(&streams["sysScaleAttr"])
                .map(|((d, v), w)| (d[0], v[0], w[3]))
                .collect::<Vec<_>>()
        },
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    // Gravity pulls along y only and air resistance (0.91) scales the velocity after the
    // move, so along x the first frame's move is the velocity now / 0.91 times w.
    let air = 0.91;
    let mut spread = (f32::MAX, f32::MIN);
    let mut checked = 0;
    for (diff, velocity, w) in rows {
        if velocity.abs() < 1e-3 {
            continue;
        }
        assert!(
            (diff / (velocity / air) - w).abs() < 1e-3,
            "{diff} {velocity} {w}"
        );
        spread = (spread.0.min(w), spread.1.max(w));
        checked += 1;
    }
    assert!(checked >= 3, "{checked}");
    assert!(spread.1 - spread.0 > 0.2, "{spread:?}");
    assert!(spread.0 >= 0.5 && spread.1 <= 1.5, "{spread:?}");
}

/// Training mode's quarter-speed slow motion runs the game's effects a quarter of a frame a
/// step, smoothly (the reference video of Sephiroth's charge changes on every video frame). Four
/// steps at a quarter make the same particles, near where one step at full rate puts them,
/// but each step's move, which the shaders stretch sparks by, is a quarter as long.
#[test]
fn a_quarter_frame_rate_moves_the_same_particles_a_quarter_as_far_a_step() {
    type Run = (f32, Vec<[f32; 4]>, Vec<[f32; 4]>);
    let Some((full, quarter)) = with_fields(
        "fighter/edge/ef_edge.eff",
        "P_EdgeFire2Hold",
        "sparks2",
        |data, raw, _| {
            let run = |rate: f32, steps: usize| -> Run {
                let mut s = sim::EmitterSim::new(data, raw);
                s.set_frame_rate(rate);
                for _ in 0..steps {
                    s.step();
                }
                let streams = s.streams();
                (
                    s.time,
                    streams["sysLocalPosAttr"].clone(),
                    streams["sysLocalDiffAttr"].clone(),
                )
            };
            (run(1.0, 12), run(0.25, 48))
        },
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert_eq!(full.0, 12.0);
    assert_eq!(quarter.0, 12.0);
    assert_eq!(full.1.len(), quarter.1.len());
    assert!(full.1.len() > 5);
    let length = |v: &[f32; 4]| glam::Vec3::new(v[0], v[1], v[2]).length();
    let mut checked = 0;
    for ((p1, d1), (p4, d4)) in full
        .1
        .iter()
        .zip(&full.2)
        .zip(quarter.1.iter().zip(&quarter.2))
    {
        let gap = glam::Vec3::new(p1[0] - p4[0], p1[1] - p4[1], p1[2] - p4[2]).length();
        assert!(gap < 0.2, "{p1:?} {p4:?}");
        if length(d1) > 0.02 {
            assert!(
                (length(d4) / length(d1) - 0.25).abs() < 0.05,
                "{d1:?} {d4:?}"
            );
            checked += 1;
        }
    }
    assert!(checked >= 3, "{checked}");
}
