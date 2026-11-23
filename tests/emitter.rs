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
/// every mesh particle 57 degrees about X and Y, so the charge ball was seen side on. The
/// stream holds the initial angle alone (InitializeParticle draws nothing for it): the shaders
/// add the random range (static block 0x710) times the particle's random values. The preview
/// drew a random angle into the stream as well, so the shaders turned particles twice.
#[test]
fn particles_start_at_their_own_initial_rotation() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let rotation = |file: &str, set: &str, name: &str| {
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
        let mut s = sim::EmitterSim::new(
            &p.emitter_list.emitter_sets[set].emitters[index].data,
            &fixed,
        );
        s.spawn_from_seed(0x51, 0.0, 40);
        let range: Vec<f32> = (0..3)
            .map(|k| f32::from_le_bytes(fixed[0x710 + 4 * k..0x714 + 4 * k].try_into().unwrap()))
            .collect();
        let statics = static_block::static_block(&fixed);
        assert_eq!(&statics[0x710..0x71c], &fixed[0x710..0x71c]);
        (s.streams()["sysInitRotateAttr"].clone(), range)
    };
    let edge = "fighter/edge/ef_edge.eff";
    // No initial angle and no random range: no rotation.
    for particle in rotation(edge, "P_EdgeFire2Hold", "sphere1").0 {
        assert_eq!(particle, [0.0; 4]);
    }
    // Random ranges of 50 degrees about X and a full turn about Y and Z, no initial angle: the
    // stream is zero, the range is the shaders'.
    let (ring, range) = rotation(edge, "P_EdgeFire2Hold", "lightning_ring1_bk");
    assert!(!ring.is_empty());
    assert!((range[0] - 0.873).abs() < 1e-3, "{range:?}");
    assert!(ring.iter().all(|r| *r == [0.0; 4]), "{ring:?}");
    // Diddy's popgun chips: half a turn on every axis to start, a full turn of random range.
    let (chips, range) = rotation("fighter/diddy/ef_diddy.eff", "P_DiddyPopgun", "chip1");
    assert!(!chips.is_empty());
    assert!((range[0] - std::f32::consts::TAU).abs() < 1e-3, "{range:?}");
    let pi = std::f32::consts::PI;
    assert!(chips.iter().all(|r| *r == [pi, pi, pi, 0.0]), "{chips:?}");
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

/// Every emitter of a set under the game's effect folder, parsed and with its load-time fixes,
/// and its children, in the order the set creates them.
type SetEmitters = Vec<(effect_library::emitter::EmitterData, Vec<u8>, ptcl::Emitter)>;

fn set_emitters(file: &str, set: &str) -> Option<SetEmitters> {
    let root = std::env::var_os("VFX_GAME_EFFECTS")?;
    let data = std::fs::read(std::path::Path::new(&root).join(file)).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let index = p
        .emitter_list
        .emitter_sets
        .iter()
        .position(|s| s.name == set)
        .unwrap();
    let raws = &ptcl::emitter_sets(&p.base_bytes)[index];
    Some(
        p.emitter_list.emitter_sets[index]
            .emitters
            .iter()
            .zip(&raws.emitters)
            .map(|(e, raw)| {
                let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
                (e.data.clone(), fixed, raw.clone())
            })
            .collect(),
    )
}

/// One named emitter of a set: parsed, fixed and raw.
fn named(
    file: &str,
    set: &str,
    name: &str,
) -> Option<(effect_library::emitter::EmitterData, Vec<u8>, ptcl::Emitter)> {
    set_emitters(file, set)?
        .into_iter()
        .find(|(d, _, _)| d.name.as_deref() == Some(name))
}

fn v3(v: &[f32; 4]) -> glam::Vec3 {
    glam::Vec3::new(v[0], v[1], v[2])
}

/// Sephiroth's blue charge (P_EdgeFire2Hold) spawned twice. The game seeds every spawn of a set
/// anew (EmitterSet::ResetCommon draws the set's seed from its global generator; an emitter of
/// seed type 0 draws its own, type 1 takes the set's), so a looped or repeated effect never
/// plays the same particles twice. The preview gave every emitter one fixed seed, and each loop
/// was a copy of the last.
#[test]
fn every_spawn_draws_its_own_particles_as_the_game_seeds_it() {
    let Some(set) = set_emitters("fighter/edge/ef_edge.eff", "P_EdgeFire2Hold") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let run = |spawn: Option<u32>| -> Vec<Vec<[f32; 4]>> {
        let mut seeds = spawn.map(sim::SpawnSeeds::new);
        set.iter()
            .map(|(data, fixed, _)| {
                let mut s = match seeds.as_mut() {
                    Some(seeds) => sim::EmitterSim::with_seed(data, fixed, seeds.emitter(fixed)),
                    None => sim::EmitterSim::new(data, fixed),
                };
                for _ in 0..20 {
                    s.step();
                }
                let streams = s.streams();
                streams
                    .get("sysRandomAttr")
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .chain(streams.get("sysLocalPosAttr").cloned().unwrap_or_default())
                    .collect()
            })
            .collect()
    };
    // The same spawn plays the same particles, and so does the fixed seed: tests and the first
    // frame of a move stay repeatable.
    assert_eq!(run(Some(1)), run(Some(1)));
    assert_eq!(run(None), run(None));
    let (a, b) = (run(Some(1)), run(Some(2)));
    let mut differ = 0;
    for ((data, fixed, _), (a, b)) in set.iter().zip(a.iter().zip(&b)) {
        assert!(fixed[0x757] < 2, "{:?} has a fixed seed", data.name);
        if !a.is_empty() {
            assert_ne!(a, b, "{:?} played the same particles twice", data.name);
            differ += 1;
        }
    }
    assert!(differ >= 5, "{differ}");
}

/// The seed types in one real set (Alucard's transform, P_AlucardTransform): emitters of type 1
/// share the set's seed and so one sequence; type 0 draw their own; every spawn moves both.
#[test]
fn seed_type_one_emitters_share_the_sets_seed() {
    let Some(set) = set_emitters("assist/alucard/ef_alucard.eff", "P_AlucardTransform") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let seeds = |spawn| {
        let mut s = sim::SpawnSeeds::new(spawn);
        set.iter()
            .map(|(_, fixed, _)| (fixed[0x757], s.emitter(fixed)))
            .collect::<Vec<_>>()
    };
    let (a, b) = (seeds(7), seeds(8));
    let shared: Vec<u32> = a.iter().filter(|(t, _)| *t == 1).map(|(_, s)| *s).collect();
    let own: Vec<u32> = a.iter().filter(|(t, _)| *t == 0).map(|(_, s)| *s).collect();
    assert!(shared.len() >= 2 && own.len() >= 2, "{a:?}");
    assert!(shared.iter().all(|s| *s == shared[0]), "{a:?}");
    assert!(own.windows(2).all(|w| w[0] != w[1]) && !own.contains(&shared[0]));
    for ((_, x), (_, y)) in a.iter().zip(&b) {
        assert_ne!(x, y);
    }
}

/// Bowser Jr.'s Final Smash paint (P_KoopajrFinalPaintSplash paintA3) emits 10 a frame for five
/// frames with a 50% random rate, and its paint never dies. TimeEmit divides the random share
/// by -100: each check emits 10 * (1 - 0.5 r), 25 to 50 in all. The preview added it, 50 to 75.
#[test]
fn a_random_rate_takes_from_the_rate() {
    let Some((data, fixed, _)) = named(
        "fighter/koopajr/ef_koopajr.eff",
        "P_KoopajrFinalPaintSplash",
        "paintA3",
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert_eq!(
        (
            data.emission.rate,
            i32::from_le_bytes(fixed[0x804..0x808].try_into().unwrap())
        ),
        (10.0, 50)
    );
    let mut counts = Vec::new();
    for spawn in 0..40 {
        let mut s =
            sim::EmitterSim::with_seed(&data, &fixed, sim::SpawnSeeds::new(spawn).emitter(&fixed));
        for _ in 0..12 {
            s.step();
        }
        counts.push(s.count());
    }
    for n in &counts {
        assert!((24..=50).contains(n), "{counts:?}");
    }
    let mean = counts.iter().sum::<usize>() as f32 / counts.len() as f32;
    assert!((mean - 37.5).abs() < 4.0, "{mean} {counts:?}");
}

/// The Darz tear-up sparks (P_DarzTearUpAnger particle1_H) fly up at a diffusion angle of 90
/// degrees. InitializeParticle spreads the direction over a cap whose height runs evenly from
/// 1 - angle / 90 to 1: here every way above the horizon, half of them below 30 degrees up. The
/// preview leant a unit vector times sin(angle) on the direction, which bunched them upward
/// (and at 180 degrees spread them not at all).
#[test]
fn a_diffusion_angle_spreads_directions_over_its_cap() {
    let Some((data, fixed, _)) = named("boss/darz/ef_darz.eff", "P_DarzTearUpAnger", "particle1_H")
    else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert_eq!(data.particle_velocity.diffusion_dir_angle, 90.0);
    let mut s = sim::EmitterSim::new(&data, &fixed);
    s.spawn_from_seed(0x1234_5678, 0.0, 1000);
    let heights: Vec<f32> = s.streams()["sysLocalVecAttr"]
        .iter()
        .map(|v| v3(v).normalize().y)
        .collect();
    assert_eq!(heights.len(), 1000);
    let under = heights.iter().filter(|y| **y < -1e-4).count();
    assert_eq!(under, 0);
    let mean = heights.iter().sum::<f32>() / heights.len() as f32;
    let low = heights.iter().filter(|y| **y < 0.5).count();
    assert!((mean - 0.5).abs() < 0.04, "{mean}");
    assert!((420..=580).contains(&low), "{low}");
}

/// Jiro's spin ball (P_JiroNSpin ball1_main_hight) is a point emitter with a position random of
/// 0.5: InitializeParticle adds a direction from the runtime's table times 0.5, so particles
/// start on a sphere about the point. The preview jittered them in the XZ plane only.
#[test]
fn position_random_scatters_points_over_a_sphere() {
    let Some((data, fixed, _)) =
        named("assist/jiro/ef_jiro.eff", "P_JiroNSpin", "ball1_main_hight")
    else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert_eq!(data.shape_info.volume_type, 0);
    let mut s = sim::EmitterSim::new(&data, &fixed);
    s.spawn_from_seed(99, 0.0, 400);
    let points: Vec<glam::Vec3> = s.streams()["sysLocalPosAttr"].iter().map(v3).collect();
    for p in &points {
        assert!((p.length() - 0.5).abs() < 1e-3, "{p}");
    }
    let high = points.iter().filter(|p| p.y.abs() > 0.25).count();
    assert!((150..=250).contains(&high), "{high}");
}

/// The Dark Knight's smoke (P_DarknightSmoke smoke2_L) has an XZ diffusion of 0.25: each
/// particle also flies out across the emitter's Y axis, away from it, at that speed times its
/// speed random. The preview had no XZ diffusion. Compared here against the same emitter with
/// the value cleared, which draws the same numbers.
#[test]
fn xz_diffusion_pushes_particles_out_from_the_axis() {
    let Some((data, fixed, _)) = named(
        "assist/darkknight/ef_darkknight.eff",
        "P_DarknightSmoke",
        "smoke2_L",
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let xz = f32::from_le_bytes(fixed[0x97c..0x980].try_into().unwrap());
    assert_eq!(xz, 0.25);
    let mut cleared = fixed.clone();
    cleared[0x97c..0x980].copy_from_slice(&0f32.to_le_bytes());
    let born = |raw: &[u8]| {
        let mut s = sim::EmitterSim::new(&data, raw);
        s.spawn_from_seed(5, 0.0, 60);
        let st = s.streams();
        (st["sysLocalPosAttr"].clone(), st["sysLocalVecAttr"].clone())
    };
    let ((pos, with), (_, without)) = (born(&fixed), born(&cleared));
    let low = 1.0 - data.particle_velocity.vel_random / 100.0;
    for ((p, a), b) in pos.iter().zip(&with).zip(&without) {
        let push = v3(a) - v3(b);
        let out = glam::Vec3::new(p[0], 0.0, p[2]).normalize();
        assert!(push.y.abs() < 1e-5, "{push}");
        assert!(push.normalize().dot(out) > 0.999, "{push} {out}");
        assert!(
            (xz * low - 1e-4..=xz + 1e-4).contains(&push.length()),
            "{push}"
        );
    }
}

/// Mii Gunner's absorber grains (P_MiiGunnerAbsorber Tsubu1) move only by a random velocity of
/// one unit on each axis (res+0x980), read from the runtime's cube table. The preview had no
/// such velocity: the grains sat still.
#[test]
fn random_velocity_per_axis_comes_from_the_cube_table() {
    let Some((data, fixed, _)) = named(
        "fighter/miigunner/ef_miigunner.eff",
        "P_MiiGunnerAbsorber",
        "Tsubu1",
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let mut s = sim::EmitterSim::new(&data, &fixed);
    s.spawn_from_seed(3, 0.0, 300);
    let vel: Vec<glam::Vec3> = s.streams()["sysLocalVecAttr"].iter().map(v3).collect();
    for v in &vel {
        assert!(v.abs().max_element() < 1.0 + 1e-4, "{v}");
    }
    for axis in 0..3 {
        let fast = vel.iter().filter(|v| v[axis].abs() > 0.5).count();
        assert!((100..=200).contains(&fast), "{axis} {fast}");
    }
}

/// Corrin's victory sword aura (P_KamuiSwordAuraWin aura1) redraws its own transform after
/// every emission (res+0x758, Smash 0x30e0): each emission comes from a new spot within half a
/// unit of its place on each axis. The preview kept the first spot.
#[test]
fn an_emitter_that_moves_by_emission_draws_a_new_place_each_time() {
    let Some((data, fixed, _)) =
        named("fighter/kamui/ef_kamui.eff", "P_KamuiSwordAuraWin", "aura1")
    else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert_eq!(fixed[0x758], 1);
    let base = glam::Vec3::new(
        f32::from_le_bytes(fixed[0x770..0x774].try_into().unwrap()),
        f32::from_le_bytes(fixed[0x774..0x778].try_into().unwrap()),
        f32::from_le_bytes(fixed[0x778..0x77c].try_into().unwrap()),
    );
    let mut s = sim::EmitterSim::new(&data, &fixed);
    let mut places = Vec::new();
    for _ in 0..12 {
        s.step();
        places.push(s.emitter_srt(&glam::Mat4::IDENTITY).w_axis.truncate());
    }
    for p in &places {
        assert!((*p - base).abs().max_element() <= 0.5 + 1e-4, "{p} {base}");
    }
    let mut distinct = places.clone();
    distinct.dedup();
    assert!(distinct.len() >= 8, "{places:?}");
}

/// Bomberman's bomb sparks (P_BombermanBomb spark1) carry the simple random field (FRN1): every
/// five frames of a spark's life (from its first) its velocity gains a point of the runtime's
/// cube table times 0.13. The preview did not run it, and every spark flew straight. Each
/// spark's velocity is followed from frame to frame (by its random values), less what air
/// resistance and gravity did: what is left is the field's kick.
#[test]
fn the_simple_random_field_kicks_velocities_every_interval() {
    let Some((data, fixed, raw)) = named(
        "assist/bomberman/ef_bomberman.eff",
        "P_BombermanBomb",
        "spark1",
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let fields = sim::Fields::of(&raw.attributes);
    let kick = fields.random_kick.expect("FRN1");
    assert_eq!((kick.power.x, kick.interval), (0.13, 5));
    assert!(fields.random_noise.is_none() && fields.spin.is_none() && fields.convergence.is_none());
    assert!(fields.curl_noise.is_none() && fields.position_add.is_none());
    let statics = &data.emitter_static;
    let gravity = glam::Vec3::new(
        statics.gravity_dir_x,
        statics.gravity_dir_y,
        statics.gravity_dir_z,
    ) * statics.gravity_scale;
    let mut s = sim::EmitterSim::new(&data, &fixed);
    s.set_fields(fields);
    let mut last: std::collections::HashMap<[u32; 4], glam::Vec3> = Default::default();
    let (mut kicked, mut quiet) = (0, 0);
    for _ in 0..16 {
        let before = s.time;
        s.step();
        let st = s.streams();
        let (Some(random), Some(vec)) = (st.get("sysRandomAttr"), st.get("sysLocalVecAttr")) else {
            continue;
        };
        let mut now = std::collections::HashMap::new();
        for (r, v) in random.iter().zip(vec) {
            let id = r.map(f32::to_bits);
            let birth = v[3];
            let v = v3(v);
            if let Some(prev) = last.get(&id) {
                let age = before - birth;
                let left = v - (*prev * statics.air_res + gravity);
                if (age as u32).is_multiple_of(5) {
                    assert!(left.abs().max_element() <= 0.13 + 1e-4, "{left} at {age}");
                    kicked += usize::from(left.length() > 1e-3);
                } else {
                    assert!(left.length() < 1e-4, "{left} at {age}");
                    quiet += 1;
                }
            }
            now.insert(id, v);
        }
        last = now;
    }
    assert!(kicked >= 3 && quiet >= 10, "{kicked} {quiet}");
}

/// The Rathalos fireball's back flare (P_LioleusFireBall flare_back) carries the random noise
/// field (FRND, Smash 0x11ad0): each particle's velocity follows four sine waves along its own
/// life, from a start its random values pick, so each wanders its own way. The preview did not
/// run it.
#[test]
fn the_random_noise_field_moves_each_particle_its_own_way() {
    let Some((data, fixed, raw)) = named(
        "assist/lioleus/ef_lioleus.eff",
        "P_LioleusFireBall",
        "flare_back",
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let fields = sim::Fields::of(&raw.attributes);
    let noise = fields.random_noise.expect("FRND");
    assert!(
        noise.period > 0.0 && noise.power.length() > 0.0,
        "{noise:?}"
    );
    // The noise draws nothing, so with it and without it the same particles are born.
    assert!(fields.random_kick.is_none());
    let run = |fields: sim::Fields| {
        let mut s = sim::EmitterSim::new(&data, &fixed);
        s.set_fields(fields);
        s.spawn_from_seed(77, 0.0, 12);
        for _ in 0..6 {
            s.step();
        }
        s.streams()["sysLocalVecAttr"].clone()
    };
    let (noisy, plain) = (run(fields), run(sim::Fields::default()));
    assert_eq!(noisy.len(), plain.len());
    assert!(noisy.len() >= 4, "{}", noisy.len());
    let pushes: Vec<glam::Vec3> = noisy
        .iter()
        .zip(&plain)
        .map(|(a, b)| v3(a) - v3(b))
        .collect();
    // Same births either way.
    for (a, b) in noisy.iter().zip(&plain) {
        assert_eq!(a[3], b[3]);
    }
    assert!(
        pushes.iter().all(|p| p.length() > 1e-4),
        "{:?}",
        &pushes[..4]
    );
    assert!(
        pushes.windows(2).all(|w| (w[0] - w[1]).length() > 1e-4),
        "{:?}",
        &pushes[..4]
    );
}

/// Shares of particles that start within half the shape's size (`size` gives a particle's
/// distance out as a share of it), from 2000 at one seed.
fn inner_share(file: &str, set: &str, name: &str, size: impl Fn(glam::Vec3) -> f32) -> Option<f32> {
    let (data, fixed, _) = named(file, set, name)?;
    let mut s = sim::EmitterSim::new(&data, &fixed);
    s.spawn_from_seed(0xbeef, 0.0, 2000);
    let points: Vec<glam::Vec3> = s.streams()["sysLocalPosAttr"].iter().map(v3).collect();
    assert_eq!(points.len(), 2000);
    Some(points.iter().filter(|p| size(**p) < 0.5).count() as f32 / points.len() as f32)
}

/// Fills spread like the game's. The Kawasaki plate's circle fill (caliber 1, radius 2) takes
/// the square root of an even share of the area (Smash 0xb6f0): a quarter within half the
/// radius. The Thwomp's smoke sphere fill (caliber 1, radius 15.5) takes sqrt(r) of the radius
/// (Smash 0xc890): a quarter too. The preview put half of each inside half the radius.
#[test]
fn filled_circles_and_spheres_spread_like_the_game() {
    let Some(circle) = inner_share(
        "assist/kawasaki/ef_kawasaki.eff",
        "P_KawasakiPlate",
        "P_plate1",
        |p| p.length() / 2.0,
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert!((circle - 0.25).abs() < 0.03, "{circle}");
    let sphere = inner_share("assist/dossun/ef_dossun.eff", "P_DossunIn", "smoke2", |p| {
        p.length() / 15.5
    })
    .unwrap();
    assert!((sphere - 0.25).abs() < 0.03, "{sphere}");
}

/// Inkling Shiokarazu's ground splash (P_ShiokarazuFootKimiiro P_ground_G) fills a box 60 by 50
/// by 10 with a caliber of 0.5: hollow inside half its size (Smash 0xd140). The preview filled
/// it right through.
#[test]
fn a_filled_box_with_a_caliber_stays_out_of_its_hollow() {
    let Some(inner) = inner_share(
        "assist/shiokarazu/ef_shiokarazu.eff",
        "P_ShiokarazuFootKimiiro",
        "P_ground_G",
        |p| (p / glam::Vec3::new(60.0, 50.0, 10.0)).abs().max_element(),
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert!(inner < 0.001, "{inner}");
}

/// The Hebigarasu smoke stones (P_HebigarasuSmoke stone2) fly out of a sphere with a latitude
/// of 90 degrees about +Y: every direction above the horizon (Smash 0xb9d0). The preview threw
/// half of them down.
#[test]
fn a_sphere_with_a_latitude_keeps_to_its_cap() {
    let Some((data, fixed, _)) = named(
        "assist/hebigarasu/ef_hebigarasu.eff",
        "P_HebigarasuSmoke",
        "stone2",
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert_eq!(
        (data.shape_info.volume_type, fixed[0x83a], fixed[0x83e]),
        (4, 1, 2)
    );
    let mut s = sim::EmitterSim::new(&data, &fixed);
    s.spawn_from_seed(11, 0.0, 500);
    let points: Vec<glam::Vec3> = s.streams()["sysLocalPosAttr"].iter().map(v3).collect();
    // Its position random (res+0x810) scatters each stone a little about its place.
    let scatter = f32::from_le_bytes(fixed[0x810..0x814].try_into().unwrap());
    let below = points.iter().filter(|p| p.y < -scatter - 1e-3).count();
    let above = points.iter().filter(|p| p.y > 0.0).count();
    assert_eq!(below, 0, "scatter {scatter}");
    assert!(above >= 470, "{above}");
}

/// Kirby's Final Smash stars (P_KirbyFinalStarB star1) pick their colour from four keys by
/// each star's random value (colour type 3), and their trail (star1_Child) takes the star's
/// colour. effect_library reads type 3 as Unknown(3) (it calls type 1 Random, which no game file
/// uses), so every trail took the star's constant colour instead of its key.
#[test]
fn a_child_takes_its_parents_random_colour_key() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let data =
        std::fs::read(std::path::Path::new(&root).join("fighter/kirby/ef_kirby.eff")).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let set = p
        .emitter_list
        .emitter_sets
        .iter()
        .position(|s| s.name == "P_KirbyFinalStarB")
        .unwrap();
    let index = p.emitter_list.emitter_sets[set]
        .emitters
        .iter()
        .position(|e| e.data.name.as_deref() == Some("star1"))
        .unwrap();
    let parsed = &p.emitter_list.emitter_sets[set].emitters[index];
    let raw = &ptcl::emitter_sets(&p.base_bytes)[set].emitters[index];
    let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
    let child_fixed = static_block::fixed_emitter_data(&raw.children[0], &|_| None, None);
    assert_eq!(parsed.children[0].data.name.as_deref(), Some("star1_Child"));
    assert_eq!(
        (
            parsed.data.particle_color.color0_type.as_u8(),
            child_fixed[0x7dc]
        ),
        (3, 1)
    );
    let keys: Vec<[f32; 3]> = parsed.data.emitter_static.color0.keys
        [..parsed.data.emitter_static.num_color0_keys as usize]
        .iter()
        .map(|k| [k.x, k.y, k.z])
        .collect();
    let distinct_keys = {
        let mut k = keys.clone();
        k.dedup();
        k.len()
    };
    assert!(distinct_keys >= 2, "{keys:?}");
    let mut s = sim::EmitterSim::new(&parsed.data, &fixed);
    s.add_child(&parsed.children[0].data, &child_fixed);
    let mut colours: Vec<[f32; 3]> = Vec::new();
    for _ in 0..40 {
        s.step();
        if let Some(c) = s.children()[0].streams().get("sysColor0Attr") {
            colours.extend(c.iter().map(|c| [c[0], c[1], c[2]]));
        }
    }
    assert!(colours.len() >= 10, "{}", colours.len());
    let emitter =
        |k: usize| f32::from_le_bytes(fixed[0x7ac + 4 * k..0x7b0 + 4 * k].try_into().unwrap());
    // The game hands the key times the star's colour scale (res+0x3b0, 1.6 here; Smash 0x1066c
    // in CalculateParticleColor0VecFromTime) and its emitter colour.
    let scale = f32::from_le_bytes(fixed[0x3b0..0x3b4].try_into().unwrap());
    assert_eq!(scale, 1.6);
    let near =
        |a: [f32; 3], b: [f32; 3]| (0..3).all(|k| (a[k] - b[k] * emitter(k) * scale).abs() < 1e-4);
    for c in &colours {
        assert!(
            keys.iter().any(|k| near(*c, *k)),
            "{c:?} is none of {keys:?}"
        );
    }
    let mut seen: Vec<[f32; 3]> = Vec::new();
    for c in &colours {
        if !seen.contains(c) {
            seen.push(*c);
        }
    }
    assert!(seen.len() >= 2, "{seen:?}");
}

/// A game matrix as the live probe prints it: three rows of (x, y, z, translation).
fn probe_rows(r: [f32; 12]) -> glam::Mat4 {
    glam::Mat4::from_cols_array(&[
        r[0], r[4], r[8], 0.0, r[1], r[5], r[9], 0.0, r[2], r[6], r[10], 0.0, r[3], r[7], r[11],
        1.0,
    ])
}

/// Mario's fireball shoot flame (MARIO_FB_SHOOT fire1) pulls up along the world's Y (world
/// gravity, res+0x7f1) although the emitter is turned 130 degrees about X and rides his hand.
/// Read live from the game (sd:/slight/vfx_probe.txt, Mario's neutral special facing right,
/// ~/.cache/builds/visionary/vfx-re/vfx_place_fb.txt, emitter 0x11e1ae3250): the emitter's
/// matrix each frame and the particle's velocity in emitter space. Each frame's change, less
/// the air resistance, is the gravity the game added: almost all of it along the emitter's X,
/// which the hand points up, and none along its own Y. The preview pulled every emitter along
/// its own Y, so this flame flowed sideways out of the hand instead of rising.
#[test]
fn world_gravity_pulls_along_the_worlds_axes_as_the_game_does() {
    let Some((data, fixed, _)) = named(
        "fighter/mario/ef_mario.eff",
        "P_MarioFireballShoot",
        "fire1",
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert_eq!(fixed[0x7f1], 1, "fire1 is a world gravity emitter");
    // Emitter matrices at emitter times 2..7 and the particle's velocity at times 2..8.
    let matrices = [
        [
            -0.100, 1.378, -0.644, -19.738, 1.823, 0.027, -0.278, 8.138, -0.256, -0.601, -1.702,
            0.168,
        ],
        [
            -0.073, 1.288, -0.627, -19.826, 1.704, -0.002, -0.253, 8.013, -0.240, -0.594, -1.579,
            0.174,
        ],
        [
            -0.061, 1.214, -0.596, -19.921, 1.607, -0.015, -0.246, 7.930, -0.238, -0.572, -1.486,
            0.213,
        ],
        [
            -0.058, 1.163, -0.569, -20.001, 1.539, -0.021, -0.250, 7.890, -0.243, -0.553, -1.422,
            0.263,
        ],
        [
            -0.061, 1.132, -0.551, -20.063, 1.495, -0.023, -0.263, 7.884, -0.255, -0.543, -1.381,
            0.312,
        ],
        [
            -0.069, 1.110, -0.537, -20.116, 1.463, -0.021, -0.280, 7.903, -0.270, -0.539, -1.352,
            0.359,
        ],
    ]
    .map(probe_rows);
    let game = [
        [-0.047, -0.003, 0.029],
        [-0.015, -0.002, 0.023],
        [0.016, -0.002, 0.017],
        [0.044, -0.002, 0.011],
        [0.072, -0.002, 0.004],
        [0.098, -0.003, -0.002],
        [0.122, -0.002, -0.008],
    ]
    .map(glam::Vec3::from);
    let air = data.emitter_static.air_res;
    // What gravity added over a frame: the change less the air resistance's share.
    let pull = |before: glam::Vec3, after: glam::Vec3| after - before * air;
    let mut s = sim::EmitterSim::new(&data, &fixed);
    let local = s.emitter_srt(&glam::Mat4::IDENTITY);
    let set = |m: &glam::Mat4| *m * local.inverse();
    for _ in 0..10 {
        if s.count() > 0 {
            break;
        }
        s.step_at(&set(&matrices[0]));
    }
    assert!(s.count() > 0);
    let velocity = |s: &sim::EmitterSim| v3(&s.streams()["sysLocalVecAttr"][0]);
    for (k, m) in matrices.iter().enumerate().skip(1) {
        let before = velocity(&s);
        s.step_at(&set(m));
        let ours = pull(before, velocity(&s));
        let theirs = pull(game[k - 1], game[k]);
        assert!(
            (ours - theirs).length() < 0.004,
            "frame {}: preview adds {ours}, the game {theirs}",
            k + 2
        );
    }
}

/// The fireball's flame trail (MARIO_FB_BULLET_R fire1) does not follow its emitter (res+0x753
/// = 1): each flame stays where it was born while the fireball rolls on. Its world gravity is
/// turned into the flame's space by the emitter's matrix at the flame's birth, not the one the
/// rolling fireball has now. Read live from the game (vfx_place_fb.txt, emitter 0x11e1ae4b10):
/// the flame born at time 2 under the first matrix keeps taking that matrix's share of gravity
/// while the fireball turns 60 degrees more over the next two frames.
#[test]
fn world_gravity_of_a_particle_left_behind_turns_by_its_birth_matrix() {
    let Some((data, fixed, _)) = named(
        "fighter/mario/ef_mario.eff",
        "P_MarioFireballBulletR",
        "fire1",
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert_eq!((fixed[0x7f1], fixed[0x753]), (1, 1));
    let matrices = [
        [
            -0.000, 0.748, 0.664, -17.504, 0.000, 0.664, -0.748, 7.333, -1.000, -0.000, -0.000,
            -0.000,
        ],
        [
            -0.000, 0.978, 0.207, -15.710, 0.000, 0.207, -0.978, 5.968, -1.000, -0.000, -0.000,
            -0.000,
        ],
        [
            -0.000, 0.938, -0.346, -14.403, 0.000, -0.346, -0.938, 4.329, -1.000, -0.000, 0.000,
            -0.000,
        ],
    ]
    .map(probe_rows);
    // The first flame's velocity at times 2, 3 and 4.
    let game = [
        [-0.159, -0.208, -0.350],
        [-0.154, -0.189, -0.355],
        [-0.149, -0.170, -0.359],
    ]
    .map(glam::Vec3::from);
    let air = data.emitter_static.air_res;
    let pull = |before: glam::Vec3, after: glam::Vec3| after - before * air;
    let mut s = sim::EmitterSim::new(&data, &fixed);
    let local = s.emitter_srt(&glam::Mat4::IDENTITY);
    let set = |m: &glam::Mat4| *m * local.inverse();
    // It emits by the distance it travels (res+0x7f2): it comes to the first matrix along the
    // fireball's path, a frame's move at a time, turned as it is there.
    assert_eq!(fixed[0x7f2], 1);
    let step = matrices[1].w_axis - matrices[0].w_axis;
    for back in (0..8).rev() {
        let mut at = matrices[0];
        at.w_axis -= step * back as f32;
        s.step_at(&set(&at));
    }
    assert!(s.count() > 0);
    let velocity = |s: &sim::EmitterSim| v3(&s.streams()["sysLocalVecAttr"][0]);
    for (k, m) in matrices.iter().enumerate().skip(1) {
        let before = velocity(&s);
        s.step_at(&set(m));
        let ours = pull(before, velocity(&s));
        let theirs = pull(game[k - 1], game[k]);
        assert!(
            (ours - theirs).length() < 0.004,
            "frame {}: preview adds {ours}, the game {theirs}",
            k + 2
        );
    }
}

/// Sephiroth's up special throws its wind (EDGE_OCTASLASH_ARC2 wind2) up the screen however
/// the move turns the arc: the emitter's velocity is world-oriented (res+0x7f3), so
/// InitializeParticle (Smash 0xa2d0) takes its designated direction through the inverse of the
/// emitter's matrix before the shaders turn it back. The move spawns the arc turned 196 degrees
/// about Z (SpecialHi2, frame 2); taken in the emitter's own space, the wind blew down.
#[test]
fn a_world_oriented_velocity_keeps_its_world_direction() {
    let Some((data, fixed, _)) = named("fighter/edge/ef_edge.eff", "P_EdgeOctaslashArc2", "wind2")
    else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert_eq!(fixed[0x7f3], 1);
    let f = |at: usize| f32::from_le_bytes(fixed[at..at + 4].try_into().unwrap());
    let designated = glam::Vec3::new(f(0x96c), f(0x970), f(0x974));
    // The script's turn (-0, -3.5, 196 degrees, X then Y then Z) at the call's size of 1.2.
    let turn = glam::Quat::from_euler(
        glam::EulerRot::ZYX,
        196f32.to_radians(),
        (-3.5f32).to_radians(),
        0.0,
    );
    let set =
        glam::Mat4::from_scale_rotation_translation(glam::Vec3::splat(1.2), turn, glam::Vec3::ZERO);
    let mut s = sim::EmitterSim::new(&data, &fixed);
    for _ in 0..10 {
        if s.count() > 0 {
            break;
        }
        s.step_at(&set);
    }
    assert!(s.count() > 0);
    let srt = s.emitter_srt(&set);
    // Its own spreads (outward speed, random velocity) lean it a little off the direction.
    for v in &s.streams()["sysLocalVecAttr"] {
        let world = srt.transform_vector3(v3(v));
        let along = world.normalize().dot(designated.normalize());
        assert!(along > 0.9, "{world} does not blow along {designated}");
    }
}

/// A set's emitter with its children, parsed and with their load-time fixes, run by `f`.
fn with_children<T>(
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
    Some(f(&mut s))
}

/// Sephiroth's Scintilla wall (P_EdgeSenkouShield HexagonFlare01) is seven child emitters, one
/// row of hexagons each, placed by its own translation: 9.9, 5.9, 1.95, -1.95, -5.9, -9.9 and
/// -13.8 up the wall. A child's matrix is its parent's times its own (Smash 0x8090). Children
/// were drawn on their parent's matrix, so all seven rows fell on one line: the tester's
/// "horizontal row of stacked hexagons" beside Sephiroth's chest. The set is turned and scaled
/// as edge/SpecialLw spawns it (Y -125, scale 0.85); every row's particles must stand at its
/// height in the set's space, and draw at the set's scale.
#[test]
fn child_emitters_stand_on_their_own_transform() {
    use glam::{Mat4, Quat, Vec3};
    let set = Mat4::from_scale_rotation_translation(
        Vec3::splat(0.85),
        Quat::from_rotation_y((-125f32).to_radians()),
        Vec3::new(30.0, 13.0, 9.0),
    );
    let Some(rows) = with_children(
        "fighter/edge/ef_edge.eff",
        "P_EdgeSenkouShield",
        "HexagonFlare01",
        |s| {
            for _ in 0..8 {
                s.step_at(&set);
            }
            s.children()
                .iter()
                .map(|child| {
                    assert!(child.count() >= 4, "{} hexagons", child.count());
                    let srt = child.emitter_srt(&set);
                    let streams = child.streams();
                    let in_set = |world: Vec3| set.inverse().transform_point3(world).y;
                    let heights: Vec<f32> = streams["sysLocalPosAttr"]
                        .iter()
                        .map(|p| in_set(srt.transform_point3(v3(p))))
                        .collect();
                    // Each particle's birth matrix (the shaders' sysEmtMat rows) stands there
                    // too.
                    let births: Vec<f32> = (0..child.count())
                        .map(|i| {
                            let row = |name: &str| streams[name][i][3];
                            in_set(Vec3::new(
                                row("sysEmtMat0Attr"),
                                row("sysEmtMat1Attr"),
                                row("sysEmtMat2Attr"),
                            ))
                        })
                        .collect();
                    let block = child.dynamic_block(&srt, &child.emitter_rt(&set), [1.0; 4]);
                    let scale = f32::from_le_bytes(block[0x34..0x38].try_into().unwrap());
                    (heights, births, scale)
                })
                .collect::<Vec<_>>()
        },
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let want = [9.9, 5.9, 1.95, -1.95, -5.9, -9.9, -13.8];
    assert_eq!(rows.len(), want.len());
    for ((heights, births, scale), want) in rows.iter().zip(want) {
        for y in heights.iter().chain(births) {
            assert!(
                (y - want).abs() < 1e-3,
                "row at {y}, file puts it at {want}"
            );
        }
        assert!((scale - 0.85).abs() < 1e-5, "particle scale {scale}");
    }
}

/// Sephiroth's victory sparks (P_EdgeWinSparks sparks1_b) keep the matrix they were born with
/// (res+0x753 = 1), so the game hands their child (Emitter2) each spark's position in the
/// world (Smash 0x8d70) and leaves the child on its own transform alone (Smash 0x8090): moving
/// or turning the set afterwards does not move the child's emitter, yet its particles start
/// where the sparks are, at the set, not at the world's origin.
#[test]
fn a_child_of_particles_that_keep_their_birth_matrix_lives_in_the_world() {
    use glam::{Mat4, Quat, Vec3};
    let at = Vec3::new(500.0, 300.0, -200.0);
    let set = Mat4::from_rotation_translation(Quat::from_rotation_y(1.0), at);
    let Some((fixed_matrix, distances)) = with_children(
        "fighter/edge/ef_edge.eff",
        "P_EdgeWinSparks",
        "sparks1_b",
        |s| {
            for _ in 0..40 {
                s.step_at(&set);
            }
            let child = &s.children()[1];
            assert!(child.count() > 0);
            let elsewhere = Mat4::from_translation(Vec3::new(-40.0, 7.0, 3.0));
            let fixed_matrix = child.emitter_srt(&set) == child.emitter_srt(&elsewhere);
            let srt = child.emitter_srt(&set);
            let distances: Vec<f32> = child.streams()["sysLocalPosAttr"]
                .iter()
                .map(|p| (srt.transform_point3(v3(p)) - at).length())
                .collect();
            (fixed_matrix, distances)
        },
    ) else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    assert!(fixed_matrix);
    for d in distances {
        assert!(d < 100.0, "a child particle {d} from the set");
    }
}

/// Roy's eruption (P_RoyEruption flare2) sits 8 above the set by its own translation, and its
/// child (smoke3) emits from its particles. The child's matrix is its parent's matrix, set and
/// parent's own transform both, times its own (Smash 0x8090), and the child's particles keep
/// that as their birth matrix (smoke3 res+0x753 = 1): the smoke rises from the flare, 8 up,
/// not from the set's origin.
#[test]
fn a_childs_matrix_carries_its_parents_own_transform() {
    use glam::{Mat4, Quat, Vec3};
    let set =
        Mat4::from_rotation_translation(Quat::from_rotation_z(0.5), Vec3::new(-20.0, 4.0, 7.0));
    let Some((emitter, births)) =
        with_children("fighter/roy/ef_roy.eff", "P_RoyEruption", "flare2", |s| {
            for _ in 0..12 {
                s.step_at(&set);
            }
            let child = &s.children()[0];
            let births: Vec<Vec3> = child.streams()["sysEmtMat0Attr"]
                .iter()
                .zip(&child.streams()["sysEmtMat1Attr"])
                .zip(&child.streams()["sysEmtMat2Attr"])
                .map(|((a, b), c)| Vec3::new(a[3], b[3], c[3]))
                .collect();
            (child.emitter_srt(&set).w_axis.truncate(), births)
        })
    else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let flare = set.transform_point3(Vec3::new(0.0, 8.0, 0.0));
    assert!(
        (emitter - flare).length() < 1e-4,
        "{emitter} is not at the flare {flare}"
    );
    assert!(!births.is_empty());
    for at in births {
        assert!(
            (at - flare).length() < 1e-4,
            "smoke born on {at}, the flare is at {flare}"
        );
    }
}
