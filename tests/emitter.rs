//! Emitter state against a real effect. Point VFX_GAME_EFFECTS at the game's `effect` folder
//! (skipped when unset).
use vfx_runtime::{ptcl, sim, static_block};

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
    let file = vfx_runtime::bnsh::ShaderFile::read(&bin).unwrap();
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
