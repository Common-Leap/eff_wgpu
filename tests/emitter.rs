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
    // fire1 (emitter 0): distance emission, one particle every 2.0 units, frames 2-5.
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
    // 1.4 a frame over frames 2-5: 1.4, 2.8 (one, 0.8 left), 2.2 (one, 0.2 left), 1.6.
    assert_eq!(counts(1.4), 2);
    // 3.0 a frame: one per frame plus the carried remainder, six in all.
    assert_eq!(counts(3.0), 6);
}
