//! A vertex stage given all eight keys of its tables draws what it drew, and draws keys it was
//! not made with. Point VFX_GAME_EFFECTS at the game's `effect` folder (skipped when unset or
//! when there is no GPU).
use eff_wgpu::maxwell::keys::{animate, keys_read, StageCode, Table};
use eff_wgpu::{bnsh, ptcl, render, sim, static_block, texture, view_block};
use glam::{Mat4, Vec3};

struct Game {
    renderers: [render::EffectRenderer; 2],
    queue: wgpu::Queue,
    white: render::GpuTexture,
    sampler: wgpu::Sampler,
    ptcl: effect_library::ptcl_file::PtclFile,
    sets: Vec<ptcl::EmitterSet>,
    /// Each program's tables that now run through eight keys.
    widened: Vec<Vec<Table>>,
    read: Vec<[u32; 5]>,
}

/// A fighter's file, with a renderer of its shaders as they are and one of them widened.
fn game(file: &str) -> Option<Game> {
    let root = std::env::var_os("VFX_GAME_EFFECTS")?;
    let data = std::fs::read(std::path::Path::new(&root).join(file)).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let ptcl = eff.ptcl_file.unwrap();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    let binary = ptcl.shader_info.as_ref()?.binary_data.as_ref()?;
    let shaders = bnsh::ShaderFile::read(binary).unwrap();
    let raw = effect_library::bnsh::BnshFile::read(binary).unwrap();
    let mut wide = shaders.clone();
    let (mut widened, mut read) = (Vec::new(), Vec::new());
    for (program, variation) in wide.programs.iter_mut().zip(&raw.variations) {
        let mut tables = Vec::new();
        let mut counts = [0; 5];
        if let (Some(stage), Some(code)) = (
            program.stages[bnsh::STAGE_VERTEX].as_mut(),
            variation.binary_program.stages[bnsh::STAGE_VERTEX].as_ref(),
        ) {
            let keys = keys_read(&stage.code);
            counts = keys.map(|k| k.keys);
            tables = Table::ALL
                .into_iter()
                .filter(|t| keys[*t as usize].keys > 0 && !keys[*t as usize].picks)
                .collect();
            let out = animate(
                &StageCode {
                    control: code.control_code.clone(),
                    code: code.byte_code.clone(),
                },
                &stage.reflection.inputs,
                &tables,
            )
            .unwrap();
            stage.code = out.code[0x30..].to_vec();
            stage.control = out.control;
        }
        widened.push(tables);
        read.push(counts);
    }
    let target = wgpu::TextureFormat::Rgba8Unorm;
    let renderers = [shaders, wide].map(|s| {
        let r = render::EffectRenderer::new(&device, &queue, &s, target, None);
        let eye = Vec3::new(0.0, 0.0, 25.0);
        let rows = |m: Mat4| m.transpose().to_cols_array_2d();
        let cam = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
        let proj =
            glam::camera::rh::proj::directx::perspective(45f32.to_radians(), 1.0, 1.0, 1000.0);
        let view = view_block::view_block(&rows(cam), &rows(proj), eye.into(), 1.0, 1000.0, 1.0);
        r.set_view(&queue, &view);
        r
    });
    let white = render::upload_texture(&device, &queue, &texture::decode(&ptcl, 0).unwrap());
    let sampler = render::make_sampler(&device, 0, 0, 0, 8.0);
    Some(Game {
        renderers,
        queue,
        white,
        sampler,
        sets: ptcl::emitter_sets(&ptcl.base_bytes),
        ptcl,
        widened,
        read,
    })
}

impl Game {
    /// What an emitter's vertex shader puts out after `steps` frames, from either renderer.
    fn run(
        &self,
        (set, emitter): (usize, usize),
        steps: u32,
        wide: bool,
        change: impl Fn(&mut Vec<u8>),
    ) -> Option<(Vec<[f32; 4]>, Vec<f32>)> {
        let mut raw = self.sets[set].emitters[emitter].clone();
        change(&mut raw.data);
        let parsed = &self.ptcl.emitter_list.emitter_sets[set].emitters[emitter].data;
        let fixed = static_block::fixed_emitter_data(&raw, &|_| None, None);
        let mut s = sim::EmitterSim::new(parsed, &fixed);
        for frame in 0..steps {
            s.step_at(&Mat4::from_translation(Vec3::X * (1.4 * frame as f32)));
        }
        if s.count() == 0 {
            return None;
        }
        let srt = s.emitter_srt(&Mat4::IDENTITY);
        let streams = s.streams();
        let dynamic = s.dynamic_block(&srt, &s.emitter_rt(&Mat4::IDENTITY), [1.0; 4]);
        let statics = static_block::static_block(&fixed);
        let custom2 = render::default_custom2();
        let custom0 = static_block::custom_block(&raw);
        let rs = &parsed.render_state;
        let draw = render::EmitterDraw {
            program: parsed.shader_references.shader_index as usize,
            state: render::RenderState {
                output: render::Output::Colour,
                blend: render::Blend::from_type(rs.is_blend_enable, rs.blend_type),
                depth_test: false,
                depth_write: false,
                depth_func: rs.depth_func,
                display_side: 0,
            },
            static_block: &statics,
            dynamic_block: &dynamic,
            custom0_block: custom0.as_deref(),
            user_block: None,
            custom2_block: Some(&custom2),
            streams: &streams,
            count: s.count() as u32,
            textures: [Some(&self.white); 3],
            samplers: [Some(&self.sampler); 3],
            mesh: None,
        };
        let renderer = &self.renderers[usize::from(wide)];
        renderer.warm(draw.program, draw.state, false);
        let (_, out) = renderer.probe_vertex(&self.queue, &draw, 8)?;
        // The last row of each vertex is the probe's own.
        let shown = s.count().min(8) as usize;
        let per = out.len() / (shown * 4);
        let time = f32::from_le_bytes(dynamic[0x20..0x24].try_into().unwrap());
        let through = (0..shown)
            .map(|i| {
                let (born, life) = (
                    streams["sysLocalVecAttr"][i][3],
                    streams["sysLocalPosAttr"][i][3],
                );
                (time - born) / life.trunc()
            })
            .collect();
        let rows = out.chunks(per).flat_map(|v| v[..per - 1].to_vec());
        Some((rows.collect(), through))
    }

    fn emitters(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.sets
            .iter()
            .enumerate()
            .flat_map(|(s, set)| (0..set.emitters.len()).map(move |e| (s, e)))
    }

    fn program(&self, (set, emitter): (usize, usize)) -> usize {
        self.ptcl.emitter_list.emitter_sets[set].emitters[emitter]
            .data
            .shader_references
            .shader_index as usize
    }
}

/// Each table's colour type byte, constant, key count and keys in the emitter's data.
const TABLES: [(Option<(usize, usize, usize)>, usize, usize); 5] = [
    (Some((0x99c, 0x9a0, 12)), 0x60, 0x3c0),
    (Some((0x99e, 0x9ac, 4)), 0x64, 0x440),
    (Some((0x99d, 0x9b0, 12)), 0x68, 0x4c0),
    (Some((0x99f, 0x9bc, 4)), 0x6c, 0x540),
    (None, 0x70, 0x600),
];

/// Constants as tables of one key, which is how a part is stored once a key is added to it.
fn as_keys(data: &mut [u8]) {
    for (constant, count, keys) in TABLES {
        let Some((kind, value, size)) = constant else {
            continue;
        };
        if data[kind] == 0 {
            data[kind] = 2;
            data[count..count + 4].copy_from_slice(&1i32.to_le_bytes());
            data.copy_within(value..value + size, keys);
        }
    }
}

fn close(a: &[[f32; 4]], b: &[[f32; 4]]) -> Result<(), String> {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        for (x, y) in a.iter().zip(b) {
            let same = (x.is_nan() && y.is_nan())
                || x == y
                || (x - y).abs() <= 1e-3 * (1.0 + x.abs().max(y.abs()));
            if !same {
                return Err(format!("row {i}: {a:?} {b:?}"));
            }
        }
    }
    Ok(())
}

#[test]
fn a_widened_shader_draws_the_same_from_the_same_keys() {
    let files =
        std::env::var("VFX_KEYS_FILES").unwrap_or_else(|_| "fighter/mario/ef_mario.eff".into());
    for file in files.split(',') {
        let Some(game) = game(file) else {
            eprintln!("skipped");
            return;
        };
        let mut programs = std::collections::BTreeSet::new();
        let mut bad = Vec::new();
        let (mut checked, mut animated) = (0, 0);
        for at in game.emitters() {
            let program = game.program(at);
            if program >= game.widened.len() || !programs.insert(program) {
                continue;
            }
            for steps in [1, 4, 11] {
                let a = game.run(at, steps, false, |_| {});
                let b = game.run(at, steps, true, |data| as_keys(data));
                let (Some((a, _)), Some((b, _))) = (a, b) else {
                    continue;
                };
                checked += 1;
                animated += usize::from(game.read[program].iter().any(|n| *n > 1));
                if let Err(why) = close(&a, &b) {
                    bad.push(format!("{at:?} program {program} after {steps}: {why}"));
                }
            }
        }
        eprintln!(
            "{file}: {checked} compared, {animated} of them animated, {} differ",
            bad.len()
        );
        assert!(checked > 50 && animated > 50);
        assert!(bad.is_empty(), "{}", bad[..bad.len().min(12)].join("\n"));
    }
}

/// Every part that can be given keys follows five it was not made with, the opacity rising and
/// falling twice over each particle's life.
#[test]
fn a_widened_shader_follows_keys_it_was_not_made_with() {
    const KEYS: [(f32, f32); 5] = [(0.2, 0.0), (1.0, 0.25), (0.2, 0.5), (1.0, 0.75), (0.2, 1.0)];
    let Some(game) = game("fighter/mario/ef_mario.eff") else {
        eprintln!("skipped");
        return;
    };
    let alpha = Table::Alpha0 as usize;
    let (Some((kind, _, _)), count, keys) = TABLES[alpha] else {
        unreachable!();
    };
    let follow = |data: &mut Vec<u8>, values: &dyn Fn(usize) -> f32| {
        as_keys(data);
        data[kind] = 2;
        data[count..count + 4].copy_from_slice(&(KEYS.len() as i32).to_le_bytes());
        // It plays once over the life.
        data[0x84..0x88].fill(0);
        for (k, (_, time)) in KEYS.iter().enumerate() {
            let at = keys + 16 * k;
            data[at..at + 4].copy_from_slice(&values(k).to_le_bytes());
            data[at + 12..at + 16].copy_from_slice(&time.to_le_bytes());
        }
    };
    let expected = |t: f32| {
        let span = KEYS.windows(2).find(|w| t <= w[1].1).unwrap_or(&KEYS[3..5]);
        let ((a, at), (b, to)) = (span[0], span[1]);
        a + (b - a) * ((t - at) / (to - at)).clamp(0.0, 1.0)
    };
    let mut programs = std::collections::BTreeSet::new();
    let (mut checked, mut of_one, mut heights) = (0, 0, Vec::new());
    for at in game.emitters() {
        let program = game.program(at);
        let widened = game
            .widened
            .get(program)
            .is_some_and(|t| t.contains(&Table::Alpha0));
        if !widened || !programs.insert(program) {
            continue;
        }
        for steps in [3, 9] {
            let go = |values: &dyn Fn(usize) -> f32| {
                game.run(at, steps, true, |data| follow(data, values))
            };
            let (Some((low, through)), Some((high, _)), Some((keyed, _))) =
                (go(&|_| 0.2), go(&|_| 1.0), go(&|k| KEYS[k].0))
            else {
                continue;
            };
            let per = low.len() / through.len();
            for (i, ((low, high), keyed)) in low.iter().zip(&high).zip(&keyed).enumerate() {
                for c in 0..4 {
                    let reach = high[c] - low[c];
                    // Only what the opacity drives, and a particle still alive.
                    let t = through[i / per];
                    if reach.abs() < 0.05 || !(0.0..=1.0).contains(&t) || !keyed[c].is_finite() {
                        continue;
                    }
                    let got = 0.2 + 0.8 * (keyed[c] - low[c]) / reach;
                    assert!(
                        (got - expected(t)).abs() < 0.02,
                        "{at:?} after {steps}: {got} at {t}, not {}",
                        expected(t)
                    );
                    checked += 1;
                    of_one += usize::from(game.read[program][alpha] == 1);
                    heights.push(got);
                }
            }
        }
    }
    let spread = heights
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), h| (a.min(*h), b.max(*h)));
    eprintln!("{checked} opacities, {of_one} of shaders made for one; {spread:?}");
    assert!(checked > 200 && of_one > 50 && spread.0 < 0.3 && spread.1 > 0.9);
}
