//! Shaders that read the frame copy (slot 3) or depth (slot 4) build and draw. Point
//! VFX_GAME_EFFECTS at the game's `effect` folder (skipped when unset or without a GPU).
//!
//! The layout once stopped at the emitter's three samplers, so these ~2,900 shader stages failed
//! pipeline creation and wgpu's default handler aborted the editor on Mario's F.L.U.D.D. hit
//! and Final Smash moves.
use glam::Mat4;
use std::sync::{Arc, Mutex};
use vfx_runtime::{bnsh, ptcl, render, sim, static_block};

#[test]
fn frame_and_depth_reading_emitters_prepare_without_errors() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let data =
        std::fs::read(std::path::Path::new(&root).join("fighter/mario/ef_mario.eff")).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("no GPU; skipped");
        return;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let errors: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = errors.clone();
    device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
        sink.lock().unwrap().push(e.to_string())
    }));
    let shaders = bnsh::ShaderFile::read(
        p.shader_info
            .as_ref()
            .unwrap()
            .binary_data
            .as_ref()
            .unwrap(),
    )
    .unwrap();
    let mut renderer = render::EffectRenderer::new(
        &device,
        &queue,
        &shaders,
        wgpu::TextureFormat::Rgba8Unorm,
        None,
    );
    let raw = ptcl::emitter_sets(&p.base_bytes);
    let meshes = vfx_runtime::mesh::meshes(p);
    let mut prepared = 0;
    for name in ["P_MarioPumpHit2", "P_MarioFinalBg"] {
        let index = p
            .emitter_list
            .emitter_sets
            .iter()
            .position(|s| s.name == name)
            .unwrap();
        for (e, r) in p.emitter_list.emitter_sets[index]
            .emitters
            .iter()
            .zip(&raw[index].emitters)
        {
            let fixed = static_block::fixed_emitter_data(r, &|_| None, None);
            let mesh_id = u64::from_le_bytes(fixed[0x8c8..0x8d0].try_into().unwrap());
            let mesh = (mesh_id != u64::MAX)
                .then(|| meshes.get(&mesh_id))
                .flatten();
            for frames in [1, 10, 40] {
                let mut s = sim::EmitterSim::new(&e.data, &fixed);
                for _ in 0..frames {
                    s.step();
                }
                if s.count() == 0 || (mesh_id != u64::MAX && mesh.is_none()) {
                    continue;
                }
                let srt = s.emitter_srt(&Mat4::IDENTITY);
                let streams = s.streams();
                let dynamic = s.dynamic_block(&srt, &s.emitter_rt(&Mat4::IDENTITY), [1.0; 4]);
                let statics = static_block::static_block(&fixed);
                let custom2 = render::default_custom2();
                let draw = render::EmitterDraw {
                    program: e.data.shader_references.shader_index.max(0) as usize,
                    state: render::RenderState {
                        blend: render::Blend::Normal,
                        depth_test: false,
                        depth_write: false,
                        depth_func: 0,
                        display_side: 0,
                    },
                    static_block: &statics,
                    dynamic_block: &dynamic,
                    custom0_block: None,
                    custom2_block: Some(&custom2),
                    streams: &streams,
                    count: s.count() as u32,
                    textures: [None; 3],
                    samplers: [None; 3],
                    mesh,
                };
                if renderer.prepare(&queue, &draw).is_some() {
                    prepared += 1;
                }
            }
        }
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    assert!(prepared > 0, "nothing prepared");
    assert_eq!(renderer.take_failures(), Vec::<String>::new());
    assert_eq!(*errors.lock().unwrap(), Vec::<String>::new());
}
