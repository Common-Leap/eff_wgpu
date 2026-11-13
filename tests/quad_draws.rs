//! A quad (billboard) emitter covers pixels when drawn. Point VFX_GAME_EFFECTS at the game's
//! `effect` folder (skipped when unset or when there is no GPU).
//!
//! Quads take every particle stream per instance and nothing per vertex. They were once laid
//! out behind an attribute-less vertex buffer and read zeros, so every billboard collapsed to a
//! point while mesh emitters still drew.
use glam::{Mat4, Vec3};
use vfx_runtime::{bnsh, ptcl, render, sim, static_block, texture, view_block};

#[test]
fn mario_fireball_flame_quads_cover_pixels() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let data =
        std::fs::read(std::path::Path::new(&root).join("fighter/mario/ef_mario.eff")).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let set = eff
        .entry_names
        .iter()
        .zip(&eff.entries)
        .find(|(n, _)| n.as_str() == "MARIO_FB_BULLET_R")
        .map(|(_, e)| e.emitter_set_id as usize - 1)
        .unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    else {
        eprintln!("no GPU; skipped");
        return;
    };
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let size = 256u32;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let shaders = bnsh::ShaderFile::read(
        p.shader_info
            .as_ref()
            .unwrap()
            .binary_data
            .as_ref()
            .unwrap(),
    )
    .unwrap();
    let mut renderer = render::EffectRenderer::new(&device, &queue, &shaders, format, None);
    let eye = Vec3::new(0.0, 0.0, 25.0);
    let rows = |m: Mat4| m.transpose().to_cols_array_2d();
    let cam = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(45f32.to_radians(), 1.0, 1.0, 1000.0);
    renderer.set_view(
        &queue,
        &view_block::view_block(&rows(cam), &rows(proj), eye.into(), 1.0, 1000.0, 1.0),
    );

    // Emitter 0, "fire1": quads with no mesh.
    let raw = &ptcl::emitter_sets(&p.base_bytes)[set].emitters[0];
    let parsed = &p.emitter_list.emitter_sets[set].emitters[0].data;
    assert_eq!(
        u64::from_le_bytes(raw.data[0x8c8..0x8d0].try_into().unwrap()),
        u64::MAX
    );
    let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
    let mut s = sim::EmitterSim::new(parsed, &fixed);
    // fire1 emits by distance: move the set the way the fireball flies (1.4 a frame).
    for frame in 0..6 {
        s.step_at(&Mat4::from_translation(Vec3::new(
            1.4 * frame as f32,
            0.0,
            0.0,
        )));
    }
    assert!(s.count() > 0);
    let ids: Vec<u64> = p
        .texture_info
        .as_ref()
        .unwrap()
        .descriptors
        .iter()
        .map(|d| d.id)
        .collect();
    let sampler = parsed.sampler0.as_ref().unwrap();
    let index = ids.iter().position(|id| *id == sampler.texture_id).unwrap();
    let tex = render::upload_texture(&device, &queue, &texture::decode(p, index).unwrap());
    let smp = render::make_sampler(
        &device,
        sampler.wrap_u.as_u8(),
        sampler.wrap_v.as_u8(),
        sampler.filter,
        sampler.max_lod,
    );
    let srt = s.emitter_srt(&Mat4::IDENTITY);
    let streams = s.streams();
    let dynamic = s.dynamic_block(&srt, &s.emitter_rt(&Mat4::IDENTITY), [1.0; 4]);
    let statics = static_block::static_block(&fixed);
    let custom2 = render::default_custom2();
    let rs = &parsed.render_state;
    let draw = render::EmitterDraw {
        program: parsed.shader_references.shader_index as usize,
        state: render::RenderState {
            blend: render::Blend::from_type(rs.is_blend_enable, rs.blend_type),
            depth_test: false,
            depth_write: false,
            depth_func: rs.depth_func,
            display_side: 0,
        },
        static_block: &statics,
        dynamic_block: &dynamic,
        custom0_block: None,
        custom2_block: Some(&custom2),
        streams: &streams,
        count: s.count() as u32,
        textures: [Some(&tex), None, None],
        samplers: [Some(&smp), None, None],
        mesh: None,
    };
    let prepared = renderer
        .prepare(&queue, &draw)
        .expect("the emitter prepares");
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        renderer.draw(&mut pass, &prepared);
    }
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (size * size * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let pixels = readback.slice(..).get_mapped_range().to_vec();
    let lit = pixels.chunks_exact(4).filter(|p| p[0] > 32).count();
    assert!(lit > 500, "only {lit} pixels lit");
}
