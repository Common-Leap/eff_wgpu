//! cargo run --example render_frame -- ef_x.eff SET FRAME out.png
//! Draws one emitter set FRAME frames after it starts, with the game's shaders, white textures.
use glam::{Mat4, Vec3};
use vfx_runtime::{bnsh, ptcl, render, sim, static_block, view_block};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (path, set, frame, out) = (
        &a[1],
        a[2].parse::<usize>().unwrap(),
        a[3].parse::<u32>().unwrap(),
        &a[4],
    );
    let data = std::fs::read(path).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let ptcl_file = eff.ptcl_file.unwrap();
    let shaders = bnsh::ShaderFile::read(
        ptcl_file
            .shader_info
            .as_ref()
            .unwrap()
            .binary_data
            .as_ref()
            .unwrap(),
    )
    .unwrap();
    let raw_sets = ptcl::emitter_sets(&ptcl_file.base_bytes);
    let parsed = &ptcl_file.emitter_list.emitter_sets[set];
    eprintln!(
        "set {set} {:?}: {} emitters",
        parsed.name,
        parsed.emitters.len()
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .unwrap();
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| eprintln!("wgpu: {e}")));
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let size = 512;
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
    let mut renderer = render::EffectRenderer::new(&device, &queue, &shaders, format, None);

    let eye = Vec3::new(0.0, 10.0, 40.0);
    let cam = Mat4::look_at_rh(eye, Vec3::new(0.0, 8.0, 0.0), Vec3::Y);
    let proj = Mat4::perspective_rh(45f32.to_radians(), 1.0, 1.0, 1000.0);
    let rows = |m: Mat4| m.transpose().to_cols_array_2d();
    renderer.set_view(
        &queue,
        &view_block::view_block(&rows(cam), &rows(proj), eye.into(), 1.0, 1000.0, 1.0),
    );

    let mut fixed = Vec::new();
    for (j, e) in parsed.emitters.iter().enumerate() {
        let raw = &raw_sets[set].emitters[j];
        fixed.push((
            e,
            raw,
            static_block::fixed_emitter_data(raw, &|_| None, None),
        ));
    }
    let mut prepared = Vec::new();
    for (j, (e, raw, fixed_data)) in fixed.iter().enumerate() {
        let mut s = sim::EmitterSim::new(&e.data, fixed_data);
        for _ in 0..frame {
            s.step();
        }
        let srt = s.emitter_srt(&Mat4::IDENTITY);
        let mut streams = s.streams(&srt);
        // PROBE_VERTEX="x,y,u,v,alpha": one mesh vertex's inputs, for probing mesh programs.
        if let Ok(v) = std::env::var("PROBE_VERTEX") {
            let f: Vec<f32> = v.split(',').map(|x| x.parse().unwrap()).collect();
            let n = s.count();
            streams.insert("sysPosAttr", vec![[f[0], f[1], 0.0, 1.0]; n]);
            streams.insert("sysTexCoordAttr", vec![[f[2], f[3], 0.0, 0.0]; n]);
            streams.insert("sysNormalAttr", vec![[0.0, 0.0, 1.0, 0.0]; n]);
            streams.insert("sysVertexColor0Attr", vec![[1.0, 1.0, 1.0, f[4]]; n]);
        }
        let dynamic = s.dynamic_block(&srt, &s.emitter_rt(&Mat4::IDENTITY), [1.0; 4]);
        let statics = static_block::static_block(fixed_data);
        let rs = &e.data.render_state;
        let program = e.data.shader_references.shader_index as usize;
        if let Some(err) = renderer.program_error(program) {
            eprintln!("emitter {j}: program {program}: {err}");
            continue;
        }
        let custom = static_block::custom_block(raw);
        let custom2 = render::default_custom2();
        let draw = render::EmitterDraw {
            program,
            state: render::RenderState {
                blend: render::Blend::from_type(rs.is_blend_enable, rs.blend_type),
                depth_test: false,
                depth_write: false,
                depth_func: rs.depth_func,
                display_side: if std::env::var_os("NOCULL").is_some() {
                    0
                } else {
                    rs.display_side
                },
            },
            static_block: &statics,
            dynamic_block: &dynamic,
            custom0_block: custom.as_deref(),
            custom2_block: Some(&custom2),
            streams: &streams,
            count: s.count() as u32,
            textures: [None; 3],
            samplers: [None; 3],
            mesh: None,
        };
        eprintln!(
            "emitter {j}: program {program}, {} particles, blend {:?}",
            s.count(),
            draw.state.blend
        );
        if std::env::var_os("PROBE").is_some() {
            if let Some((outs, values)) = renderer.probe_vertex(&queue, &draw, 1) {
                eprintln!("  vertex outputs {outs:?}");
                for (k, v) in values.iter().enumerate() {
                    eprintln!("  [{k}] {v:?}");
                }
            }
            for (name, v) in &streams {
                eprintln!("  {name}: {:?}", v.first());
            }
        }
        if let Some(p) = renderer.prepare(&queue, &draw) {
            prepared.push(p);
        }
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.25,
                        g: 0.25,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        for p in &prepared {
            renderer.draw(&mut pass, p);
        }
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
    let file = std::fs::File::create(out).unwrap();
    let mut enc = png::Encoder::new(file, size, size);
    enc.set_color(png::ColorType::Rgba);
    enc.write_header()
        .unwrap()
        .write_image_data(&pixels)
        .unwrap();
}
