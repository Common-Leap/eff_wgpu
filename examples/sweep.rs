//! cargo run --release --example sweep -- EFFECT_ROOT [FILTER] : prepare and draw every emitter
//! set of every .eff at a few frames, reporting wgpu errors and panics per set.
use glam::{Mat4, Vec3};
use std::sync::{Arc, Mutex};
use vfx_runtime::{bnsh, mesh, ptcl, render, sim, static_block, texture, view_block};

fn files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            files(&p, out);
        } else if p.extension().is_some_and(|x| x == "eff") {
            out.push(p);
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut list = Vec::new();
    files(std::path::Path::new(&a[1]), &mut list);
    list.sort();
    if let Some(f) = a.get(2) {
        list.retain(|p| p.to_string_lossy().contains(f.as_str()));
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default())).unwrap();
    let errors: Arc<Mutex<Vec<String>>> = Arc::default();
    let sink = errors.clone();
    device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
        sink.lock().unwrap().push(e.to_string())
    }));
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let (mut sets_total, mut bad) = (0, 0);
    for path in &list {
        let Ok(data) = std::fs::read(path) else {
            continue;
        };
        let Ok(eff) = effect_library::namco_file::NamcoEffectFile::load(&data) else {
            println!("{}: does not load", path.display());
            continue;
        };
        let Some(p) = eff.ptcl_file.as_ref() else {
            continue;
        };
        let Some(bin) = p.shader_info.as_ref().and_then(|s| s.binary_data.as_ref()) else {
            continue;
        };
        let Ok(shaders) = bnsh::ShaderFile::read(bin) else {
            println!("{}: shaders do not read", path.display());
            continue;
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut renderer = render::EffectRenderer::new(&device, &queue, &shaders, format, None);
            let eye = Vec3::new(0.0, 10.0, 40.0);
            let rows = |m: Mat4| m.transpose().to_cols_array_2d();
            renderer.set_view(
                &queue,
                &view_block::view_block(
                    &rows(Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y)),
                    &rows(Mat4::perspective_rh(0.8, 1.0, 1.0, 1000.0)),
                    eye.into(),
                    1.0,
                    1000.0,
                    1.0,
                ),
            );
            let meshes = mesh::meshes(p);
            let ids: Vec<u64> = p
                .texture_info
                .as_ref()
                .map(|i| i.descriptors.iter().map(|d| d.id).collect())
                .unwrap_or_default();
            let mut tex_cache: std::collections::HashMap<usize, Option<render::GpuTexture>> =
                Default::default();
            let raw = ptcl::emitter_sets(&p.base_bytes);
            let mut out = Vec::new();
            for (si, set) in p.emitter_list.emitter_sets.iter().enumerate() {
                let Some(raw_set) = raw.get(si) else { continue };
                errors.lock().unwrap().clear();
                let set_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    for frame in [1u32, 10, 40] {
                        let mut prepared = Vec::new();
                        let mut keep = Vec::new();
                        for (e, r) in set.emitters.iter().zip(&raw_set.emitters) {
                            let fixed = static_block::fixed_emitter_data(r, &|_| None, None);
                            let mesh_id =
                                u64::from_le_bytes(fixed[0x8c8..0x8d0].try_into().unwrap());
                            let mesh = (mesh_id != u64::MAX)
                                .then(|| meshes.get(&mesh_id))
                                .flatten();
                            if mesh_id != u64::MAX && mesh.is_none() {
                                continue;
                            }
                            let mut s = sim::EmitterSim::new(&e.data, &fixed);
                            for _ in 0..frame {
                                s.step();
                            }
                            if s.count() == 0 {
                                continue;
                            }
                            let srt = s.emitter_srt(&Mat4::IDENTITY);
                            let streams = s.streams();
                            let dynamic =
                                s.dynamic_block(&srt, &s.emitter_rt(&Mat4::IDENTITY), [1.0; 4]);
                            let statics = static_block::static_block(&fixed);
                            let custom0 = static_block::custom_block(r);
                            let custom2 = render::default_custom2();
                            let slots = [&e.data.sampler0, &e.data.sampler1, &e.data.sampler2];
                            for s in slots.iter().copied().flatten() {
                                if let Some(i) = ids.iter().position(|id| *id == s.texture_id) {
                                    tex_cache.entry(i).or_insert_with(|| {
                                        texture::decode(p, i)
                                            .map(|t| render::upload_texture(&device, &queue, &t))
                                    });
                                }
                            }
                            let samplers: [Option<wgpu::Sampler>; 3] = std::array::from_fn(|k| {
                                slots[k].as_ref().map(|s| {
                                    render::make_sampler(
                                        &device,
                                        s.wrap_u.as_u8(),
                                        s.wrap_v.as_u8(),
                                        s.filter,
                                        s.max_lod,
                                    )
                                })
                            });
                            let views: [Option<&render::GpuTexture>; 3] =
                                std::array::from_fn(|k| {
                                    let id = slots[k].as_ref()?.texture_id;
                                    tex_cache.get(&ids.iter().position(|x| *x == id)?)?.as_ref()
                                });
                            let rs = &e.data.render_state;
                            let draw = render::EmitterDraw {
                                program: e.data.shader_references.shader_index.max(0) as usize,
                                state: render::RenderState {
                                    blend: render::Blend::from_type(
                                        rs.is_blend_enable,
                                        rs.blend_type,
                                    ),
                                    depth_test: false,
                                    depth_write: false,
                                    depth_func: rs.depth_func,
                                    display_side: rs.display_side,
                                },
                                static_block: &statics,
                                dynamic_block: &dynamic,
                                custom0_block: custom0.as_deref(),
                                custom2_block: Some(&custom2),
                                streams: &streams,
                                count: s.count() as u32,
                                textures: views,
                                samplers: std::array::from_fn(|k| samplers[k].as_ref()),
                                mesh,
                            };
                            if let Some(pd) = renderer.prepare(&queue, &draw) {
                                prepared.push(pd);
                            }
                            keep.push(samplers);
                        }
                        let mut enc = device.create_command_encoder(&Default::default());
                        {
                            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                            for pd in &prepared {
                                renderer.draw(&mut pass, pd);
                            }
                        }
                        queue.submit([enc.finish()]);
                    }
                    device.poll(wgpu::PollType::wait_indefinitely()).ok();
                }));
                let mut errs = errors.lock().unwrap().clone();
                errs.extend(renderer.take_failures());
                match set_result {
                    Err(e) => out.push(format!(
                        "set {si} {}: PANIC {}",
                        set.name,
                        e.downcast_ref::<String>()
                            .cloned()
                            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_default()
                    )),
                    Ok(()) if !errs.is_empty() => out.push(format!(
                        "set {si} {}: {}",
                        set.name,
                        errs[0].lines().take(12).collect::<Vec<_>>().join(" | ")
                    )),
                    _ => {}
                }
            }
            (p.emitter_list.emitter_sets.len(), out)
        }));
        match result {
            Err(_) => {
                println!("{}: PANIC outside a set", path.display());
                bad += 1;
            }
            Ok((n, out)) => {
                sets_total += n;
                bad += out.len();
                for o in out {
                    println!("{}: {o}", path.file_name().unwrap().to_string_lossy());
                }
            }
        }
    }
    println!("{} files, {sets_total} sets, {bad} bad", list.len());
}
