//! Shaders that read the scene behind them: the frame copy (refraction) and the depth copy
//! (floor decals). Point VFX_GAME_EFFECTS at the game's `effect` folder (skipped when unset or
//! when there is no GPU).
use eff_wgpu::{bnsh, mesh, ptcl, render, scene, sim, static_block, texture, view_block};
use glam::{Mat4, Vec3};

const SIZE: u32 = 128;
const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const NEAR: f32 = 1.0;
const FAR: f32 = 1000.0;
/// Down onto the stage floor, at the origin, from the front and above.
const EYE: Vec3 = Vec3::new(0.0, 40.0, 60.0);

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some(Gpu { device, queue })
}

fn camera() -> (Mat4, Mat4) {
    (
        glam::camera::rh::view::look_at_mat4(EYE, Vec3::ZERO, Vec3::Y),
        glam::camera::rh::proj::directx::perspective(45f32.to_radians(), 1.0, NEAR, FAR),
    )
}

fn texture(gpu: &Gpu, format: wgpu::TextureFormat, samples: u32) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// The scene as Visionary hands it over: colour already resolved, depth with 4 samples.
/// `floor` draws a flat floor at that height into the depth (none: nothing behind).
/// `colour(x)` is the scene colour of column `x`.
fn scene_copy(gpu: &Gpu, floor: Option<f32>, colour: impl Fn(u32) -> [u8; 4]) -> scene::SceneCopy {
    let depth = texture(gpu, wgpu::TextureFormat::Depth32Float, 4);
    let depth_view = depth.create_view(&Default::default());
    let colour_texture = texture(gpu, TARGET, 1);
    let pixels: Vec<u8> = (0..SIZE * SIZE).flat_map(|i| colour(i % SIZE)).collect();
    gpu.queue.write_texture(
        colour_texture.as_image_copy(),
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SIZE * 4),
            rows_per_image: None,
        },
        colour_texture.size(),
    );
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(
                    "@group(0) @binding(0) var<uniform> m: mat4x4<f32>;
                    @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                        var c = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
                        return m * vec4(c[i].x * 500.0, 0.0, c[i].y * 500.0, 1.0);
                    }"
                    .into(),
                ),
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 4,
                    ..Default::default()
                },
                fragment: None,
                multiview_mask: None,
                cache: None,
            });
        use wgpu::util::DeviceExt;
        let (view, proj) = camera();
        let floor_matrix = proj * view * Mat4::from_translation(Vec3::Y * floor.unwrap_or(0.0));
        let uniform = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck_floats(&floor_matrix.to_cols_array()),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        if floor.is_some() {
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..4, 0..1);
        }
    }
    let mut copy = scene::SceneCopy::new(&gpu.device, TARGET, 4);
    copy.resize(&gpu.device, SIZE, SIZE);
    copy.copy(
        &gpu.device,
        &mut encoder,
        &colour_texture.create_view(&Default::default()),
        &depth_view,
    );
    gpu.queue.submit([encoder.finish()]);
    copy
}

fn bytemuck_floats(v: &[f32]) -> &[u8] {
    // SAFETY: f32 is plain data.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// One emitter of a game effect file, stepped `frames` frames at the origin.
struct Effect {
    data: Vec<u8>,
    file: String,
}

impl Effect {
    fn load(root: &std::ffi::OsStr, file: &str) -> Self {
        Self {
            data: std::fs::read(std::path::Path::new(root).join(file)).unwrap(),
            file: file.into(),
        }
    }

    /// `emitter` of `set` after `frames` frames, ready to draw over `scene` (or no scene copy)
    /// with `custom2`, and what its pixels are.
    fn prepare(
        &self,
        gpu: &Gpu,
        set: &str,
        emitter: &str,
        frames: usize,
        scene: Option<&scene::SceneCopy>,
        custom2: &[u8],
    ) -> (render::EffectRenderer, render::PreparedDraw, render::Output) {
        let eff = effect_library::namco_file::NamcoEffectFile::load(&self.data).unwrap();
        let p = eff.ptcl_file.as_ref().unwrap();
        let set_index = p
            .emitter_list
            .emitter_sets
            .iter()
            .position(|s| s.name == set)
            .unwrap_or_else(|| panic!("{set} in {}", self.file));
        let parsed_set = &p.emitter_list.emitter_sets[set_index];
        let index = parsed_set
            .emitters
            .iter()
            .position(|e| e.data.name.as_deref() == Some(emitter))
            .unwrap();
        let parsed = &parsed_set.emitters[index].data;
        let raw = &ptcl::emitter_sets(&p.base_bytes)[set_index].emitters[index];
        let fixed = static_block::fixed_emitter_data(raw, &|_| None, None);
        let shaders = bnsh::ShaderFile::read(
            p.shader_info
                .as_ref()
                .unwrap()
                .binary_data
                .as_ref()
                .unwrap(),
        )
        .unwrap();
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let renderer = render::EffectRenderer::new(
            &gpu.device,
            &gpu.queue,
            &shaders,
            TARGET,
            Some(depth_format),
        );
        let (view, proj) = camera();
        let rows = |m: Mat4| m.transpose().to_cols_array_2d();
        renderer.set_view(
            &gpu.queue,
            &view_block::view_block(&rows(view), &rows(proj), EYE.into(), NEAR, FAR, 1.0),
        );
        if let Some(copy) = scene {
            renderer.set_scene(
                Some(copy.colour_view().clone()),
                Some(copy.depth_view().clone()),
            );
        }
        let mut s = sim::EmitterSim::new(parsed, &fixed);
        for _ in 0..frames {
            s.step_at(&Mat4::IDENTITY);
        }
        assert!(s.count() > 0, "{set}/{emitter} has no particles");
        let ids: Vec<u64> = p
            .texture_info
            .as_ref()
            .unwrap()
            .descriptors
            .iter()
            .map(|d| d.id)
            .collect();
        let samplers = [
            parsed.sampler0.as_ref(),
            parsed.sampler1.as_ref(),
            parsed.sampler2.as_ref(),
        ];
        let textures: Vec<Option<render::GpuTexture>> = samplers
            .iter()
            .map(|s| {
                let id = (*s)?.texture_id;
                let index = ids.iter().position(|i| *i == id)?;
                Some(render::upload_texture(
                    &gpu.device,
                    &gpu.queue,
                    &texture::decode(p, index)?,
                ))
            })
            .collect();
        let gpu_samplers: Vec<Option<wgpu::Sampler>> = samplers
            .iter()
            .map(|s| {
                let s = (*s)?;
                Some(render::make_sampler(
                    &gpu.device,
                    s.wrap_u.as_u8(),
                    s.wrap_v.as_u8(),
                    s.filter,
                    s.max_lod,
                ))
            })
            .collect();
        let meshes = mesh::meshes(p);
        let mesh_id = u64::from_le_bytes(fixed[0x8c8..0x8d0].try_into().unwrap());
        let mesh = (mesh_id != u64::MAX).then(|| &meshes[&mesh_id]);
        let srt = s.emitter_srt(&Mat4::IDENTITY);
        let streams = s.streams();
        let dynamic = s.dynamic_block(&srt, &s.emitter_rt(&Mat4::IDENTITY), [1.0; 4]);
        let statics = static_block::static_block(&fixed);
        let custom0 = static_block::custom_block(raw);
        let rs = &parsed.render_state;
        let output = render::Output::of_draw_path(parsed.emitter_info.draw_path as u8).unwrap();
        let draw = render::EmitterDraw {
            program: parsed.shader_references.shader_index as usize,
            state: render::RenderState {
                output,
                blend: render::Blend::from_type(rs.is_blend_enable, rs.blend_type),
                depth_test: rs.is_depth_test,
                depth_write: rs.is_depth_mask,
                depth_func: rs.depth_func,
                display_side: rs.display_side,
            },
            static_block: &statics,
            dynamic_block: &dynamic,
            custom0_block: custom0.as_deref(),
            user_block: None,
            custom2_block: Some(custom2),
            streams: &streams,
            count: s.count() as u32,
            textures: std::array::from_fn(|i| textures[i].as_ref()),
            samplers: std::array::from_fn(|i| gpu_samplers[i].as_ref()),
            mesh,
        };
        let prepared = renderer.prepare(&gpu.queue, &draw).expect("prepares");
        assert!(renderer.take_failures().is_empty());
        (renderer, prepared, output)
    }

    /// Draw `emitter` of `set` after `frames` frames over `scene` (or no scene copy), with
    /// `custom2`, and return the target's pixels.
    fn draw(
        &self,
        gpu: &Gpu,
        set: &str,
        emitter: &str,
        frames: usize,
        scene: Option<&scene::SceneCopy>,
        custom2: &[u8],
    ) -> Vec<u8> {
        let (renderer, prepared, output) = self.prepare(gpu, set, emitter, frames, scene, custom2);
        assert_eq!(output, render::Output::Colour);
        let depth_format = wgpu::TextureFormat::Depth32Float;
        let target = texture(gpu, TARGET, 1);
        let target_view = target.create_view(&Default::default());
        // The effect's own depth test runs against a clear depth buffer: what hides it is the
        // shader's own read of the scene copy, which is under test.
        let depth = texture(gpu, depth_format, 1);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth.create_view(&Default::default()),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            renderer.draw(&mut pass, &prepared);
        }
        let pixels = read(gpu, &target, &mut encoder, 4);
        gpu.queue.submit([encoder.finish()]);
        pixels()
    }
}

/// Queue a read of `texture` (`bytes` a pixel) and return what waits for and maps it.
fn read<'a>(
    gpu: &'a Gpu,
    texture: &wgpu::Texture,
    encoder: &mut wgpu::CommandEncoder,
    bytes: u32,
) -> impl FnOnce() -> Vec<u8> + 'a {
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (SIZE * SIZE * bytes) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * bytes),
                rows_per_image: None,
            },
        },
        texture.size(),
    );
    move || {
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let out = buffer
            .slice(..)
            .get_mapped_range()
            .expect("mapped")
            .to_vec();
        out
    }
}

/// Pixels the draw covered (any alpha left in the cleared target counts, as does colour).
fn covered(pixels: &[u8]) -> usize {
    pixels
        .chunks_exact(4)
        .filter(|p| p[0] > 8 || p[1] > 8 || p[2] > 8)
        .count()
}

/// Kazuya's ground crack (Heaven's Impact) keeps a pixel only where the scene's depth lies
/// within a set distance behind it. With no depth copy (1.0, nothing behind) the preview drew
/// none of it; over a floor at its own height it draws, and over a floor far below it does not.
#[test]
fn ground_cracks_draw_on_the_floor_behind_them() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let Some(gpu) = gpu() else {
        eprintln!("no GPU; skipped");
        return;
    };
    let effect = Effect::load(&root, "fighter/demon/ef_demon.eff");
    let custom2 = render::default_custom2();
    let grey = |_| [90, 90, 90, 255];
    let draw = |scene: Option<&scene::SceneCopy>| {
        covered(&effect.draw(
            &gpu,
            "P_DemonHeavensImpact",
            "groundCrack2",
            10,
            scene,
            &custom2,
        ))
    };
    let on_floor = draw(Some(&scene_copy(&gpu, Some(0.0), grey)));
    let nothing_behind = draw(Some(&scene_copy(&gpu, None, grey)));
    let no_copy = draw(None);
    let floor_far_below = draw(Some(&scene_copy(&gpu, Some(-30.0), grey)));
    eprintln!(
        "on floor {on_floor}, nothing behind {nothing_behind}, no copy {no_copy}, far below {floor_far_below}"
    );
    assert!(on_floor > 300, "the crack covers only {on_floor} pixels");
    assert_eq!(nothing_behind, 0);
    assert_eq!(no_copy, 0);
    assert_eq!(floor_far_below, 0);
}

/// Zelda's Din's Fire bomb bends the scene behind it: its pixels are the scene's own colour
/// from about where they sit. The scene here is green but for a red first column. Without row
/// 5 of the per-draw block (the screen x scale) every read landed on that column.
#[test]
fn refraction_reads_the_scene_behind_itself() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let Some(gpu) = gpu() else {
        eprintln!("no GPU; skipped");
        return;
    };
    let effect = Effect::load(&root, "fighter/zelda/ef_zelda.eff");
    let copy = scene_copy(&gpu, Some(-30.0), |x| {
        if x < 4 {
            [255, 0, 0, 255]
        } else {
            [0, 255, 0, 255]
        }
    });
    let tally = |custom2: &[u8]| {
        let pixels = effect.draw(
            &gpu,
            "P_ZeldaDinBomb",
            "refraction",
            6,
            Some(&copy),
            custom2,
        );
        // Away from the red columns themselves, a pixel shows red only if it read them.
        let away = || {
            pixels
                .chunks_exact(4)
                .enumerate()
                .filter(|(i, _)| *i as u32 % SIZE >= 8)
                .map(|(_, p)| p)
        };
        let green = away().filter(|p| p[1] > 16 && p[1] > p[0]).count();
        let red = away().filter(|p| p[0] > 16 && p[0] > p[1]).count();
        (green, red)
    };
    let (green, red) = tally(&render::default_custom2());
    eprintln!("with row 5: green {green}, red {red}");
    assert!(green > 300, "only {green} pixels show the scene behind");
    assert_eq!(red, 0);
    // The block as the capture read it, 0x50 bytes: row 5 reads zero.
    let (green, red) = tally(&render::default_custom2()[..0x50]);
    eprintln!("without row 5: green {green}, red {red}");
    assert!(red > 300 && green == 0, "green {green}, red {red}");
}

/// Draw `picture` into a scene as the models do: every sample of `color`, resolved (or straight
/// into `resolved` with one sample a pixel).
fn draw_picture(
    gpu: &Gpu,
    encoder: &mut wgpu::CommandEncoder,
    picture: &wgpu::Texture,
    samples: u32,
    color: &wgpu::TextureView,
    resolved: &wgpu::TextureView,
) {
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
                    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
                }
                @group(0) @binding(0) var picture: texture_2d<f32>;
                @fragment fn fs(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
                    return textureLoad(picture, vec2<i32>(at.xy), 0);
                }"
                .into(),
            ),
        });
    let pipeline = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(TARGET.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&picture.create_view(&Default::default())),
        }],
    });
    let resolves = samples > 1;
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: if resolves { color } else { resolved },
            resolve_target: resolves.then_some(resolved),
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.draw(0..3, 0..1);
}

/// A picture `column` colours, bent by what `draw` draws as offsets, with `samples` a pixel.
fn bent(
    gpu: &Gpu,
    samples: u32,
    column: impl Fn(u32) -> [u8; 4],
    draw: impl FnOnce(&mut wgpu::RenderPass<'_>),
) -> Vec<u8> {
    let (color, resolve) = (texture(gpu, TARGET, samples), texture(gpu, TARGET, 1));
    let depth = texture(gpu, wgpu::TextureFormat::Depth32Float, samples);
    let [color, resolved, depth] =
        [&color, &resolve, &depth].map(|made| made.create_view(&Default::default()));
    let picture = texture(gpu, TARGET, 1);
    let pixels: Vec<u8> = (0..SIZE * SIZE).flat_map(|i| column(i % SIZE)).collect();
    gpu.queue.write_texture(
        picture.as_image_copy(),
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SIZE * 4),
            rows_per_image: None,
        },
        picture.size(),
    );
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    draw_picture(gpu, &mut encoder, &picture, samples, &color, &resolved);
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    });
    let mut distortion = scene::Distortion::new(&gpu.device, TARGET, samples);
    assert!(
        distortion.begin(&mut encoder, &depth).is_none(),
        "not fitted"
    );
    distortion.fit(&gpu.device, SIZE, SIZE);
    draw(&mut distortion.begin(&mut encoder, &depth).unwrap());
    distortion.bend(&gpu.device, &mut encoder, &color, &resolved);
    let pixels = read(gpu, &resolve, &mut encoder, 4);
    gpu.queue.submit([encoder.finish()]);
    pixels()
}

/// Every column a different red: a moved pixel says which column it came from.
fn ramp(x: u32) -> [u8; 4] {
    [(x * 2) as u8, 100, 0, 255]
}

/// The game's final pass reads the scene at each pixel's UV plus its offset.
#[test]
fn the_picture_moves_by_the_offsets_drawn() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU; skipped");
        return;
    };
    // A quarter of the screen to the right, over the left half.
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                    var c = array(vec2(-1.0, -1.0), vec2(0.0, -1.0), vec2(-1.0, 1.0), vec2(0.0, 1.0));
                    return vec4(c[i], 0.0, 1.0);
                }
                @fragment fn fs() -> @location(0) vec4<f32> { return vec4(0.25, 0.0, 0.0, 1.0); }"
                    .into(),
            ),
        });
    for samples in [1, 4] {
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(scene::OFFSETS_FORMAT.into())],
                }),
                multiview_mask: None,
                cache: None,
            });
        let moved = bent(&gpu, samples, ramp, |pass| {
            pass.set_pipeline(&pipeline);
            pass.draw(0..4, 0..1);
        });
        let still = bent(&gpu, samples, ramp, |_| {});
        for (i, (moved, still)) in moved.chunks(4).zip(still.chunks(4)).enumerate() {
            let x = i as u32 % SIZE;
            assert_eq!(still, ramp(x), "no offsets, {samples} samples");
            let from = if x < SIZE / 2 { x + SIZE / 4 } else { x };
            assert_eq!(moved, ramp(from), "column {x}, {samples} samples");
        }
    }
}

/// Emitters on the game's distortion draw path write offsets, not colours: drawn as colour they
/// were black sheets (Ryu's final smash wave, Sephiroth's down throw). They move the picture,
/// and leave nothing in it that was not there.
#[test]
fn distortion_emitters_move_the_picture_and_draw_nothing_on_it() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    let Some(gpu) = gpu() else {
        eprintln!("no GPU; skipped");
        return;
    };
    for (file, set, emitter) in [
        (
            "fighter/ryu/ef_ryu.eff",
            "P_RyuFinalShinkuhadoBullet2",
            "wind_circle1_rif",
        ),
        (
            "fighter/edge/ef_edge.eff",
            "P_EdgeThrowLwGravity",
            "line2_rf",
        ),
    ] {
        let effect = Effect::load(&root, file);
        let (renderer, prepared, output) =
            effect.prepare(&gpu, set, emitter, 6, None, &render::default_custom2());
        assert_eq!(output, render::Output::Offsets, "{emitter}");
        // Every other column bright: the least move shows.
        let stripes = |x: u32| [if x % 2 == 0 { 0 } else { 255 }, 100, 0, 255];
        let moved = bent(&gpu, 1, stripes, |pass| renderer.draw(pass, &prepared));
        let mut changed = 0;
        for (i, pixel) in moved.chunks(4).enumerate() {
            assert_eq!(pixel[1..], [100, 0, 255], "{emitter} drew a colour");
            changed += usize::from(pixel[0] != stripes(i as u32 % SIZE)[0]);
        }
        eprintln!("{emitter}: {changed} pixels moved");
        assert!(changed > 30, "{emitter} moved {changed} pixels");
    }
}

/// Every emitter of the game's custom shader 5 is on a distortion path and every one of custom
/// shader 6 on the motion blur's, and no emitter of either draws a colour.
#[test]
fn the_draw_path_says_what_an_emitter_writes() {
    let Some(root) = std::env::var_os("VFX_GAME_EFFECTS") else {
        eprintln!("VFX_GAME_EFFECTS not set; skipped");
        return;
    };
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "eff") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    walk(std::path::Path::new(&root), &mut files);
    let mut seen = [0; 2];
    for file in files {
        let data = std::fs::read(&file).unwrap();
        let Ok(eff) = effect_library::namco_file::NamcoEffectFile::load(&data) else {
            continue;
        };
        let Some(p) = eff.ptcl_file else { continue };
        for e in p.emitter_list.emitter_sets.iter().flat_map(|s| &s.emitters) {
            let output = render::Output::of_draw_path(e.data.emitter_info.draw_path as u8);
            match e.data.shader_references.custom_shader_index {
                5 => {
                    assert_eq!(output, Some(render::Output::Offsets), "{:?}", e.data.name);
                    seen[0] += 1;
                }
                6 => {
                    assert_eq!(output, None, "{:?}", e.data.name);
                    seen[1] += 1;
                }
                _ => {}
            }
        }
    }
    assert_eq!(seen, [22, 65]);
}

/// The per-draw block is the 0x60 bytes 13.0.4 binds, with row 5 mapping screen x as is.
#[test]
fn custom2_has_the_screen_scale_row() {
    for block in [render::default_custom2(), render::game_custom2()] {
        assert_eq!(block.len(), 0x60);
        let row5: Vec<f32> = block[0x50..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(row5, [1.0, 0.0, 0.0, 0.0]);
    }
}
