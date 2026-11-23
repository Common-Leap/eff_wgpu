//! The frame behind the effects, copied for their shaders the way the game hands it to them:
//! sysFrameBufferTexture (slot 3) is the scene's colour and sysDepthBufferTexture (slot 4) its
//! depth, both as they stand before the effects draw.
//!
//! Refraction shaders sample the colour at their own screen position plus an offset (Zelda's
//! Din's Fire bomb); floor decals sample the depth and keep only the pixels that lie within a
//! set distance of the scene behind them (Kazuya's ground cracks), so they draw on the floor
//! and nowhere else. Without a copy the renderer falls back to a flat colour and to depth 1.0,
//! where refraction draws as a solid sheet and every floor decal is discarded.

/// Where the copies go: colour in the scene's own format, depth as one float per pixel, which
/// the effect shaders read as an ordinary texture.
pub const DEPTH_COPY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

const SHADER: &str = r#"
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
@group(0) @binding(0) var source: SOURCE;
fn texel(at: vec4<f32>) -> vec2<i32> {
    let size = vec2<i32>(textureDimensions(source)) - 1;
    return clamp(vec2<i32>(at.xy), vec2<i32>(0), size);
}
@fragment fn main(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    return READ;
}
"#;

struct Copier {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl Copier {
    fn new(
        device: &wgpu::Device,
        source: wgpu::BindingType,
        wgsl_type: &str,
        read: &str,
        target: wgpu::TextureFormat,
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vfx scene copy"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: source,
                count: None,
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vfx scene copy"),
            source: wgpu::ShaderSource::Wgsl(
                SHADER
                    .replace("SOURCE", wgsl_type)
                    .replace("READ", read)
                    .into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vfx scene copy"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vfx scene copy"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(target.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { layout, pipeline }
    }

    fn run(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        target: &wgpu::TextureView,
    ) {
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vfx scene copy"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source),
            }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("vfx scene copy"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Copies of a scene's colour and depth, sized to it, for [crate::render::EffectRenderer::set_scene].
pub struct SceneCopy {
    colour_format: wgpu::TextureFormat,
    colour: Copier,
    depth: Copier,
    size: (u32, u32),
    colour_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

impl SceneCopy {
    /// Copies of a single-sampled colour target in `colour_format` and of a depth buffer with
    /// `depth_samples` samples per pixel (the first sample is taken, as a game frame has one).
    pub fn new(
        device: &wgpu::Device,
        colour_format: wgpu::TextureFormat,
        depth_samples: u32,
    ) -> Self {
        let colour = Copier::new(
            device,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            "texture_2d<f32>",
            "textureLoad(source, texel(at), 0)",
            colour_format,
        );
        let multisampled = depth_samples > 1;
        let depth = Copier::new(
            device,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled,
            },
            if multisampled {
                "texture_depth_multisampled_2d"
            } else {
                "texture_depth_2d"
            },
            "vec4<f32>(textureLoad(source, texel(at), 0), 0.0, 0.0, 1.0)",
            DEPTH_COPY_FORMAT,
        );
        let (colour_view, depth_view) = Self::targets(device, colour_format, (1, 1));
        Self {
            colour_format,
            colour,
            depth,
            size: (1, 1),
            colour_view,
            depth_view,
        }
    }

    fn targets(
        device: &wgpu::Device,
        colour_format: wgpu::TextureFormat,
        (width, height): (u32, u32),
    ) -> (wgpu::TextureView, wgpu::TextureView) {
        let make = |label, format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        (
            make("vfx scene colour", colour_format),
            make("vfx scene depth", DEPTH_COPY_FORMAT),
        )
    }

    /// Match the scene's size. The views change, so hand them to the renderers again.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let size = (width.max(1), height.max(1));
        if size != self.size {
            self.size = size;
            (self.colour_view, self.depth_view) = Self::targets(device, self.colour_format, size);
        }
    }

    /// The colour copy, for slot 3.
    pub fn colour_view(&self) -> &wgpu::TextureView {
        &self.colour_view
    }

    /// The depth copy (0 near to 1 far), for slot 4.
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// Record the copies: call it after the scene is drawn and before the effects are.
    pub fn copy(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        colour: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) {
        self.colour.run(device, encoder, colour, &self.colour_view);
        self.depth.run(device, encoder, depth, &self.depth_view);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A multisampled depth buffer and a colour target copy pixel for pixel, the depth as the
    /// float the effect shaders compare against. Skipped without a GPU.
    #[test]
    fn copies_colour_and_multisampled_depth() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("no GPU; skipped");
            return;
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let (width, height) = (8u32, 4u32);
        let texture = |format, samples, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | usage,
                view_formats: &[],
            })
        };
        let colour_format = wgpu::TextureFormat::Rgba16Float;
        let colour = texture(colour_format, 1, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth = texture(
            wgpu::TextureFormat::Depth32Float,
            4,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let colour_view = colour.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &colour_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.25,
                        g: 0.5,
                        b: 2.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.75),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        let mut copy = SceneCopy::new(&device, colour_format, 4);
        copy.resize(&device, width, height);
        copy.copy(&device, &mut encoder, &colour_view, &depth_view);
        // Read both back: 8 bytes a colour pixel, 4 a depth pixel (rows padded to 256).
        let read = |encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 256 * height as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                view.texture().as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: None,
                    },
                },
                view.texture().size(),
            );
            buffer
        };
        let colour_out = read(&mut encoder, copy.colour_view());
        let depth_out = read(&mut encoder, copy.depth_view());
        queue.submit([encoder.finish()]);
        for buffer in [&colour_out, &depth_out] {
            buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let bytes = |buffer: &wgpu::Buffer| buffer.slice(..).get_mapped_range().unwrap().to_vec();
        let colour_bytes = bytes(&colour_out);
        let depth_bytes = bytes(&depth_out);
        for y in 0..height as usize {
            for x in 0..width as usize {
                let at = y * 256 + x * 8;
                let half = |i: usize| {
                    let bits = u16::from_le_bytes([
                        colour_bytes[at + i * 2],
                        colour_bytes[at + i * 2 + 1],
                    ]);
                    // Every value here is exact in half precision: sign 0, exponent, mantissa.
                    let exponent = ((bits >> 10) & 0x1f) as i32 - 15;
                    (1.0 + (bits & 0x3ff) as f32 / 1024.0) * 2f32.powi(exponent)
                };
                assert_eq!([half(0), half(1), half(2), half(3)], [0.25, 0.5, 2.0, 1.0]);
                let at = y * 256 + x * 4;
                let depth = f32::from_le_bytes(depth_bytes[at..at + 4].try_into().unwrap());
                assert_eq!(depth, 0.75, "depth at {x},{y}");
            }
        }
    }
}
