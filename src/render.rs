//! Drawing emitters with the game's own shaders on wgpu.
//!
//! Each emitter names a shader program in its file. The program's Maxwell code is translated
//! once (`maxwell::wgsl`) and drawn the way `EmitterCalculator::DrawEmitterUsingBoundShader`
//! (Smash 0x1a1d0) draws it: a 4-vertex triangle strip instanced once per particle, with the
//! per-particle streams bound by attribute name, uniform blocks at their banks (view 8,
//! static 9, dynamic 10, custom 16) and the emitter's textures at their sampler slots.

use std::collections::HashMap;

use crate::bnsh::{Program, ShaderFile};
use crate::maxwell::{decode::decode_program, wgsl};

/// Banks the runtime fills; every other bank a program reads gets zeros.
pub const BANK_VIEW: u8 = 8;
pub const BANK_STATIC: u8 = 9;
pub const BANK_DYNAMIC: u8 = 10;
pub const BANK_CUSTOM0: u8 = 14;
pub const BANK_CUSTOM2: u8 = 16;

const BANK_BYTES: u64 = wgsl::CBUF_ROWS as u64 * 16;

/// nn::vfx blend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Blend {
    Normal,
    Add,
    Sub,
    Screen,
    Multiply,
    Off,
}

impl Blend {
    pub fn from_type(enabled: bool, kind: u8) -> Self {
        if !enabled {
            return Blend::Off;
        }
        match kind {
            1 => Blend::Add,
            2 => Blend::Sub,
            3 => Blend::Screen,
            4 => Blend::Multiply,
            _ => Blend::Normal,
        }
    }

    fn state(self) -> Option<wgpu::BlendState> {
        use wgpu::{BlendComponent, BlendFactor as F, BlendOperation as Op};
        let colour = match self {
            Blend::Off => return None,
            Blend::Normal => BlendComponent {
                src_factor: F::SrcAlpha,
                dst_factor: F::OneMinusSrcAlpha,
                operation: Op::Add,
            },
            Blend::Add => BlendComponent {
                src_factor: F::SrcAlpha,
                dst_factor: F::One,
                operation: Op::Add,
            },
            Blend::Sub => BlendComponent {
                src_factor: F::SrcAlpha,
                dst_factor: F::One,
                operation: Op::ReverseSubtract,
            },
            Blend::Screen => BlendComponent {
                src_factor: F::OneMinusDst,
                dst_factor: F::One,
                operation: Op::Add,
            },
            Blend::Multiply => BlendComponent {
                src_factor: F::Zero,
                dst_factor: F::Src,
                operation: Op::Add,
            },
        };
        Some(wgpu::BlendState {
            color: colour,
            alpha: BlendComponent {
                src_factor: F::One,
                dst_factor: F::OneMinusSrcAlpha,
                operation: Op::Add,
            },
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderState {
    pub blend: Blend,
    pub depth_test: bool,
    pub depth_write: bool,
    /// nn::gfx comparison function (0 never .. 7 always).
    pub depth_func: u8,
    /// 0 both sides, 1 front, 2 back.
    pub display_side: u8,
}

/// One emitter's draw for this frame.
pub struct EmitterDraw<'a> {
    pub program: usize,
    pub state: RenderState,
    pub static_block: &'a [u8],
    pub dynamic_block: &'a [u8],
    /// sysCustomShaderUniformBlock0 (the emitter's custom shader data).
    pub custom0_block: Option<&'a [u8]>,
    /// sysCustomShaderUniformBlock2: the game's per-draw block (0x50 bytes).
    pub custom2_block: Option<&'a [u8]>,
    /// Per-particle streams by attribute name (`sysLocalPosAttr`, ...), one vec4 per particle.
    pub streams: &'a HashMap<&'static str, Vec<[f32; 4]>>,
    pub count: u32,
    /// Texture for each sampler slot the emitter uses.
    pub textures: [Option<&'a wgpu::TextureView>; 3],
    pub samplers: [Option<&'a wgpu::Sampler>; 3],
    /// The emitter's mesh primitive, when it draws one instead of a quad.
    pub mesh: Option<&'a crate::mesh::Mesh>,
}

/// Inputs a mesh supplies per vertex, in `mesh::Mesh` vertex order.
const VERTEX_NAMES: [&str; 4] = [
    "sysPosAttr",
    "sysTexCoordAttr",
    "sysNormalAttr",
    "sysVertexColor0Attr",
];

struct Compiled {
    vertex: wgpu::ShaderModule,
    fragment: wgpu::ShaderModule,
    vs: wgsl::Interface,
    fs: wgsl::Interface,
    /// Vertex attribute location of each named input.
    locations: HashMap<String, u32>,
    vs_literals: wgpu::Buffer,
    fs_literals: wgpu::Buffer,
}

pub struct EffectRenderer {
    device: wgpu::Device,
    target: wgpu::TextureFormat,
    depth: Option<wgpu::TextureFormat>,
    programs: Vec<Program>,
    compiled: HashMap<usize, Result<Compiled, String>>,
    pipelines: HashMap<(usize, RenderState, bool), wgpu::RenderPipeline>,
    group0: wgpu::BindGroupLayout,
    group1: wgpu::BindGroupLayout,
    zero_bank: wgpu::Buffer,
    view_bank: wgpu::Buffer,
    zero_stream: wgpu::Buffer,
    blank: wgpu::TextureView,
    blank_sampler: wgpu::Sampler,
}

/// Banks every pipeline layout declares (a program that does not read one simply ignores it).
/// Bindings every pipeline layout declares: vertex literals (1), pixel literals (3), view (8),
/// static (9), dynamic (10), plugin (12), reserved custom (13), custom 0 (14), custom 2 (16).
const BANKS: [u8; 9] = [1, 3, 8, 9, 10, 12, 13, 14, 16];
const SLOTS: u32 = 3;
const MAX_INSTANCES: u64 = 4096;

impl EffectRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shaders: &ShaderFile,
        target: wgpu::TextureFormat,
        depth: Option<wgpu::TextureFormat>,
    ) -> Self {
        let group0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vfx banks"),
            entries: &BANKS
                .iter()
                .map(|b| wgpu::BindGroupLayoutEntry {
                    binding: *b as u32,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let mut entries = Vec::new();
        for s in 0..SLOTS {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: s * 2,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: s * 2 + 1,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        let group1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vfx textures"),
            entries: &entries,
        });
        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        use wgpu::util::DeviceExt;
        let blank = device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("vfx blank"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &[255, 255, 255, 255],
            )
            .create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            device: device.clone(),
            target,
            depth,
            programs: shaders.programs.clone(),
            compiled: HashMap::new(),
            pipelines: HashMap::new(),
            group0,
            group1,
            zero_bank: buffer("vfx zero bank", BANK_BYTES),
            view_bank: buffer("vfx view bank", BANK_BYTES),
            zero_stream: buffer("vfx zero stream", MAX_INSTANCES * 16),
            blank,
            blank_sampler: device.create_sampler(&wgpu::SamplerDescriptor::default()),
        }
    }

    /// The camera block (see `view_block`), shared by every emitter this frame.
    pub fn set_view(&self, queue: &wgpu::Queue, view_block: &[u8]) {
        queue.write_buffer(&self.view_bank, 0, view_block);
    }

    /// Why a program can't be drawn, if it can't.
    pub fn program_error(&mut self, program: usize) -> Option<String> {
        self.compile(program).err()
    }

    /// Run an emitter's vertex shader on the GPU as compute and read back what it outputs:
    /// for each of the first `instances` particles, 4 vertices of [position, varyings...].
    pub fn probe_vertex(
        &mut self,
        queue: &wgpu::Queue,
        draw: &EmitterDraw,
        instances: u32,
    ) -> Option<(Vec<u32>, Vec<[f32; 4]>)> {
        use wgpu::util::DeviceExt;
        let prepared = self.prepare(queue, draw)?;
        let program = self.programs.get(draw.program)?;
        let vs = program.stages[crate::bnsh::STAGE_VERTEX].as_ref()?;
        let t = wgsl::translate(
            wgsl::StageKind::Vertex,
            &decode_program(vs.code.get(0x50..)?),
        )
        .ok()?;
        let source = wgsl::vertex_as_compute(&t);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vfx probe"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let group2 = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[0u32, 1].map(|b| wgpu::BindGroupLayoutEntry {
                    binding: b,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: b == 0 },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }),
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&self.group0), Some(&self.group1), Some(&group2)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let n = instances.min(prepared.count);
        let outputs: Vec<u32> = t.interface.outputs.iter().copied().collect();
        let per = outputs.len() + 2;
        let input = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &self.instance_bytes(draw, &t.interface, n).max_len(16),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_size = (n as usize * 4 * per * 16) as u64;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: out_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let g2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &group2,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &prepared.group0, &[]);
            pass.set_bind_group(1, &prepared.group1, &[]);
            pass.set_bind_group(2, &g2, &[]);
            pass.dispatch_workgroups(n * 4, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &read, 0, out_size);
        queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let bytes = read.slice(..).get_mapped_range().to_vec();
        let values = bytes
            .chunks(16)
            .map(|c| {
                std::array::from_fn(|i| f32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap()))
            })
            .collect();
        Some((outputs, values))
    }

    fn instance_bytes(&self, draw: &EmitterDraw, iface: &wgsl::Interface, count: u32) -> Vec<u8> {
        let c = match self.compiled.get(&draw.program) {
            Some(Ok(c)) => c,
            _ => return Vec::new(),
        };
        let by_location: HashMap<u32, &str> =
            c.locations.iter().map(|(n, l)| (*l, n.as_str())).collect();
        let mut out = Vec::new();
        for i in 0..count as usize {
            for loc in &iface.inputs {
                let v = by_location
                    .get(loc)
                    .and_then(|n| draw.streams.get(n))
                    .and_then(|d| d.get(i))
                    .copied()
                    .unwrap_or([0.0; 4]);
                out.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            }
        }
        out
    }

    fn compile(&mut self, program: usize) -> Result<(), String> {
        if !self.compiled.contains_key(&program) {
            let made = Self::compile_program(&self.device, self.programs.get(program));
            self.compiled.insert(program, made);
        }
        self.compiled[&program]
            .as_ref()
            .map(|_| ())
            .map_err(Clone::clone)
    }

    fn compile_program(
        device: &wgpu::Device,
        program: Option<&Program>,
    ) -> Result<Compiled, String> {
        let program = program.ok_or("no such shader program")?;
        let vs = program.stages[crate::bnsh::STAGE_VERTEX]
            .as_ref()
            .ok_or("no vertex stage")?;
        let fs = program.stages[crate::bnsh::STAGE_FRAGMENT]
            .as_ref()
            .ok_or("no pixel stage")?;
        let body = |code: &[u8]| decode_program(code.get(0x50..).unwrap_or_default());
        let v = wgsl::translate(wgsl::StageKind::Vertex, &body(&vs.code))?;
        let f = wgsl::translate(wgsl::StageKind::Fragment, &body(&fs.code))?;
        let module = |label, source: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        };
        let literals = |bytes: &[u8]| {
            use wgpu::util::DeviceExt;
            let mut contents = vec![0u8; BANK_BYTES as usize];
            let n = bytes.len().min(contents.len());
            contents[..n].copy_from_slice(&bytes[..n]);
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vfx literals"),
                contents: &contents,
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        Ok(Compiled {
            vs_literals: literals(vs.literals()),
            fs_literals: literals(fs.literals()),
            vertex: module("vfx vertex", &v.source),
            fragment: module("vfx pixel", &f.source),
            locations: vs
                .reflection
                .inputs
                .iter()
                .filter(|b| b.slot >= 0)
                .map(|b| (b.name.clone(), b.slot as u32))
                .collect(),
            vs: v.interface,
            fs: f.interface,
        })
    }

    /// Input locations fed per vertex (mesh) and per instance (particle), in buffer order.
    fn split_inputs(c: &Compiled, mesh: bool) -> (Vec<(u32, usize)>, Vec<u32>) {
        let mut per_vertex = Vec::new();
        let mut per_instance = Vec::new();
        for loc in &c.vs.inputs {
            let name = c
                .locations
                .iter()
                .find(|(_, l)| *l == loc)
                .map(|(n, _)| n.as_str());
            match name.and_then(|n| VERTEX_NAMES.iter().position(|v| *v == n)) {
                Some(slot) if mesh => per_vertex.push((*loc, slot)),
                _ => per_instance.push(*loc),
            }
        }
        (per_vertex, per_instance)
    }

    fn pipeline(
        &mut self,
        program: usize,
        state: RenderState,
        mesh: bool,
    ) -> Option<&wgpu::RenderPipeline> {
        self.compile(program).ok()?;
        if !self.pipelines.contains_key(&(program, state, mesh)) {
            let c = self.compiled[&program].as_ref().ok()?;
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("vfx"),
                    bind_group_layouts: &[Some(&self.group0), Some(&self.group1)],
                    immediate_size: 0,
                });
            // Slot 0: the mesh's four vertex inputs (64 bytes a vertex). Slot 1: every particle
            // stream interleaved, one vec4 each in input order.
            let (per_vertex, per_instance) = Self::split_inputs(c, mesh);
            let vertex_attributes: Vec<wgpu::VertexAttribute> = per_vertex
                .iter()
                .map(|(loc, slot)| wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: *slot as u64 * 16,
                    shader_location: *loc,
                })
                .collect();
            let instance_attributes: Vec<wgpu::VertexAttribute> = per_instance
                .iter()
                .enumerate()
                .map(|(i, loc)| wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: i as u64 * 16,
                    shader_location: *loc,
                })
                .collect();
            let buffers = vec![
                wgpu::VertexBufferLayout {
                    array_stride: 64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &vertex_attributes,
                },
                wgpu::VertexBufferLayout {
                    array_stride: (instance_attributes.len().max(1) as u64) * 16,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &instance_attributes,
                },
            ];
            let compare = match state.depth_func {
                0 => wgpu::CompareFunction::Never,
                1 => wgpu::CompareFunction::Less,
                2 => wgpu::CompareFunction::Equal,
                3 => wgpu::CompareFunction::LessEqual,
                4 => wgpu::CompareFunction::Greater,
                5 => wgpu::CompareFunction::NotEqual,
                6 => wgpu::CompareFunction::GreaterEqual,
                _ => wgpu::CompareFunction::Always,
            };
            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("vfx emitter"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &c.vertex,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        buffers: &buffers,
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: if mesh {
                            wgpu::PrimitiveTopology::TriangleList
                        } else {
                            wgpu::PrimitiveTopology::TriangleStrip
                        },
                        cull_mode: match state.display_side {
                            1 => Some(wgpu::Face::Back),
                            2 => Some(wgpu::Face::Front),
                            _ => None,
                        },
                        ..Default::default()
                    },
                    depth_stencil: self.depth.map(|format| wgpu::DepthStencilState {
                        format,
                        depth_write_enabled: Some(state.depth_write),
                        depth_compare: Some(if state.depth_test {
                            compare
                        } else {
                            wgpu::CompareFunction::Always
                        }),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &c.fragment,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: self.target,
                            blend: state.blend.state(),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
            self.pipelines.insert((program, state, mesh), pipeline);
        }
        self.pipelines.get(&(program, state, mesh))
    }

    /// Upload one emitter's data and record its draw. Buffers live in the returned frame data,
    /// which must outlive the render pass.
    pub fn prepare(&mut self, queue: &wgpu::Queue, draw: &EmitterDraw) -> Option<PreparedDraw> {
        let count = draw.count.min(MAX_INSTANCES as u32);
        if count == 0 {
            return None;
        }
        let mesh = draw.mesh.is_some();
        self.pipeline(draw.program, draw.state, mesh)?;
        let pipeline = self.pipelines[&(draw.program, draw.state, mesh)].clone();
        let c = self.compiled[&draw.program].as_ref().ok()?;
        let make = |label: &str, bytes: &[u8], usage: wgpu::BufferUsages, size: u64| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(bytes.len() as u64).next_multiple_of(16),
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buffer, 0, bytes);
            buffer
        };
        let uniform = wgpu::BufferUsages::UNIFORM;
        let statics = make("vfx static", draw.static_block, uniform, BANK_BYTES);
        let dynamic = make("vfx dynamic", draw.dynamic_block, uniform, BANK_BYTES);
        let custom0 = draw
            .custom0_block
            .map(|b| make("vfx custom0", b, uniform, BANK_BYTES));
        let custom2 = draw
            .custom2_block
            .map(|b| make("vfx custom2", b, uniform, BANK_BYTES));
        let bank = |b: u8| -> &wgpu::Buffer {
            match b {
                1 => &c.vs_literals,
                3 => &c.fs_literals,
                BANK_VIEW => &self.view_bank,
                BANK_STATIC => &statics,
                BANK_DYNAMIC => &dynamic,
                BANK_CUSTOM0 => custom0.as_ref().unwrap_or(&self.zero_bank),
                BANK_CUSTOM2 => custom2.as_ref().unwrap_or(&self.zero_bank),
                _ => &self.zero_bank,
            }
        };
        let group0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vfx banks"),
            layout: &self.group0,
            entries: &BANKS
                .iter()
                .map(|b| wgpu::BindGroupEntry {
                    binding: *b as u32,
                    resource: bank(*b).as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut entries = Vec::new();
        for s in 0..SLOTS as usize {
            entries.push(wgpu::BindGroupEntry {
                binding: s as u32 * 2,
                resource: wgpu::BindingResource::TextureView(
                    draw.textures[s].unwrap_or(&self.blank),
                ),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: s as u32 * 2 + 1,
                resource: wgpu::BindingResource::Sampler(
                    draw.samplers[s].unwrap_or(&self.blank_sampler),
                ),
            });
        }
        let group1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vfx textures"),
            layout: &self.group1,
            entries: &entries,
        });
        let by_location: HashMap<u32, &str> =
            c.locations.iter().map(|(n, l)| (*l, n.as_str())).collect();
        let (_, inputs) = Self::split_inputs(c, mesh);
        let mut interleaved: Vec<[f32; 4]> = Vec::with_capacity(count as usize * inputs.len());
        for i in 0..count as usize {
            for loc in &inputs {
                let value = by_location
                    .get(loc)
                    .and_then(|name| draw.streams.get(name))
                    .and_then(|d| d.get(i))
                    .copied()
                    .unwrap_or([0.0; 4]);
                interleaved.push(value);
            }
        }
        if interleaved.is_empty() {
            interleaved.push([0.0; 4]);
        }
        let instances = Some(make(
            "vfx instances",
            bytemuck_cast(&interleaved),
            wgpu::BufferUsages::VERTEX,
            16,
        ));
        let (vertices, indices) = match draw.mesh {
            Some(m) => {
                let flat: Vec<[f32; 4]> = m.vertices.iter().flatten().copied().collect();
                let idx: Vec<u8> = m.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
                (
                    make(
                        "vfx mesh",
                        bytemuck_cast(&flat),
                        wgpu::BufferUsages::VERTEX,
                        64,
                    ),
                    Some((
                        make("vfx indices", &idx, wgpu::BufferUsages::INDEX, 4),
                        m.indices.len() as u32,
                    )),
                )
            }
            None => (
                make("vfx quad", &[0u8; 256], wgpu::BufferUsages::VERTEX, 256),
                None,
            ),
        };
        let _ = &c.fs;
        Some(PreparedDraw {
            pipeline,
            group0,
            group1,
            instances,
            vertices,
            indices,
            count,
            _keep: vec![statics, dynamic],
        })
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, prepared: &PreparedDraw) {
        pass.set_pipeline(&prepared.pipeline);
        pass.set_bind_group(0, &prepared.group0, &[]);
        pass.set_bind_group(1, &prepared.group1, &[]);
        pass.set_vertex_buffer(0, prepared.vertices.slice(..));
        if let Some(buffer) = &prepared.instances {
            pass.set_vertex_buffer(1, buffer.slice(..));
        }
        match &prepared.indices {
            Some((buffer, n)) => {
                pass.set_index_buffer(buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*n, 0, 0..prepared.count);
            }
            None => pass.draw(0..4, 0..prepared.count),
        }
    }
}

pub struct PreparedDraw {
    pipeline: wgpu::RenderPipeline,
    group0: wgpu::BindGroup,
    group1: wgpu::BindGroup,
    instances: Option<wgpu::Buffer>,
    vertices: wgpu::Buffer,
    indices: Option<(wgpu::Buffer, u32)>,
    count: u32,
    _keep: Vec<wgpu::Buffer>,
}

fn bytemuck_cast(values: &[[f32; 4]]) -> &[u8] {
    // SAFETY: [f32; 4] is plain data with no padding.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

trait MaxLen {
    fn max_len(self, n: usize) -> Self;
}

impl MaxLen for Vec<u8> {
    fn max_len(mut self, n: usize) -> Self {
        if self.len() < n {
            self.resize(n, 0);
        }
        self
    }
}

/// Smash's sysCustomShaderUniformBlock2 (0x50 bytes) as its effect manager sets it up
/// (main 0x260ab0c on v0): row 0 x is the draw's multiplier (1), rows 0-1 otherwise carry stage
/// lighting, row 1 z is -1 (set per draw at 0x2600fxx; directional particles cross their motion
/// with -z times the eye vector), row 2 is (1, 0, 1, 0), row 3 per-set data (0), row 4
/// (1, -99999, 1, 0).
pub fn default_custom2() -> Vec<u8> {
    let rows: [f32; 20] = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, -1.0, 0.0, //
        1.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, //
        1.0, -99999.0, 1.0, 0.0,
    ];
    rows.iter().flat_map(|v| v.to_le_bytes()).collect()
}
