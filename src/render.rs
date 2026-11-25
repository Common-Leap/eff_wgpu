//! Drawing emitters with the game's own shaders on wgpu.
//!
//! Each emitter names a shader program in its file. The program's Maxwell code is translated
//! once (`maxwell::wgsl`) and drawn the way `EmitterCalculator::DrawEmitterUsingBoundShader`
//! (Smash 0x1a1d0) draws it: a 4-vertex triangle strip instanced once per particle, with the
//! per-particle streams bound by attribute name, uniform blocks at their banks (view 8,
//! static 9, dynamic 10, custom 16) and the emitter's textures at their sampler slots.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::bnsh::{Program, ShaderFile};
use crate::maxwell::{decode::decode_program, wgsl};

/// Banks the runtime fills; every other bank a program reads gets zeros.
pub const BANK_VIEW: u8 = 8;
pub const BANK_STATIC: u8 = 9;
pub const BANK_DYNAMIC: u8 = 10;
pub const BANK_CUSTOM0: u8 = 14;
/// sysCustomShaderReservedUniformBlockParam (slot 10): the emitter's custom shader parameters.
/// 64 programs of the shared file read their colour modulation from it (sys_crown's crowns);
/// left as zeros, they drew black.
pub const BANK_CUSTOM_PARAM: u8 = 13;
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
            // Rendercontext::Initialize (SMO 0xb51990): 3 is (zero, src colour), 4 is
            // (one minus dst colour, one).
            3 => Blend::Multiply,
            4 => Blend::Screen,
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
            // nn::vfx leaves the target's alpha as it was (ZERO, DST_ALPHA; seen in a capture
            // of the game's fireball draw).
            alpha: BlendComponent {
                src_factor: F::Zero,
                dst_factor: F::DstAlpha,
                operation: Op::Add,
            },
        })
    }
}

/// What an emitter's pixel shader writes, which its draw path (res+0x7d4 in the emitter's data,
/// `emitter_info.draw_path`) decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Output {
    /// A colour, into the scene.
    Colour,
    /// How far to move the picture behind it, into the game's distortion buffer: (value - 0.5)
    /// times the part's strength over (960, 540), in screen UV (see `scene::Distortion`).
    Offsets,
}

impl Output {
    /// Draw paths 12 and 16 hold every emitter of custom shader 5 in the game (22 of 37,420:
    /// Ryu's final smash wave, Sephiroth's down throw, Bayonetta's bats and stilettos); drawn
    /// as colour, their offsets are black. Path 6 holds every emitter of custom shader 6 (65,
    /// all named radialblur but one), which write a velocity for the game's motion blur: there
    /// is none here, so nothing to draw.
    pub fn of_draw_path(path: u8) -> Option<Self> {
        match path {
            6 => None,
            12 | 16 => Some(Output::Offsets),
            _ => Some(Output::Colour),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderState {
    pub output: Output,
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
    /// The emitter's custom shader data (its CSDP attribute), bound as
    /// sysCustomShaderReservedUniformBlockParam, and as sysCustomShaderUniformBlock0 unless
    /// `user_block` gives that one.
    pub custom0_block: Option<&'a [u8]>,
    /// sysCustomShaderUniformBlock0 as the game fills it for this draw, where it does: Inkling's
    /// ink shaders read her ink colour from it (row 0 multiplies the colour, row 1 is added).
    pub user_block: Option<&'a [u8]>,
    /// sysCustomShaderUniformBlock2: the game's per-draw block (0x60 bytes).
    pub custom2_block: Option<&'a [u8]>,
    /// Per-particle streams by attribute name (`sysLocalPosAttr`, ...), one vec4 per particle.
    pub streams: &'a HashMap<&'static str, Vec<[f32; 4]>>,
    pub count: u32,
    /// Texture for each sampler slot the emitter uses.
    pub textures: [Option<&'a GpuTexture>; 3],
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
    /// Texture bindings in the dimensions this program's shaders declare.
    group1: wgpu::BindGroupLayout,
    arrays: [bool; SLOTS as usize],
}

/// A texture on the GPU, viewed both as a plain 2D image (its first layer) and as an array, so
/// it can be bound to whichever a shader declares.
pub struct GpuTexture {
    pub flat: wgpu::TextureView,
    pub array: wgpu::TextureView,
}

impl GpuTexture {
    fn view(&self, array: bool) -> &wgpu::TextureView {
        if array {
            &self.array
        } else {
            &self.flat
        }
    }
}

/// The texture layout for a program: 2D or 2D-array per slot, as its stages declare.
/// The depth copy is one 32-bit float a pixel, which is not filterable everywhere: its slot
/// takes it unfiltered, through a sampler that does not filter.
fn texture_layout(device: &wgpu::Device, arrays: &[bool; SLOTS as usize]) -> wgpu::BindGroupLayout {
    let mut entries = Vec::new();
    for s in 0..SLOTS {
        let filterable = s as usize != SLOT_DEPTH;
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: s * 2,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: if arrays[s as usize] {
                    wgpu::TextureViewDimension::D2Array
                } else {
                    wgpu::TextureViewDimension::D2
                },
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: s * 2 + 1,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(if filterable {
                wgpu::SamplerBindingType::Filtering
            } else {
                wgpu::SamplerBindingType::NonFiltering
            }),
            count: None,
        });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vfx textures"),
        entries: &entries,
    })
}

pub struct EffectRenderer {
    device: wgpu::Device,
    target: wgpu::TextureFormat,
    depth: Option<wgpu::TextureFormat>,
    /// Samples per pixel of the target (and depth) this draws into.
    samples: u32,
    programs: Vec<Program>,
    /// The caches sit behind locks so a worker thread can compile ahead (`warm`) while the
    /// paint thread draws what is ready. Compiling happens outside the locks.
    compiled: Mutex<HashMap<(usize, bool), Result<Arc<Compiled>, String>>>,
    /// `None` for a pipeline wgpu rejected: that emitter is skipped rather than drawn.
    pipelines: Mutex<HashMap<PipelineKey, Option<wgpu::RenderPipeline>>>,
    /// Shader or pipeline errors caught since the last `take_failures`.
    failures: Mutex<Vec<String>>,
    group0: wgpu::BindGroupLayout,
    zero_bank: wgpu::Buffer,
    view_bank: wgpu::Buffer,
    blank: GpuTexture,
    /// Nearest, clamped: also what the depth copy is read through.
    blank_sampler: wgpu::Sampler,
    /// Linear, clamped: the frame copy, which refraction reads between pixels.
    frame_sampler: wgpu::Sampler,
    /// What the pixel shaders read as the frame drawn so far (sysFrameBufferTexture, slot 3)
    /// and its depth (sysDepthBufferTexture, slot 4), see `scene::SceneCopy`. Without a scene
    /// copy the frame is the background colour and the depth is 1.0, "nothing behind": soft
    /// particles draw in full and floor decals not at all.
    background: wgpu::Texture,
    scene: Mutex<(Option<wgpu::TextureView>, Option<wgpu::TextureView>)>,
}

/// Banks every pipeline layout declares (a program that does not read one simply ignores it).
/// Bindings every pipeline layout declares: vertex literals (1), pixel literals (3), view (8),
/// static (9), dynamic (10), plugin (12), reserved custom (13), custom 0 (14), custom 2 (16).
const BANKS: [u8; 9] = [1, 3, 8, 9, 10, 12, 13, 14, 16];
/// Texture slots: the emitter's three samplers, then the frame copy (3) and depth (4) the game
/// binds for every draw. A layout without the last two fails pipeline creation for the 2,900
/// shader stages that read them, and wgpu treats that as fatal.
const SLOTS: u32 = 5;
const SLOT_FRAME: usize = 3;
const SLOT_DEPTH: usize = 4;
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
        let blank = upload_texture(
            device,
            queue,
            &crate::texture::Texture {
                width: 1,
                height: 1,
                layers: 1,
                srgb: false,
                mips: vec![vec![255, 255, 255, 255]],
            },
        );
        let background = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("vfx background"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[0, 0, 0, 255],
        );
        Self {
            device: device.clone(),
            target,
            depth,
            samples: 1,
            programs: shaders.programs.clone(),
            compiled: Mutex::default(),
            pipelines: Mutex::default(),
            group0,
            zero_bank: buffer("vfx zero bank", BANK_BYTES),
            view_bank: buffer("vfx view bank", BANK_BYTES),
            blank,
            blank_sampler: device.create_sampler(&wgpu::SamplerDescriptor::default()),
            frame_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            background,
            scene: Mutex::default(),
            failures: Mutex::default(),
        }
    }

    /// Draw into a multisampled target with this many samples (and depth with as many).
    /// Set it before the first draw: pipelines are built for it.
    pub fn with_sample_count(mut self, samples: u32) -> Self {
        self.samples = samples.max(1);
        self
    }

    /// Shader and pipeline errors caught since the last call. A rejected program is skipped,
    /// never fatal: wgpu's default handler aborts the whole app on a validation error.
    pub fn take_failures(&self) -> Vec<String> {
        std::mem::take(&mut *lock(&self.failures))
    }

    /// Whether `prepare` can draw this emitter without compiling anything first.
    pub fn is_ready(&self, program: usize, state: RenderState, mesh: bool) -> bool {
        lock(&self.pipelines).contains_key(&(program, state, mesh))
    }

    /// Compile a program and its pipeline now, so a later `prepare` does not stall on it.
    /// Safe to call from another thread while this renderer draws.
    pub fn warm(&self, program: usize, state: RenderState, mesh: bool) {
        self.pipeline(program, state, mesh);
    }

    /// The colour shaders see as "the frame behind" when no scene copy is given.
    pub fn set_background(&self, queue: &wgpu::Queue, rgba: [u8; 4]) {
        queue.write_texture(
            self.background.as_image_copy(),
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    /// A copy of the frame drawn so far and its depth (as 0..1 in an unfiltered float texture,
    /// see `scene::SceneCopy`), for refraction, soft-particle and floor-decal shaders. `None`
    /// falls back to the background and to 1.0. Draws prepared from then on read these.
    pub fn set_scene(&self, color: Option<wgpu::TextureView>, depth: Option<wgpu::TextureView>) {
        *lock(&self.scene) = (color, depth);
    }

    /// The camera block (see `view_block`), shared by every emitter this frame.
    pub fn set_view(&self, queue: &wgpu::Queue, view_block: &[u8]) {
        queue.write_buffer(&self.view_bank, 0, view_block);
    }

    /// Why a program can't be drawn, if it can't.
    pub fn program_error(&self, program: usize) -> Option<String> {
        self.compile(program, Output::Colour).err()
    }

    /// Run an emitter's vertex shader on the GPU as compute and read back what it outputs:
    /// for each of the first `instances` particles, 4 vertices of [position, varyings...].
    pub fn probe_vertex(
        &self,
        queue: &wgpu::Queue,
        draw: &EmitterDraw,
        instances: u32,
    ) -> Option<(Vec<u32>, Vec<[f32; 4]>)> {
        use wgpu::util::DeviceExt;
        let prepared = self.prepare(queue, draw)?;
        let c_group1 = self
            .compile(draw.program, draw.state.output)
            .ok()?
            .group1
            .clone();
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
                bind_group_layouts: &[Some(&self.group0), Some(&c_group1), Some(&group2)],
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
        let bytes = read.slice(..).get_mapped_range().ok()?.to_vec();
        let values = bytes
            .chunks(16)
            .map(|c| {
                std::array::from_fn(|i| f32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap()))
            })
            .collect();
        Some((outputs, values))
    }

    fn instance_bytes(&self, draw: &EmitterDraw, iface: &wgsl::Interface, count: u32) -> Vec<u8> {
        let Ok(c) = self.compile(draw.program, draw.state.output) else {
            return Vec::new();
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

    fn compile(&self, program: usize, output: Output) -> Result<Arc<Compiled>, String> {
        // Only a colour is encoded: an offset is a number, negative as often as not.
        let encode = output == Output::Colour && encodes_srgb(self.target);
        if let Some(made) = lock(&self.compiled).get(&(program, encode)) {
            return made.clone();
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut made =
            Self::compile_program(&self.device, self.programs.get(program), encode).map(Arc::new);
        if let Some(error) = caught(scope) {
            let message = format!("program {program}: {error}");
            lock(&self.failures).push(message.clone());
            made = Err(message);
        }
        // Another thread may have compiled it meanwhile; keep the first so both agree.
        lock(&self.compiled)
            .entry((program, encode))
            .or_insert(made)
            .clone()
    }

    fn compile_program(
        device: &wgpu::Device,
        program: Option<&Program>,
        encode_output: bool,
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
        let mut f = wgsl::translate(wgsl::StageKind::Fragment, &body(&fs.code))?;
        if encode_output {
            // The shaders compute linear colour (sRGB textures arrive linear, constants are
            // linear); the game's frame is encoded to sRGB on the way to the screen. A target
            // that is not an sRGB format gets that encoding here. Blending then happens on
            // encoded values, where the game blends linear ones.
            f.source = f.source.replace(
                "return run();",
                "let c = run(); return vec4<f32>(srgb_encode(c.rgb), c.a);",
            );
            f.source.push_str(SRGB_ENCODE);
        }
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
        let mut arrays = [false; SLOTS as usize];
        for (slot, dim) in v
            .interface
            .textures
            .iter()
            .chain(f.interface.textures.iter())
        {
            if let Some(a) = arrays.get_mut(*slot as usize) {
                *a |= *dim == wgsl::TexDim::Array2D;
            }
        }
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
            group1: texture_layout(device, &arrays),
            arrays,
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
        &self,
        program: usize,
        state: RenderState,
        mesh: bool,
    ) -> Option<wgpu::RenderPipeline> {
        let c = self.compile(program, state.output).ok()?;
        if let Some(made) = lock(&self.pipelines).get(&(program, state, mesh)) {
            return made.clone();
        }
        {
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("vfx"),
                    bind_group_layouts: &[Some(&self.group0), Some(&c.group1)],
                    immediate_size: 0,
                });
            // Slot 0: the mesh's four vertex inputs (64 bytes a vertex). Slot 1: every particle
            // stream interleaved, one vec4 each in input order.
            let (per_vertex, per_instance) = Self::split_inputs(&c, mesh);
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
            let instance_layout = wgpu::VertexBufferLayout {
                array_stride: (instance_attributes.len().max(1) as u64) * 16,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &instance_attributes,
            };
            // Quads read nothing per vertex (corners come from the vertex index): their
            // particle streams go in slot 0. An attribute-less layout in front of them left
            // every stream reading zero, so quad emitters collapsed to a point.
            let buffers = if mesh {
                vec![
                    wgpu::VertexBufferLayout {
                        array_stride: 64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attributes,
                    },
                    instance_layout,
                ]
            } else {
                vec![instance_layout]
            };
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
                        buffers: &buffers.into_iter().map(Some).collect::<Vec<_>>(),
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
                    multisample: wgpu::MultisampleState {
                        count: self.samples,
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &c.fragment,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: match state.output {
                                Output::Colour => self.target,
                                Output::Offsets => crate::scene::OFFSETS_FORMAT,
                            },
                            blend: state.blend.state(),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
            let pipeline = match caught(scope) {
                Some(error) => {
                    lock(&self.failures).push(format!("program {program} pipeline: {error}"));
                    None
                }
                None => Some(pipeline),
            };
            lock(&self.pipelines)
                .entry((program, state, mesh))
                .or_insert(pipeline)
                .clone()
        }
    }

    /// Upload one emitter's data and record its draw. Buffers live in the returned frame data,
    /// which must outlive the render pass.
    pub fn prepare(&self, queue: &wgpu::Queue, draw: &EmitterDraw) -> Option<PreparedDraw> {
        let count = draw.count.min(MAX_INSTANCES as u32);
        if count == 0 {
            return None;
        }
        let mesh = draw.mesh.is_some();
        let pipeline = self.pipeline(draw.program, draw.state, mesh)?;
        let c = self.compile(draw.program, draw.state.output).ok()?;
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
        let user = draw
            .user_block
            .map(|b| make("vfx user block", b, uniform, BANK_BYTES));
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
                BANK_CUSTOM0 => user
                    .as_ref()
                    .or(custom0.as_ref())
                    .unwrap_or(&self.zero_bank),
                BANK_CUSTOM_PARAM => custom0.as_ref().unwrap_or(&self.zero_bank),
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
        let background = self.background.create_view(&Default::default());
        let scene = lock(&self.scene).clone();
        for s in 0..SLOTS as usize {
            let view = match s {
                SLOT_FRAME => scene.0.as_ref().unwrap_or(&background),
                SLOT_DEPTH => scene.1.as_ref().unwrap_or(&self.blank.flat),
                _ => draw.textures[s].unwrap_or(&self.blank).view(c.arrays[s]),
            };
            let sampler = match s {
                SLOT_FRAME => &self.frame_sampler,
                _ => draw
                    .samplers
                    .get(s)
                    .copied()
                    .flatten()
                    .unwrap_or(&self.blank_sampler),
            };
            entries.push(wgpu::BindGroupEntry {
                binding: s as u32 * 2,
                resource: wgpu::BindingResource::TextureView(view),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: s as u32 * 2 + 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            });
        }
        let group1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vfx textures"),
            layout: &c.group1,
            entries: &entries,
        });
        let by_location: HashMap<u32, &str> =
            c.locations.iter().map(|(n, l)| (*l, n.as_str())).collect();
        let (_, inputs) = Self::split_inputs(&c, mesh);
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
        let instance_slot = if prepared.indices.is_some() {
            pass.set_vertex_buffer(0, prepared.vertices.slice(..));
            1
        } else {
            0
        };
        if let Some(buffer) = &prepared.instances {
            pass.set_vertex_buffer(instance_slot, buffer.slice(..));
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

/// Smash's sysCustomShaderUniformBlock2 (0x60 bytes). Rows 0-4 are as read from the game's GPU
/// memory in a capture of a training-stage frame (13.0.4, Ryujinx + RenderDoc): every effect draw in the
/// frame bound this one block. Row 0 x is a global colour scale: the game writes 0.5 and draws
/// into a buffer its post pass scales back up (x1.3703 in gamma space, x2 linear). Use
/// [game_custom2] when drawing into a scene that gets that post pass; this block's 1.0 is the
/// equivalent for a target shown as is.
///
/// Row 5 is past what the capture read. 13.0.4 builds the block at main+0x355a808 (0x60 bytes
/// from a global; v0 built 0x50) and fills row 5 at main+0x374de74 with (1, 0, 0, 0) from
/// rodata 0x4467ca0 while a byte flag is clear, else a half-width (0.5, 0.5, 0, 0). Every
/// shader that reads the frame or depth copy maps its screen x through row 5 (x times row 5 x,
/// plus row 5 y times view row 17 w): left at zero, each of them read the copy's first column
/// instead of the pixel behind it.
pub fn default_custom2() -> Vec<u8> {
    custom2_with_scale(1.0)
}

/// [default_custom2] exactly as the game binds it, with the 0.5 scale its post pass undoes.
pub fn game_custom2() -> Vec<u8> {
    custom2_with_scale(0.5)
}

fn custom2_with_scale(scale: f32) -> Vec<u8> {
    let rows: [f32; 24] = [
        scale, 0.0, 0.0, 0.0, //
        1.0, 1.0, 1.0, 1.0, //
        1.0, 0.0, 1.0, -1.0, //
        0.0, 0.0, 0.0, 0.0, //
        1.0, -99999.0, 1.0, 0.0, //
        1.0, 0.0, 0.0, 0.0,
    ];
    rows.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Whether shaders must encode their output to sRGB themselves: for 8-bit targets shown as
/// they are. Float and 10-bit targets hold linear values like the game's own scene buffer.
fn encodes_srgb(target: wgpu::TextureFormat) -> bool {
    matches!(
        target,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
    )
}

/// A decoded texture on the GPU with all its mip levels.
pub fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &crate::texture::Texture,
) -> GpuTexture {
    use wgpu::util::DeviceExt;
    let data: Vec<u8> = texture.mips.iter().flatten().copied().collect();
    let gpu = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("vfx texture"),
            size: wgpu::Extent3d {
                width: texture.width,
                height: texture.height,
                depth_or_array_layers: texture.layers.max(1),
            },
            mip_level_count: texture.mips.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if texture.srgb {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::MipMajor,
        &data,
    );
    GpuTexture {
        flat: gpu.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            array_layer_count: Some(1),
            ..Default::default()
        }),
        array: gpu.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        }),
    }
}

/// A sampler as nn::vfx's TextureSampler::Initialize builds it: wrap 0 mirror, 1 repeat,
/// 2 clamp to edge, 3 mirror once (clamped); a non-zero filter byte means point sampling.
pub fn make_sampler(
    device: &wgpu::Device,
    wrap_u: u8,
    wrap_v: u8,
    filter: u8,
    max_lod: f32,
) -> wgpu::Sampler {
    let mode = |m: u8| match m {
        1 => wgpu::AddressMode::Repeat,
        2 | 3 => wgpu::AddressMode::ClampToEdge,
        _ => wgpu::AddressMode::MirrorRepeat,
    };
    let f = if filter != 0 {
        wgpu::FilterMode::Nearest
    } else {
        wgpu::FilterMode::Linear
    };
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("vfx sampler"),
        address_mode_u: mode(wrap_u),
        address_mode_v: mode(wrap_v),
        address_mode_w: wgpu::AddressMode::MirrorRepeat,
        mag_filter: f,
        min_filter: f,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        lod_max_clamp: max_lod.clamp(0.0, 32.0),
        ..Default::default()
    })
}

/// The error an error scope caught. On native wgpu the scope's future is ready as soon as it is
/// popped, so it is polled once here rather than awaited.
type PipelineKey = (usize, RenderState, bool);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn caught(scope: wgpu::ErrorScopeGuard) -> Option<wgpu::Error> {
    let mut future = std::pin::pin!(scope.pop());
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match std::future::Future::poll(future.as_mut(), &mut context) {
        std::task::Poll::Ready(error) => error,
        std::task::Poll::Pending => None,
    }
}

/// Linear to sRGB, per channel, for targets that do not encode on write.
const SRGB_ENCODE: &str = "
fn srgb_encode(c: vec3<f32>) -> vec3<f32> {
    let l = max(c, vec3<f32>(0.0));
    return select(1.055 * pow(l, vec3<f32>(1.0 / 2.4)) - 0.055, 12.92 * l, l <= vec3<f32>(0.0031308));
}
";
