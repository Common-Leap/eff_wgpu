//! The shader container an effect file carries (`grsc` / BNSH): one program per variation,
//! each with Maxwell code per stage and the names and slots of what the code reads and writes.

use std::fmt;

pub const STAGE_VERTEX: usize = 0;
pub const STAGE_FRAGMENT: usize = 4;
pub const STAGE_COMPUTE: usize = 5;

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

fn err(message: impl Into<String>) -> Error {
    Error(message.into())
}

/// One named resource and the hardware slot the code uses for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub slot: i32,
}

#[derive(Debug, Clone, Default)]
pub struct Reflection {
    pub inputs: Vec<Binding>,
    pub outputs: Vec<Binding>,
    pub samplers: Vec<Binding>,
    pub constant_buffers: Vec<Binding>,
}

#[derive(Debug, Clone, Default)]
pub struct Stage {
    /// Maxwell code as the GPU runs it, after the 0x30-byte NVN prefix when present: a shader
    /// program header (SPH, 0x50 bytes) for graphics stages, then the instructions.
    pub code: Vec<u8>,
    pub reflection: Reflection,
}

#[derive(Debug, Clone, Default)]
pub struct Program {
    /// Indexed by stage: 0 vertex, 1 hull, 2 domain, 3 geometry, 4 fragment, 5 compute.
    pub stages: [Option<Stage>; 6],
}

#[derive(Debug, Clone, Default)]
pub struct ShaderFile {
    pub programs: Vec<Program>,
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn u16(&self, at: usize) -> Result<u16, Error> {
        self.0
            .get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .ok_or_else(|| err(format!("read past end at {at:#x}")))
    }
    fn u32(&self, at: usize) -> Result<u32, Error> {
        self.0
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .ok_or_else(|| err(format!("read past end at {at:#x}")))
    }
    fn u64(&self, at: usize) -> Result<u64, Error> {
        self.0
            .get(at..at + 8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .ok_or_else(|| err(format!("read past end at {at:#x}")))
    }
    fn bytes(&self, at: usize, len: usize) -> Result<&[u8], Error> {
        self.0
            .get(at..at + len)
            .ok_or_else(|| err(format!("{len} bytes past end at {at:#x}")))
    }
    fn string(&self, at: usize) -> Result<String, Error> {
        let len = self.u16(at)? as usize;
        Ok(String::from_utf8_lossy(self.bytes(at + 2, len)?).into_owned())
    }
    fn dictionary(&self, at: usize) -> Result<Vec<String>, Error> {
        if at == 0 {
            return Ok(Vec::new());
        }
        if self.bytes(at, 4)? != b"_DIC" {
            return Err(err(format!("no dictionary at {at:#x}")));
        }
        let count = self.u32(at + 4)? as usize;
        // Entry 0 is the root node; names follow, 16 bytes each.
        (1..=count)
            .map(|index| self.string(self.u64(at + 8 + index * 16 + 8)? as usize))
            .collect()
    }
}

impl ShaderFile {
    pub fn read(data: &[u8]) -> Result<Self, Error> {
        let r = Reader(data);
        if r.bytes(0, 4)? != b"BNSH" {
            return Err(err("not a BNSH file"));
        }
        let block = r.u16(0x16)? as usize;
        if r.bytes(block, 4)? != b"grsc" {
            return Err(err("no grsc block"));
        }
        let count = r.u32(block + 0x1c)? as usize;
        let first = r.u64(block + 0x20)? as usize;
        let programs = (0..count)
            .map(|index| read_program(&r, first + index * 64))
            .collect::<Result<_, _>>()?;
        Ok(Self { programs })
    }
}

fn read_program(r: &Reader, variation: usize) -> Result<Program, Error> {
    let at = r.u64(variation + 16)? as usize;
    let mut program = Program::default();
    let reflection_table = r.u64(at + 8 + 6 * 8 + 40 + 24)? as usize;
    for stage in 0..6 {
        let code_at = r.u64(at + 8 + stage * 8)? as usize;
        if code_at == 0 {
            continue;
        }
        let code_offset = r.u64(code_at + 16)? as usize;
        let code_size = r.u32(code_at + 24)? as usize;
        let mut code = r.bytes(code_offset, code_size)?;
        // Some programs carry a 0x30-byte prefix (it starts 0x12345678) ahead of the header.
        if code.len() > 0x30 && !looks_like_sph(code) && looks_like_sph(&code[0x30..]) {
            code = &code[0x30..];
        }
        let reflection = if reflection_table != 0 {
            let refl_at = r.u64(reflection_table + stage * 8)? as usize;
            if refl_at != 0 {
                read_reflection(r, refl_at)?
            } else {
                Reflection::default()
            }
        } else {
            Reflection::default()
        };
        program.stages[stage] = Some(Stage {
            code: code.to_vec(),
            reflection,
        });
    }
    Ok(program)
}

/// SPH word 0: bits 0..5 = header type (1 vertex/tess/geometry, 2 pixel), bits 5..10 = version
/// (3), bits 10..14 = shader type (1..5).
fn looks_like_sph(code: &[u8]) -> bool {
    if code.len() < 0x50 {
        return false;
    }
    let w = u32::from_le_bytes(code[0..4].try_into().unwrap());
    let sph_type = w & 0x1f;
    let version = (w >> 5) & 0x1f;
    let shader_type = (w >> 10) & 0xf;
    (sph_type == 1 || sph_type == 2) && version == 3 && (1..=5).contains(&shader_type)
}

fn read_reflection(r: &Reader, at: usize) -> Result<Reflection, Error> {
    let inputs = r.dictionary(r.u64(at)? as usize)?;
    let outputs = r.dictionary(r.u64(at + 8)? as usize)?;
    let samplers = r.dictionary(r.u64(at + 16)? as usize)?;
    let cbufs = r.dictionary(r.u64(at + 24)? as usize)?;
    let output_idx = r.u32(at + 40)? as usize;
    let sampler_idx = r.u32(at + 44)? as usize;
    let cbuf_idx = r.u32(at + 48)? as usize;
    let slot_offset = r.u32(at + 56)? as usize;
    let slot_count = r.u32(at + 76)? as usize;
    let slots = (0..slot_count)
        .map(|i| r.u32(slot_offset + i * 4).map(|v| v as i32))
        .collect::<Result<Vec<_>, _>>()?;
    let bind = |names: Vec<String>, first: usize| {
        names
            .into_iter()
            .enumerate()
            .map(|(i, name)| Binding {
                name,
                slot: slots.get(first + i).copied().unwrap_or(-1),
            })
            .collect()
    };
    Ok(Reflection {
        inputs: bind(inputs, 0),
        outputs: bind(outputs, output_idx),
        samplers: bind(samplers, sampler_idx),
        constant_buffers: bind(cbufs, cbuf_idx),
    })
}
