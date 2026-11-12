//! The raw section tree of an effect's VFXB/ptcl data, walked the way the runtime walks it
//! (`Resource::InitializeEmitterSetResource`, Smash 0x15600). Emitters keep their bytes as the
//! file has them: the runtime uploads the first 0x750 bytes of an emitter, after a few fixes,
//! as the shader's static constant block.

pub const EMTR: [u8; 4] = *b"EMTR";

#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub at: usize,
    pub magic: [u8; 4],
    pub size: u32,
    pub child: u32,
    pub next: u32,
    pub attr: u32,
    pub data: u32,
    pub child_count: u16,
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    b.get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
}

pub fn header(b: &[u8], at: usize) -> Option<Header> {
    Some(Header {
        at,
        magic: b.get(at..at + 4)?.try_into().ok()?,
        size: u32_at(b, at + 4)?,
        child: u32_at(b, at + 8)?,
        next: u32_at(b, at + 12)?,
        attr: u32_at(b, at + 16)?,
        data: u32_at(b, at + 20)?,
        child_count: u16::from_le_bytes(b.get(at + 28..at + 30)?.try_into().ok()?),
    })
}

impl Header {
    fn link(&self, offset: u32) -> Option<usize> {
        (offset != u32::MAX).then(|| self.at + offset as usize)
    }
    pub fn first_child(&self) -> Option<usize> {
        self.link(self.child)
    }
    pub fn next_sibling(&self) -> Option<usize> {
        self.link(self.next)
    }
    pub fn first_attr(&self) -> Option<usize> {
        self.link(self.attr)
    }
    pub fn data_at(&self) -> usize {
        self.at + self.data as usize
    }
}

/// Every section in a sibling chain starting at `first`.
pub fn chain(b: &[u8], first: Option<usize>) -> Vec<Header> {
    let mut out = Vec::new();
    let mut at = first;
    while let Some(here) = at {
        let Some(h) = header(b, here) else { break };
        at = h.next_sibling();
        out.push(h);
        if out.len() > 100_000 {
            break;
        }
    }
    out
}

/// One emitter: its data bytes and the sections hung off it.
#[derive(Debug, Clone)]
pub struct Emitter {
    /// From the emitter's data offset to the end of its section.
    pub data: Vec<u8>,
    /// Attribute sections (animations `EA__`, fields, ...) by magic, with their data bytes.
    pub attributes: Vec<([u8; 4], Vec<u8>)>,
    /// Child emitters.
    pub children: Vec<Emitter>,
}

#[derive(Debug, Clone)]
pub struct EmitterSet {
    pub emitters: Vec<Emitter>,
}

/// The emitter sets in file order (the order an eff's `EmitterSet` index refers to).
pub fn emitter_sets(ptcl: &[u8]) -> Vec<EmitterSet> {
    let block = u16::from_le_bytes([ptcl[0x16], ptcl[0x17]]) as usize;
    let top = chain(ptcl, Some(block));
    let Some(esta) = top.iter().find(|h| &h.magic == b"ESTA") else {
        return Vec::new();
    };
    chain(ptcl, esta.first_child())
        .into_iter()
        .filter(|h| &h.magic == b"ESET")
        .map(|set| EmitterSet {
            emitters: chain(ptcl, set.first_child())
                .into_iter()
                .filter(|h| h.magic == EMTR)
                .map(|h| emitter(ptcl, h))
                .collect(),
        })
        .collect()
}

fn emitter(ptcl: &[u8], h: Header) -> Emitter {
    let end = (h.at + h.size as usize).min(ptcl.len());
    let start = h.data_at().min(end);
    Emitter {
        data: ptcl[start..end].to_vec(),
        attributes: chain(ptcl, h.first_attr())
            .into_iter()
            .map(|a| {
                let end = (a.at + a.size as usize).min(ptcl.len());
                (a.magic, ptcl[a.data_at().min(end)..end].to_vec())
            })
            .collect(),
        children: chain(ptcl, h.first_child())
            .into_iter()
            .filter(|c| c.magic == EMTR)
            .map(|c| emitter(ptcl, c))
            .collect(),
    }
}

/// Top-level sections by magic (GRTF textures, PRMA primitives, GRSN shaders, ...).
pub fn top_level(ptcl: &[u8]) -> Vec<Header> {
    let block = u16::from_le_bytes([ptcl[0x16], ptcl[0x17]]) as usize;
    chain(ptcl, Some(block))
}
