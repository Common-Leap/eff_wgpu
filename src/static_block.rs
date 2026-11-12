//! The constant blocks an emitter resource uploads once, built the way Smash builds them.
//!
//! Smash 0x5dc0 (EmitterResource setup) fixes up the emitter's own data in place and then copies
//! its first 0x750 bytes into the static uniform block; 0x6f10 packs the shader flag words at
//! +0x50/+0x54. The field block (0x120 bytes) is filled from the field sections in the same
//! function. Offsets below are into the emitter data (`ptcl::Emitter::data`).

use crate::ptcl::Emitter;

pub const STATIC_SIZE: usize = 0x750;
pub const FIELD_SIZE: usize = 0x120;

struct Bytes<'a>(&'a mut [u8]);

impl Bytes<'_> {
    fn u8(&self, at: usize) -> u8 {
        self.0[at]
    }
    fn i32(&self, at: usize) -> i32 {
        i32::from_le_bytes(self.0[at..at + 4].try_into().unwrap())
    }
    fn f32(&self, at: usize) -> f32 {
        f32::from_le_bytes(self.0[at..at + 4].try_into().unwrap())
    }
    fn set_u32(&mut self, at: usize, v: u32) {
        self.0[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    fn set_f32(&mut self, at: usize, v: f32) {
        self.set_u32(at, v.to_bits());
    }
    fn copy(&mut self, from: usize, to: usize, len: usize) {
        self.0.copy_within(from..from + len, to);
    }
}

fn attr<'a>(e: &'a Emitter, magic: &[u8; 4]) -> Option<&'a [u8]> {
    e.attributes
        .iter()
        .find(|(m, _)| m == magic)
        .map(|(_, d)| d.as_slice())
}

/// The emitter data after the runtime's load-time fixes. `texture_layers` answers how many
/// layers the texture with a given hash has (the runtime reads it for array-pattern samplers);
/// `primitive_vertices` is the emitter's primitive vertex count, when it draws one.
pub fn fixed_emitter_data(
    e: &Emitter,
    texture_layers: &dyn Fn(u64) -> Option<i32>,
    primitive_vertices: Option<i32>,
) -> Vec<u8> {
    let mut data = e.data.clone();
    if data.len() < 0xac0 {
        data.resize(0xac0, 0);
    }
    let mut b = Bytes(&mut data);
    // Constant colour/alpha when the matching channel is not animated.
    if b.u8(0x99c) == 0 {
        b.copy(0x9a0, 0x3c0, 12);
    }
    if b.u8(0x99e) == 0 {
        b.copy(0x9ac, 0x440, 4);
    }
    if b.u8(0x99d) == 0 {
        b.copy(0x9b0, 0x4c0, 12);
    }
    if b.u8(0x99f) == 0 {
        b.copy(0x9bc, 0x540, 4);
    }
    // Texture repeat counts by wrap mode (rodata 0x36319d0 / 0x36319e0).
    const REPEAT_U: [f32; 4] = [1.0, 2.0, 1.0, 2.0];
    const REPEAT_V: [f32; 4] = [1.0, 1.0, 2.0, 2.0];
    for (flag, dst) in [(0xa54, 0x300), (0xa64, 0x350), (0xa74, 0x3a0)] {
        let mode = b.u8(flag) as i8;
        if (0..4).contains(&mode) {
            b.set_f32(dst, REPEAT_U[mode as usize]);
            b.set_f32(dst + 4, REPEAT_V[mode as usize]);
        }
    }
    // Per-sampler UV animation blocks cleared when the sampler doesn't use them.
    for (flag, base) in [(0xa51, 0x2c0), (0xa61, 0x310), (0xa71, 0x360)] {
        if b.u8(flag) == 0 {
            for at in [8, 0xc, 0, 4, 0x10, 0x14] {
                b.set_u32(base + at, 0);
            }
        }
    }
    for (flag, base) in [(0xa52, 0x2f0), (0xa62, 0x340), (0xa72, 0x390)] {
        if b.u8(flag) == 0 {
            for at in [0, 4, 8] {
                b.set_u32(base + at, 0);
            }
        }
    }
    for (flag, base) in [(0xa53, 0x2d8), (0xa63, 0x328), (0xa73, 0x378)] {
        if b.u8(flag) == 0 {
            b.set_f32(base + 8, 1.0);
            b.set_f32(base + 0xc, 1.0);
            for at in [0, 4, 0x10, 0x14] {
                b.set_u32(base + at, 0);
            }
        }
    }
    for (i, flag) in [0x8b0, 0x8b1, 0x8b2].into_iter().enumerate() {
        if b.u8(flag) == 0 {
            for row in [0x700, 0x710, 0x720, 0x730] {
                b.set_u32(row + 4 * i, 0);
            }
        }
    }
    for (flag, hash_at, dst) in [
        (0xa50, 0x9f0, 0x110),
        (0xa60, 0xa10, 0x1a0),
        (0xa70, 0xa30, 0x230),
    ] {
        if b.u8(flag) == 5 {
            let hash = u64::from_le_bytes(b.0[hash_at..hash_at + 8].try_into().unwrap());
            if let Some(layers) = texture_layers(hash) {
                b.set_f32(dst, layers as f32);
            }
        }
    }
    // 8-key animations: keys past the last are copies of it, each with its index added to the
    // key's fourth (time) value.
    for (count_at, keys) in [
        (0x60, 0x3c0),
        (0x68, 0x4c0),
        (0x64, 0x440),
        (0x6c, 0x540),
        (0x70, 0x600),
        (0x74, 0x680),
    ] {
        let n = b.i32(count_at);
        if (1..=7).contains(&n) {
            let last = keys + (n as usize - 1) * 16;
            for k in n..8 {
                let dst = keys + k as usize * 16;
                b.copy(last, dst, 16);
                let t = b.f32(dst + 12) + k as f32;
                b.set_f32(dst + 12, t);
            }
        }
    }
    for i in 0..5 {
        let v = if b.u8(0x8d8 + i) != 0 {
            b.i32(0x8e4 + 4 * i) as f32
        } else {
            0.0
        };
        b.set_f32(0x80 + 4 * i, v);
        b.set_f32(0x94 + 4 * i, if b.u8(0x8dd + i) != 0 { 1.0 } else { 0.0 });
    }
    // Emission count by shape (jump table at 0x67a4).
    let shape = b.u8(0x838);
    if shape > 15 || (1u32 << shape) & 0xa064 == 0 {
        b.set_u32(0x874, u32::MAX);
    }
    if (2..=15).contains(&shape) && b.i32(0x874) == 0 {
        const COUNTS: [f32; 8] = [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 32.0];
        match shape {
            2 | 13 => b.set_f32(0x800, 1.0),
            5 => b.set_f32(
                0x800,
                COUNTS.get(b.u8(0x83c) as usize).copied().unwrap_or(0.0),
            ),
            6 => b.set_f32(0x800, b.u8(0x83d) as f32),
            15 => {
                if let Some(n) = primitive_vertices {
                    b.set_f32(0x800, n as f32);
                }
            }
            _ => {}
        }
    }
    let (flags0, flags1) = shader_flags(&mut b, e);
    b.set_u32(0x50, flags0);
    b.set_u32(0x54, flags1);
    if let Some(fcsf) = attr(e, b"FCSF") {
        if fcsf.len() >= 4 {
            b.copy_from(fcsf, 0, 0x5c, 4);
        }
    }
    data
}

impl Bytes<'_> {
    fn copy_from(&mut self, src: &[u8], from: usize, to: usize, len: usize) {
        self.0[to..to + len].copy_from_slice(&src[from..from + len]);
    }
}

/// Smash 0x6f10: the two shader flag words.
fn shader_flags(b: &mut Bytes, e: &Emitter) -> (u32, u32) {
    let mut f0 = 0u32;
    let mut f1 = 0u32;
    if b.u8(0x7f1) != 0 {
        f0 |= 8;
    }
    let billboard = b.u8(0x9e7) >> 4;
    if billboard < 3 {
        f0 |= [1, 2, 4][billboard as usize];
    }
    for (i, (kind_at, count_at, list_at, value_at)) in [
        (0xa50, 0x118, 0x120, 0x110),
        (0xa60, 0x1a8, 0x1b0, 0x1a0),
        (0xa70, 0x238, 0x240, 0x230),
    ]
    .into_iter()
    .enumerate()
    {
        let shift = 4 * i as u32;
        match b.u8(kind_at) {
            1 => f0 |= 0x10 << shift,
            2 => f0 |= 0x20 << shift,
            3 => f0 |= 0x40 << shift,
            4 => {
                let count = b.f32(count_at);
                let mut k = 0;
                while (k as f32) < count {
                    b.set_u32(list_at + 4 * k, k as u32);
                    k += 1;
                }
                b.set_f32(value_at, count);
                f0 |= 0x80 << shift;
            }
            _ => {}
        }
    }
    for (at, bit) in [
        (0x8ad, 0x10000),
        (0x8ae, 0x20000),
        (0x8af, 0x40000),
        (0xa55, 0x80000),
        (0xa56, 0x100000),
        (0xa65, 0x200000),
        (0xa66, 0x400000),
        (0xa75, 0x800000),
        (0xa76, 0x1000000),
        (0xa57, 0x2000000),
        (0xa67, 0x4000000),
        (0xa77, 0x8000000),
        (0x8b3, 0x10000000),
    ] {
        if b.u8(at) != 0 {
            f0 |= bit;
        }
    }
    match b.u8(0x8ac) {
        1 => f0 |= 0x40000000,
        0 => f0 |= 0x20000000,
        _ => {}
    }
    if b.u8(0x8b4) != 0 {
        f1 |= 1;
    }
    for (magic, bit) in [
        (b"FRND", 2),
        (b"FRN1", 0x100),
        (b"FPAD", 4),
        (b"FMAG", 8),
        (b"FCOV", 0x10),
        (b"FSPN", 0x20),
        (b"FCOL", 0x40),
        (b"FCLN", 0x80),
    ] {
        if attr(e, magic).is_some() {
            f1 |= bit;
        }
    }
    let colour_mode = b.u8(0x753) as i8;
    if (0..3).contains(&colour_mode) {
        f1 |= [0x200, 0x800, 0x400][colour_mode as usize];
    }
    (f0, f1)
}

/// The static uniform block: the first 0x750 bytes of the fixed emitter data.
pub fn static_block(fixed: &[u8]) -> Vec<u8> {
    fixed[..STATIC_SIZE].to_vec()
}

/// The 0x120-byte field block (present when the emitter has any field section).
pub fn field_block(e: &Emitter) -> Option<Vec<u8>> {
    let any = [
        b"FRND", b"FRN1", b"FPAD", b"FMAG", b"FCOV", b"FSPN", b"FCOL", b"FCLN", b"FCSF",
    ]
    .iter()
    .any(|m| attr(e, m).is_some());
    if !any {
        return None;
    }
    let mut out = vec![0u8; FIELD_SIZE];
    let mut w =
        |slot: usize, bits: u32| out[slot * 4..slot * 4 + 4].copy_from_slice(&bits.to_le_bytes());
    let word = |d: &[u8], at: usize| u32::from_le_bytes(d[at..at + 4].try_into().unwrap());
    let int_as_float = |d: &[u8], at: usize| (word(d, at) as i32 as f32).to_bits();
    let byte_as_float = |d: &[u8], at: usize| (d[at] as f32).to_bits();
    if let Some(d) = attr(e, b"FRND").filter(|d| d.len() >= 0x3c) {
        for (slot, at) in [
            (0, 4),
            (1, 8),
            (2, 0xc),
            (4, 0x2c),
            (5, 0x30),
            (6, 0x34),
            (7, 0x38),
            (8, 0x1c),
            (9, 0x20),
            (10, 0x24),
            (11, 0x28),
            (16, 0x14),
            (17, 0x18),
        ] {
            w(slot, word(d, at));
        }
        w(3, int_as_float(d, 0x10));
        for (slot, at) in [(12, 0), (13, 1), (14, 2)] {
            w(slot, byte_as_float(d, at));
        }
    }
    if let Some(d) = attr(e, b"FRN1").filter(|d| d.len() >= 0x10) {
        for (slot, at) in [(0x14, 0), (0x15, 4), (0x16, 8)] {
            w(slot, word(d, at));
        }
        w(0x17, int_as_float(d, 0xc));
    }
    if let Some(d) = attr(e, b"FPAD").filter(|d| d.len() >= 0x10) {
        for (slot, at) in [(0x18, 4), (0x19, 8), (0x1a, 0xc)] {
            w(slot, word(d, at));
        }
        w(0x1b, byte_as_float(d, 0));
    }
    if let Some(d) = attr(e, b"FMAG").filter(|d| d.len() >= 0x14) {
        for (slot, at) in [(0x1c, 8), (0x1d, 0xc), (0x1e, 0x10), (0x1f, 4)] {
            w(slot, word(d, at));
        }
        for (slot, at) in [(0x20, 0), (0x21, 1), (0x22, 2), (0x23, 3)] {
            w(slot, byte_as_float(d, at));
        }
    }
    if let Some(d) = attr(e, b"FCOV").filter(|d| d.len() >= 0x14) {
        for (slot, at) in [(0x24, 4), (0x25, 8), (0x26, 0xc), (0x27, 0x10)] {
            w(slot, word(d, at));
        }
        w(0x28, byte_as_float(d, 0));
    }
    if let Some(d) = attr(e, b"FSPN").filter(|d| d.len() >= 0xc) {
        w(0x2c, word(d, 0));
        w(0x2d, word(d, 8));
        w(0x2e, (word(d, 4) as f32).to_bits());
    }
    if let Some(d) = attr(e, b"FCOL").filter(|d| d.len() >= 0x14) {
        w(0x30, (d[1] as f32 + ((d[0] as u32) << 1) as f32).to_bits());
        w(0x31, word(d, 8));
        w(0x32, word(d, 0x10));
        w(0x33, word(d, 4));
    }
    if let Some(d) = attr(e, b"FCLN").filter(|d| d.len() >= 0x24) {
        for (slot, at) in [
            (0x34, 0x10),
            (0x35, 0x14),
            (0x36, 0x18),
            (0x37, 0x1c),
            (0x38, 4),
            (0x39, 8),
            (0x3a, 0xc),
            (0x3b, 0x20),
        ] {
            w(slot, word(d, at));
        }
        w(0x3c, byte_as_float(d, 1));
        w(0x3d, byte_as_float(d, 2));
    }
    if let Some(d) = attr(e, b"FCSF").filter(|d| d.len() >= 0x24) {
        for (k, at) in (4..0x24).step_by(4).enumerate() {
            w(0x40 + k, word(d, at));
        }
    }
    Some(out)
}

/// The custom shader block (`CSDP` section data, size from the section, Smash +0x300/+0x308).
pub fn custom_block(e: &Emitter) -> Option<Vec<u8>> {
    attr(e, b"CSDP").map(|d| d.to_vec())
}
