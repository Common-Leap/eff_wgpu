//! Giving a compiled shader more animation keys than it was made for.
//!
//! An effect's vertex shader animates a particle's colours, opacities and size from tables of up
//! to eight keys in the emitter's static block, but each was compiled for the count its part
//! shipped with: it reads exactly that many keys (one, for a constant), whatever the file's count
//! says. [`animate`] adds code to the end of such a shader that works out a table's value from
//! all eight keys, jumps to it before anything else, and points every read of the table's keys at
//! the result. Every key the old code blends then holds the same value, so whatever it does with
//! them comes to that value.
//!
//! Keys past the last need no case of their own: the game fills them with copies of it as it
//! loads the emitter (`static_block::fixed_emitter_data`).

use std::collections::BTreeSet;

use super::decode::{decode, decode_program, Inst, Op, Src};
use crate::bnsh::Binding;

/// The constant bank of the emitter's static block.
const STATIC: u8 = 9;
/// The constant bank of the emitter's dynamic block; its time is at +0x20.
const DYNAMIC: u8 = 10;
const TIME: u32 = 0x20;
/// Slots in a key table, 16 bytes each with the time last.
const KEYS: u32 = 8;

/// One animated value of a part.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Table {
    Colour0,
    Alpha0,
    Colour1,
    Alpha1,
    Scale,
}

impl Table {
    pub const ALL: [Table; 5] = [
        Table::Colour0,
        Table::Alpha0,
        Table::Colour1,
        Table::Alpha1,
        Table::Scale,
    ];

    /// Where its keys are in the static block.
    fn keys(self) -> u32 {
        match self {
            Table::Colour0 => 0x3c0,
            Table::Alpha0 => 0x440,
            Table::Colour1 => 0x4c0,
            Table::Alpha1 => 0x540,
            Table::Scale => 0x600,
        }
    }

    /// How many numbers a key has besides its time.
    fn width(self) -> u32 {
        match self {
            Table::Alpha0 | Table::Alpha1 => 1,
            _ => 3,
        }
    }

    /// Its loop length in frames (0 when it plays once), and its random start switch.
    fn looping(self) -> (u32, u32) {
        let at = 4 * self as u32;
        (0x80 + at, 0x94 + at)
    }

    /// The key and number a static block byte is, when it is one of this table's.
    fn slot(self, offset: u32) -> Option<(u32, u32)> {
        let at = offset.checked_sub(self.keys())?;
        let (key, number) = (at / 16, at % 16 / 4);
        (key < KEYS && number <= 3).then_some((key, number))
    }
}

/// A stage's instructions, read from its Maxwell code after the header.
fn instructions(code: &[u8]) -> Vec<Inst> {
    decode_program(code.get(0x50..).unwrap_or_default())
}

fn sources(op: &Op) -> Vec<Src> {
    match *op {
        Op::Mov { s, .. }
        | Op::Rro { s, .. }
        | Op::I2f { s, .. }
        | Op::F2i { s, .. }
        | Op::F2f { s, .. }
        | Op::I2i { s, .. } => vec![s],
        Op::Sel { b, .. }
        | Op::Fmul { b, .. }
        | Op::Fadd { b, .. }
        | Op::Fmnmx { b, .. }
        | Op::Fset { b, .. }
        | Op::Fsetp { b, .. }
        | Op::Isetp { b, .. }
        | Op::Iset { b, .. }
        | Op::Iadd { b, .. }
        | Op::Iadd3 { b, .. }
        | Op::Iscadd { b, .. }
        | Op::Lop { b, .. }
        | Op::Lop3 { b, .. }
        | Op::Shl { b, .. }
        | Op::Shr { b, .. } => vec![b],
        Op::Ffma { b, c, .. } | Op::Xmad { b, c, .. } => vec![b, c],
        Op::R2p { mask, .. } => vec![mask],
        _ => Vec::new(),
    }
}

/// The constant words a stage reads from `bank`, by byte offset. `code` is the stage's code as
/// [`crate::bnsh::Stage`] keeps it.
pub fn words_read(code: &[u8], bank: u8) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    for inst in instructions(code) {
        for src in sources(&inst.op) {
            if let Src::Cbuf { bank: b, offset } = src {
                if b == bank {
                    out.insert(offset);
                }
            }
        }
        if let Op::Ldc {
            bank: b,
            offset,
            bytes,
            ..
        } = inst.op
        {
            if b == bank {
                out.extend((0..u32::from(bytes).div_ceil(4)).map(|k| (offset as u32) + 4 * k));
            }
        }
    }
    out
}

/// How a vertex stage uses each table's keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeysRead {
    /// How many keys it reads: none, one (a constant, or the only key), or as many as the table
    /// had when it was compiled.
    pub keys: u32,
    /// Whether it picks one key for each particle instead of running through them.
    pub picks: bool,
}

/// What [`animate`] can do with a table, from the keys a vertex stage reads of it.
pub fn keys_read(code: &[u8]) -> [KeysRead; 5] {
    let mut out = [KeysRead::default(); 5];
    for inst in instructions(code) {
        let indexed = match inst.op {
            Op::Ldc {
                bank: STATIC,
                offset,
                ..
            } => Some(offset as u32),
            _ => None,
        };
        let direct = sources(&inst.op).into_iter().filter_map(|s| match s {
            Src::Cbuf {
                bank: STATIC,
                offset,
            } => Some(offset),
            _ => None,
        });
        let direct: Vec<u32> = direct.collect();
        for table in Table::ALL {
            let read = &mut out[table as usize];
            if indexed.is_some_and(|at| table.slot(at).is_some()) {
                read.picks = true;
            }
            for &at in &direct {
                if let Some((key, _)) = table.slot(at) {
                    read.keys = read.keys.max(key + 1);
                }
            }
        }
    }
    out
}

/// A stage as a BNSH holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageCode {
    /// The control section.
    pub control: Vec<u8>,
    /// The code section, with its 0x30-byte prefix.
    pub code: Vec<u8>,
}

/// Control section words: the size of the header and instructions, the literals' size and place,
/// the whole code section's size, the prefix's size, and the registers the program takes.
const CODE_SIZE: usize = 0x6f8;
const LITERAL_SIZE: usize = 0x6fc;
const LITERAL_AT: usize = 0x700;
const TOTAL_SIZE: usize = 0x704;
const PREFIX_SIZE: usize = 0x708;
const REGISTERS: usize = 0x70c;

fn page(size: usize) -> usize {
    size.next_multiple_of(0x100)
}

/// A register the hardware reads as zero, and the predicate that is always true.
const RZ: u8 = 255;
const PT: u8 = 7;

/// The added code, as instruction words with their scheduling.
struct Stub {
    words: Vec<(u64, u32)>,
    next: u8,
    /// The barrier the last instruction set, which the next one waits for.
    pending: bool,
}

/// Scheduling for an instruction: cycles before the next, whether it sets barrier 0, whether it
/// waits for it. Never yields, reuses no operand.
fn sched(stall: u32, sets: bool, waits: bool) -> u32 {
    let write = if sets { 0 } else { 7 };
    stall | 1 << 4 | write << 5 | 7 << 8 | u32::from(waits) << 11
}

fn cbuf_bits(bank: u8, offset: u32) -> u64 {
    u64::from(offset / 4) << 20 | u64::from(bank) << 34
}

impl Stub {
    fn register(&mut self) -> u8 {
        self.next += 1;
        self.next - 1
    }

    fn push(&mut self, word: u64, stall: u32, sets: bool) {
        let waits = std::mem::replace(&mut self.pending, sets);
        self.words.push((word, sched(stall, sets, waits)));
    }

    /// An instruction whose result is ready a fixed six cycles on.
    fn fixed(&mut self, word: u64) {
        self.push(word, 6, false);
    }

    /// One whose result takes as long as it takes.
    fn variable(&mut self, word: u64) {
        self.push(word, 15, true);
    }

    fn attribute(&mut self, d: u8, offset: u32) {
        self.variable(0xefd8_7f80_0007_ff00 | u64::from(offset) << 20 | u64::from(d));
    }

    fn mov(&mut self, d: u8, bank: u8, offset: u32) {
        self.fixed(0x4c98_0780_0007_0000 | cbuf_bits(bank, offset) | u64::from(d));
    }

    /// `d = c[bank][offset] - a`, `d = a - c[bank][offset]` and `d = a - b`.
    fn sub_from_constant(&mut self, d: u8, a: u8, bank: u8, offset: u32) {
        self.fixed(
            0x4c59_1000_0007_0000 | cbuf_bits(bank, offset) | u64::from(a) << 8 | u64::from(d),
        );
    }

    fn sub_constant(&mut self, d: u8, a: u8, bank: u8, offset: u32) {
        self.fixed(
            0x4c58_3000_0007_0000 | cbuf_bits(bank, offset) | u64::from(a) << 8 | u64::from(d),
        );
    }

    fn sub(&mut self, d: u8, a: u8, b: u8, guard: u64) {
        self.fixed(
            0x5c58_3000_0000_0000
                | guard << 16
                | u64::from(b) << 20
                | u64::from(a) << 8
                | u64::from(d),
        );
    }

    fn mul(&mut self, d: u8, a: u8, b: u8, guard: u64, saturate: bool) {
        self.fixed(
            0x5c68_1000_0000_0000
                | u64::from(saturate) << 50
                | guard << 16
                | u64::from(b) << 20
                | u64::from(a) << 8
                | u64::from(d),
        );
    }

    fn mul_constant(&mut self, d: u8, a: u8, bank: u8, offset: u32) {
        self.fixed(
            0x4c68_1000_0007_0000 | cbuf_bits(bank, offset) | u64::from(a) << 8 | u64::from(d),
        );
    }

    /// `d = a * b + c`, with `b` a constant in the second.
    fn fma(&mut self, d: u8, a: u8, b: u8, c: u8) {
        self.fixed(
            0x59a0_0000_0007_0000
                | u64::from(c) << 39
                | u64::from(b) << 20
                | u64::from(a) << 8
                | u64::from(d),
        );
    }

    fn fma_constant(&mut self, d: u8, a: u8, bank: u8, offset: u32, c: u8) {
        self.fixed(
            0x49a0_0000_0007_0000
                | u64::from(c) << 39
                | cbuf_bits(bank, offset)
                | u64::from(a) << 8
                | u64::from(d),
        );
    }

    fn reciprocal(&mut self, d: u8, a: u8) {
        self.variable(0x5080_0000_0047_0000 | u64::from(a) << 8 | u64::from(d));
    }

    fn floor(&mut self, d: u8, a: u8) {
        self.variable(0x5ca8_1480_0007_0a00 | u64::from(a) << 20 | u64::from(d));
    }

    /// `d = a` with its fraction cut off.
    fn truncate(&mut self, d: u8, a: u8) {
        self.variable(0x5cb0_1180_0007_1a00 | u64::from(a) << 20 | u64::from(d));
        self.variable(0x5cb8_0000_0007_2a00 | u64::from(d) << 20 | u64::from(d));
    }

    /// P0 = 0 < c[bank][offset].
    fn positive(&mut self, bank: u8, offset: u32) {
        self.push(0x4bb1_8380_0007_ff07 | cbuf_bits(bank, offset), 13, false);
    }
}

/// The attributes the added code reads, by their byte offset.
struct Attributes {
    born: u32,
    life: u32,
    random: Option<u32>,
}

impl Attributes {
    fn of(inputs: &[Binding]) -> Option<Self> {
        let slot = |name: &str| {
            let slot = inputs.iter().find(|i| i.name == name)?.slot;
            u32::try_from(slot).ok().map(|slot| 0x80 + 16 * slot)
        };
        Some(Self {
            born: slot("sysLocalVecAttr")? + 12,
            life: slot("sysLocalPosAttr")? + 12,
            random: slot("sysRandomAttr"),
        })
    }
}

/// The stage with `tables` animated over all eight of their keys, or why it can't be.
pub fn animate(
    stage: &StageCode,
    inputs: &[Binding],
    tables: &[Table],
) -> Result<StageCode, String> {
    if tables.is_empty() {
        return Ok(stage.clone());
    }
    let word = |at: usize| {
        stage
            .control
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
            .ok_or_else(|| "its control section is short".to_owned())
    };
    let (size, literal_size, literal_at, total, prefix, registers) = (
        word(CODE_SIZE)?,
        word(LITERAL_SIZE)?,
        word(LITERAL_AT)?,
        word(TOTAL_SIZE)?,
        word(PREFIX_SIZE)?,
        word(REGISTERS)?,
    );
    // The layout every stage of the game's files has: prefix, header, instructions, then the
    // literals, each on a page of their own.
    let end = prefix + size;
    let laid_out = prefix == 0x30
        && total == stage.code.len()
        && size >= 0x70
        && (size - 0x50).is_multiple_of(32)
        && if literal_size == 0 {
            total == page(end)
        } else {
            literal_at == page(end) && total == page(literal_at + literal_size)
        };
    if !laid_out {
        return Err("its code is laid out in a way Visionary does not know".to_owned());
    }
    let literals = if literal_size == 0 {
        &[][..]
    } else {
        &stage.code[literal_at..literal_at + literal_size]
    };
    let start = prefix + 0x50;
    let mut code = stage.code[start..end].to_vec();
    let attributes =
        Attributes::of(inputs).ok_or("its particles carry no age to animate by".to_owned())?;

    let insts = decode_program(&code);
    let first = insts.first().ok_or("it has no code".to_owned())?;
    if !matches!(first.op, Op::Psetp { .. } | Op::Mov { .. }) || !first.guard.is_always() {
        return Err("it starts in a way Visionary does not know".to_owned());
    }
    for &table in tables {
        let indexed = insts.iter().any(|i| {
            matches!(i.op, Op::Ldc { bank: STATIC, offset, .. }
                if table.slot(offset as u32).is_some())
        });
        if indexed {
            return Err("it picks one key for each particle".to_owned());
        }
    }

    let mut stub = Stub {
        words: Vec::new(),
        next: u8::try_from(registers).map_err(|_| "it uses every register".to_owned())?,
        pending: false,
    };
    // Room for the working registers below and every table's value.
    if registers + 9 + tables.iter().map(|t| t.width() as usize).sum::<usize>() > 250 {
        return Err("it uses every register".to_owned());
    }
    let [age, per_life, random, time, rate, at, span, from, _to] =
        std::array::from_fn(|_| stub.register());

    stub.attribute(age, attributes.born);
    stub.sub_from_constant(age, age, DYNAMIC, TIME);
    stub.attribute(per_life, attributes.life);
    stub.truncate(per_life, per_life);
    stub.reciprocal(per_life, per_life);
    match attributes.random {
        Some(offset) => stub.attribute(random, offset),
        None => stub.fixed(0x5c98_0780_0007_0000 | u64::from(RZ) << 20 | u64::from(random)),
    }
    let mut values: Vec<(Table, [u8; 3])> = Vec::new();
    for &table in tables {
        let (loop_at, random_at) = table.looping();
        // How far through: the part of the loop gone by, or of the particle's life.
        stub.mov(rate, STATIC, loop_at);
        stub.reciprocal(rate, rate);
        stub.mul_constant(time, random, STATIC, random_at);
        stub.fma_constant(time, time, STATIC, loop_at, age);
        stub.mul(time, time, rate, u64::from(PT), false);
        stub.floor(at, time);
        stub.positive(STATIC, loop_at);
        stub.sub(time, time, at, 0);
        stub.mul(time, age, per_life, 8, false);
        // Each span adds the part of it gone by, held to between none and all.
        let value: [u8; 3] = std::array::from_fn(|k| {
            if k < table.width() as usize {
                let r = stub.register();
                stub.mov(r, STATIC, table.keys() + 4 * k as u32);
                r
            } else {
                RZ
            }
        });
        for key in 0..KEYS - 1 {
            let (here, next) = (table.keys() + 16 * key, table.keys() + 16 * (key + 1));
            stub.mov(span, STATIC, here + 12);
            stub.sub_from_constant(span, span, STATIC, next + 12);
            stub.reciprocal(span, span);
            stub.sub_constant(at, time, STATIC, here + 12);
            stub.mul(at, at, span, u64::from(PT), true);
            for (k, &v) in value.iter().take(table.width() as usize).enumerate() {
                stub.mov(from, STATIC, here + 4 * k as u32);
                stub.sub_from_constant(from, from, STATIC, next + 4 * k as u32);
                stub.fma(v, at, from, v);
            }
        }
        values.push((table, value));
    }

    // Point the old code's reads of those keys at the values.
    let mut touched = Vec::new();
    for inst in &insts {
        let word_at = inst.addr as usize;
        let word = u64::from_le_bytes(code[word_at..word_at + 8].try_into().unwrap());
        let replaced = sources(&inst.op).into_iter().find_map(|src| match src {
            Src::Cbuf {
                bank: STATIC,
                offset,
            } => values.iter().find_map(|(table, value)| {
                let (_, number) = table.slot(offset)?;
                let value = *value.get(number as usize)?;
                (value != RZ).then_some((src, value))
            }),
            _ => None,
        });
        let Some((src, value)) = replaced else {
            continue;
        };
        let new = read_register(word, inst, src, value)
            .ok_or_else(|| format!("it reads a key in a way Visionary does not know: {inst}"))?;
        code[word_at..word_at + 8].copy_from_slice(&new.to_le_bytes());
        touched.push(inst.addr);
    }
    for addr in touched {
        // A pair that issues together, and an operand kept for the next instruction, were worked
        // out for the old operands.
        relax(&mut code, addr);
        if let Some(before) = previous(addr) {
            relax(&mut code, before);
        }
    }

    // Jump to the added code in place of the first instruction, which runs at its end.
    let entry = first.addr as usize;
    let original = u64::from_le_bytes(code[entry..entry + 8].try_into().unwrap());
    stub.push(original, 15, false);
    let added_at = code.len() as u32;
    let back_from = address(added_at, stub.words.len());
    let after = following(first.addr);
    stub.push(branch(back_from, after), 15, false);
    while !stub.words.len().is_multiple_of(3) {
        stub.push(0x50b0_0000_0007_0f00, 15, false);
    }
    let jump = branch(first.addr, address(added_at, 0));
    code[entry..entry + 8].copy_from_slice(&jump.to_le_bytes());
    set_sched(&mut code, first.addr, sched(15, false, false));
    for bundle in stub.words.chunks(3) {
        let scheds = bundle
            .iter()
            .enumerate()
            .fold(0u64, |all, (k, (_, s))| all | u64::from(*s) << (21 * k));
        code.extend_from_slice(&scheds.to_le_bytes());
        for (word, _) in bundle {
            code.extend_from_slice(&word.to_le_bytes());
        }
    }

    let size = 0x50 + code.len();
    let end = prefix + size;
    let mut out = stage.code[..start].to_vec();
    out.extend_from_slice(&code);
    out.resize(page(end), 0);
    let mut control = stage.control.clone();
    let mut set = |at: usize, value: usize| {
        control[at..at + 4].copy_from_slice(&(value as u32).to_le_bytes());
    };
    set(CODE_SIZE, size);
    if literal_size != 0 {
        set(LITERAL_AT, out.len());
        out.extend_from_slice(literals);
        out.resize(page(out.len()), 0);
    }
    set(TOTAL_SIZE, out.len());
    set(REGISTERS, usize::from(stub.next));
    Ok(StageCode { control, code: out })
}

/// `word` with its constant operand `src` read from register `value` instead, checked by decoding
/// it again.
fn read_register(word: u64, inst: &Inst, src: Src, value: u8) -> Option<u64> {
    let operand = 0x7ffffu64 << 20;
    let register = u64::from(value);
    let new = match (word >> 60, &inst.op) {
        // The constant is the last of three operands: the middle one, a register, trades places
        // with it.
        (0x5, Op::Ffma { c, .. }) if *c == src => {
            let middle = word >> 39 & 0xff;
            word & !(operand | 0xff << 39) & !(0xf << 59)
                | 0xb << 59
                | middle << 20
                | register << 39
        }
        (0x4, _) => word & !operand | 1 << 60 | register << 20,
        _ => return None,
    };
    let want = with_source(inst.op.clone(), src, Src::Reg(value))?;
    let now = decode(new, inst.addr);
    (now.op == want && now.guard == inst.guard).then_some(new)
}

/// `op` with every operand that is `from` made `to`.
fn with_source(mut op: Op, from: Src, to: Src) -> Option<Op> {
    let swap = |s: &mut Src| {
        if *s == from {
            *s = to;
        }
    };
    match &mut op {
        Op::Mov { s, .. } | Op::Rro { s, .. } | Op::F2f { s, .. } => swap(s),
        Op::Sel { b, .. }
        | Op::Fmul { b, .. }
        | Op::Fadd { b, .. }
        | Op::Fmnmx { b, .. }
        | Op::Fset { b, .. }
        | Op::Fsetp { b, .. }
        | Op::Lop { b, .. } => swap(b),
        Op::Ffma { b, c, .. } => {
            swap(b);
            swap(c);
        }
        _ => return None,
    }
    Some(op)
}

/// The byte address of the `index`th instruction counted from a bundle at `start`.
fn address(start: u32, index: usize) -> u32 {
    start + 32 * (index as u32 / 3) + 8 * (index as u32 % 3 + 1)
}

/// The instruction after the one at `addr`, and the one before it.
fn following(addr: u32) -> u32 {
    if addr % 32 == 24 {
        addr + 16
    } else {
        addr + 8
    }
}

fn previous(addr: u32) -> Option<u32> {
    match addr % 32 {
        8 => addr.checked_sub(16),
        _ => Some(addr - 8),
    }
}

fn branch(from: u32, to: u32) -> u64 {
    let offset = i64::from(to) - i64::from(from) - 8;
    0xe240_0000_0007_000f | ((offset as u64) & 0xff_ffff) << 20
}

fn sched_at(addr: u32) -> (usize, u32) {
    let bundle = (addr & !31) as usize;
    (bundle, 21 * ((addr % 32) / 8 - 1))
}

fn set_sched(code: &mut [u8], addr: u32, value: u32) {
    let (bundle, shift) = sched_at(addr);
    let word = u64::from_le_bytes(code[bundle..bundle + 8].try_into().unwrap());
    let word = word & !(0x1f_ffff << shift) | u64::from(value) << shift;
    code[bundle..bundle + 8].copy_from_slice(&word.to_le_bytes());
}

/// Have the instruction at `addr` issue alone and keep no operand for the next.
fn relax(code: &mut [u8], addr: u32) {
    let (bundle, shift) = sched_at(addr);
    let word = u64::from_le_bytes(code[bundle..bundle + 8].try_into().unwrap());
    let mut value = (word >> shift) as u32 & 0x1f_ffff;
    value &= !(0xf << 17);
    if value & 0xf == 0 {
        value |= 1;
    }
    set_sched(code, addr, value);
}

#[cfg(test)]
mod tests {
    use super::super::decode::{MufuOp, Rounding};
    use super::*;

    /// The added code's instructions are the ones they are meant to be.
    #[test]
    fn added_instructions_decode_as_meant() {
        let mut stub = Stub {
            words: Vec::new(),
            next: 40,
            pending: false,
        };
        stub.attribute(40, 0xdc);
        stub.mov(41, STATIC, 0x44c);
        stub.sub_from_constant(42, 41, STATIC, 0x45c);
        stub.sub_constant(43, 42, STATIC, 0x44c);
        stub.sub(44, 43, 42, 0);
        stub.mul(45, 44, 43, 8, false);
        stub.mul(45, 44, 43, 7, true);
        stub.mul_constant(46, 45, STATIC, 0x98);
        stub.fma(47, 46, 45, 44);
        stub.fma_constant(48, 47, STATIC, 0x84, 46);
        stub.reciprocal(49, 48);
        stub.floor(50, 49);
        stub.truncate(51, 50);
        stub.positive(STATIC, 0x84);
        let ops: Vec<String> = stub
            .words
            .iter()
            .map(|(w, _)| decode(*w, 8).to_string())
            .collect();
        let c = |offset: u32| Src::Cbuf {
            bank: STATIC,
            offset,
        };
        let op = |k: usize| decode(stub.words[k].0, 8);
        assert!(
            matches!(
                op(0).op,
                Op::Ald {
                    d: 40,
                    offset: 0xdc,
                    count: 1,
                    ..
                }
            ),
            "{ops:#?}"
        );
        assert!(matches!(op(1).op, Op::Mov { d: 41, s } if s == c(0x44c)));
        assert!(
            matches!(op(2).op, Op::Fadd { d: 42, a: 41, b, ma, mb, sat: false }
            if b == c(0x45c) && ma.neg && !mb.neg)
        );
        assert!(
            matches!(op(3).op, Op::Fadd { d: 43, a: 42, b, ma, mb, sat: false }
            if b == c(0x44c) && !ma.neg && mb.neg)
        );
        assert!(
            matches!(op(4).op, Op::Fadd { d: 44, a: 43, b: Src::Reg(42), ma, mb, .. }
            if !ma.neg && mb.neg)
        );
        assert_eq!((op(4).guard.reg, op(4).guard.not), (0, false));
        assert!(matches!(
            op(5).op,
            Op::Fmul {
                d: 45,
                a: 44,
                b: Src::Reg(43),
                sat: false,
                ..
            }
        ));
        assert_eq!((op(5).guard.reg, op(5).guard.not), (0, true));
        assert!(matches!(
            op(6).op,
            Op::Fmul {
                d: 45,
                sat: true,
                neg_b: false,
                scale: 0,
                ..
            }
        ));
        assert!(op(6).guard.is_always());
        assert!(matches!(op(7).op, Op::Fmul { d: 46, a: 45, b, .. } if b == c(0x98)));
        assert!(matches!(
            op(8).op,
            Op::Ffma {
                d: 47,
                a: 46,
                b: Src::Reg(45),
                c: Src::Reg(44),
                neg_b: false,
                neg_c: false,
                sat: false
            }
        ));
        assert!(
            matches!(op(9).op, Op::Ffma { d: 48, a: 47, b, c: Src::Reg(46), .. } if b == c(0x84))
        );
        assert!(matches!(
            op(10).op,
            Op::Mufu {
                d: 49,
                a: 48,
                op: MufuOp::Rcp,
                ..
            }
        ));
        assert!(matches!(
            op(11).op,
            Op::F2f {
                d: 50,
                s: Src::Reg(49),
                round: Some(Rounding::Floor),
                ..
            }
        ));
        assert!(matches!(
            op(12).op,
            Op::F2i {
                d: 51,
                s: Src::Reg(50),
                ..
            }
        ));
        assert!(matches!(
            op(13).op,
            Op::I2f {
                d: 51,
                s: Src::Reg(51),
                ..
            }
        ));
        assert!(
            matches!(op(14).op, Op::Fsetp { p0: 0, p1: 7, a: RZ, b, .. } if b == c(0x84)),
            "{}",
            ops[14]
        );
    }
}
