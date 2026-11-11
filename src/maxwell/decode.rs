//! Maxwell (SM 5.x) instruction decoding, for the subset the effect shaders use.
//!
//! Bit layouts follow envytools' gm107 tables (MIT) and Mesa NAK's SM50 encoder (MIT).
//! Code comes in 32-byte bundles: one scheduling word, then three instructions.

use std::fmt;

pub const RZ: u8 = 255;
pub const PT: u8 = 7;

fn bits(w: u64, lo: u32, len: u32) -> u64 {
    (w >> lo) & ((1u64 << len) - 1)
}

fn bit(w: u64, at: u32) -> bool {
    (w >> at) & 1 != 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pred {
    pub reg: u8,
    pub not: bool,
}

impl Pred {
    pub const TRUE: Pred = Pred {
        reg: PT,
        not: false,
    };
    fn at(w: u64, lo: u32, not_bit: u32) -> Self {
        Pred {
            reg: bits(w, lo, 3) as u8,
            not: bit(w, not_bit),
        }
    }
    pub fn is_always(self) -> bool {
        self.reg == PT && !self.not
    }
}

/// The second operand of most ALU forms: a register, a constant-buffer word or an immediate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Src {
    Reg(u8),
    /// `c[bank][offset]`, offset in bytes.
    Cbuf {
        bank: u8,
        offset: u32,
    },
    /// Raw 32-bit pattern (a float's bits for float forms).
    Imm(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatCmp {
    F,
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
    Num,
    Nan,
    Ltu,
    Equ,
    Leu,
    Gtu,
    Neu,
    Geu,
    T,
}

impl FloatCmp {
    fn from(v: u64) -> Self {
        use FloatCmp::*;
        [
            F, Lt, Eq, Le, Gt, Ne, Ge, Num, Nan, Ltu, Equ, Leu, Gtu, Neu, Geu, T,
        ][v as usize & 15]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntCmp {
    F,
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
    T,
}

impl IntCmp {
    fn from(v: u64) -> Self {
        use IntCmp::*;
        [F, Lt, Eq, Le, Gt, Ne, Ge, T][v as usize & 7]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
    Xor,
}

impl BoolOp {
    fn from(v: u64) -> Self {
        match v & 3 {
            0 => BoolOp::And,
            1 => BoolOp::Or,
            _ => BoolOp::Xor,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Or,
    Xor,
    PassB,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MufuOp {
    Cos,
    Sin,
    Ex2,
    Lg2,
    Rcp,
    Rsq,
    Sqrt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rounding {
    Nearest,
    Floor,
    Ceil,
    Trunc,
}

/// Integer conversion widths: size in bytes and signedness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntType {
    pub bytes: u8,
    pub signed: bool,
}

/// Abs/neg pair on a float operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FMod {
    pub neg: bool,
    pub abs: bool,
}

/// Which condition-code test `bra`/`exit`/`kil`/`csetp` use (envytools tabf0f8_0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CcTest(pub u8);

impl CcTest {
    pub const ALWAYS: CcTest = CcTest(0xf);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexTarget {
    D1LodZero,
    D2,
    D2LodZero,
    D2LodLevel,
    D2DepthCompare,
    D2LodLevelDepthCompare,
    D2LodZeroDepthCompare,
    Array2D,
    Array2DLodZero,
    Array2DLodZeroDepthCompare,
    D3,
    D3LodZero,
    Cube,
    CubeLodLevel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Nop,
    Exit(CcTest),
    Kil(CcTest),
    Bra {
        target: u32,
        cc: CcTest,
    },
    Ssy {
        target: u32,
    },
    Sync(CcTest),
    Mov {
        d: u8,
        s: Src,
    },
    Sel {
        d: u8,
        a: u8,
        b: Src,
        p: Pred,
    },
    Ffma {
        d: u8,
        a: u8,
        b: Src,
        c: Src,
        neg_b: bool,
        neg_c: bool,
        sat: bool,
    },
    Fmul {
        d: u8,
        a: u8,
        b: Src,
        neg_b: bool,
        sat: bool,
        scale: i8,
    },
    Fadd {
        d: u8,
        a: u8,
        b: Src,
        ma: FMod,
        mb: FMod,
        sat: bool,
    },
    Fmnmx {
        d: u8,
        a: u8,
        b: Src,
        ma: FMod,
        mb: FMod,
        p: Pred,
    },
    Mufu {
        d: u8,
        a: u8,
        op: MufuOp,
        m: FMod,
        sat: bool,
    },
    Rro {
        d: u8,
        s: Src,
        m: FMod,
    },
    Fset {
        d: u8,
        a: u8,
        b: Src,
        ma: FMod,
        mb: FMod,
        cmp: FloatCmp,
        bop: BoolOp,
        p: Pred,
        bf: bool,
    },
    Fsetp {
        p0: u8,
        p1: u8,
        a: u8,
        b: Src,
        ma: FMod,
        mb: FMod,
        cmp: FloatCmp,
        bop: BoolOp,
        p: Pred,
    },
    Isetp {
        p0: u8,
        p1: u8,
        a: u8,
        b: Src,
        cmp: IntCmp,
        signed: bool,
        bop: BoolOp,
        p: Pred,
        x: bool,
    },
    Iset {
        d: u8,
        a: u8,
        b: Src,
        cmp: IntCmp,
        signed: bool,
        bop: BoolOp,
        p: Pred,
        bf: bool,
        x: bool,
    },
    Psetp {
        p0: u8,
        p1: u8,
        a: Pred,
        b: Pred,
        op1: BoolOp,
        op2: BoolOp,
        c: Pred,
    },
    Csetp {
        p0: u8,
        p1: u8,
        cc: CcTest,
        bop: BoolOp,
        p: Pred,
    },
    R2p {
        a: u8,
        mask: Src,
        byte: u8,
        cc: bool,
    },
    I2f {
        d: u8,
        s: Src,
        from: IntType,
        to_bytes: u8,
        m: FMod,
        round: Rounding,
        byte: u8,
    },
    F2i {
        d: u8,
        s: Src,
        to: IntType,
        m: FMod,
        round: Rounding,
    },
    F2f {
        d: u8,
        s: Src,
        m: FMod,
        round: Option<Rounding>,
        sat: bool,
    },
    I2i {
        d: u8,
        s: Src,
        from: IntType,
        to: IntType,
        neg: bool,
        abs: bool,
        sat: bool,
        byte: u8,
    },
    Iadd {
        d: u8,
        a: u8,
        b: Src,
        neg_a: bool,
        neg_b: bool,
        x: bool,
        cc: bool,
        sat: bool,
    },
    Iadd3 {
        d: u8,
        a: u8,
        b: Src,
        c: u8,
        neg: [bool; 3],
        x: bool,
        cc: bool,
    },
    Iscadd {
        d: u8,
        a: u8,
        b: Src,
        shift: u8,
        neg_a: bool,
        neg_b: bool,
    },
    Xmad {
        d: u8,
        a: u8,
        b: Src,
        c: Src,
        signed_a: bool,
        signed_b: bool,
        hi_a: bool,
        hi_b: bool,
        mode: u8,
        psl: bool,
        mrg: bool,
        x: bool,
        cc: bool,
    },
    /// `p_mode`: 0 none, 1 true, 2 zero, 3 non-zero — what `p_out` is set to.
    Lop {
        d: u8,
        a: u8,
        b: Src,
        op: LogicOp,
        inv_a: bool,
        inv_b: bool,
        x: bool,
        cc: bool,
        p_out: u8,
        p_mode: u8,
    },
    Pset {
        d: u8,
        a: Pred,
        b: Pred,
        op1: BoolOp,
        op2: BoolOp,
        c: Pred,
        bf: bool,
    },
    Lop3 {
        d: u8,
        a: u8,
        b: Src,
        c: u8,
        lut: u8,
        cc: bool,
    },
    Shl {
        d: u8,
        a: u8,
        b: Src,
        wrap: bool,
    },
    Shr {
        d: u8,
        a: u8,
        b: Src,
        signed: bool,
        wrap: bool,
    },
    /// Attribute load (`ld a[...]`): `count` words from `offset` (+ register `base`).
    Ald {
        d: u8,
        base: u8,
        offset: u32,
        count: u8,
        output: bool,
    },
    /// Attribute store (`st a[...]`).
    Ast {
        s: u8,
        base: u8,
        offset: u32,
        count: u8,
    },
    /// `ld c[bank][reg + offset]`.
    Ldc {
        d: u8,
        base: u8,
        bank: u8,
        offset: i32,
        bytes: u8,
        signed: bool,
    },
    /// Varying interpolation; `mode` 0 pass, 1 multiply (by `w` reg), 2 constant (flat), 3 sc.
    Ipa {
        d: u8,
        attr: u32,
        base: u8,
        mode: u8,
        mul: u8,
        sat: bool,
    },
    Texs {
        d0: u8,
        d1: u8,
        a: u8,
        b: u8,
        handle: u32,
        target: TexTarget,
        mask: u8,
    },
    Tmml {
        d: u8,
        a: u8,
        b: u8,
        handle: u32,
    },
    /// Read a system value (`mov rd, sr`).
    S2r {
        d: u8,
        sr: u8,
    },
    Unknown(u64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Inst {
    /// Byte address inside the program, counted from the first instruction.
    pub addr: u32,
    pub guard: Pred,
    pub op: Op,
}

impl fmt::Display for Inst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.guard.is_always() {
            write!(
                f,
                "@{}P{} ",
                if self.guard.not { "!" } else { "" },
                self.guard.reg
            )?;
        }
        write!(f, "{:?}", self.op)
    }
}

/// Float immediate stored as the top 19 bits of the value plus a sign at bit 56.
fn f20(w: u64) -> u32 {
    ((bits(w, 20, 19) as u32) << 12) | ((bit(w, 56) as u32) << 31)
}

/// Signed 20-bit integer immediate: 19 bits at 20, sign at 56.
fn s20(w: u64) -> u32 {
    let v = bits(w, 20, 19) as i32;
    (if bit(w, 56) { v - (1 << 19) } else { v }) as u32
}

fn cbuf(w: u64) -> Src {
    Src::Cbuf {
        bank: bits(w, 34, 5) as u8,
        offset: (bits(w, 20, 14) as u32) * 4,
    }
}

fn reg(w: u64, lo: u32) -> u8 {
    bits(w, lo, 8) as u8
}

/// Operand B by the form encoded in the top bits: 0x5c.. register, 0x4c.. cbuf, 0x38.. imm.
fn src_b(w: u64, float: bool) -> Src {
    match w >> 60 {
        0x5 => Src::Reg(reg(w, 20)),
        0x4 => cbuf(w),
        _ => Src::Imm(if float { f20(w) } else { s20(w) }),
    }
}

fn int_type(v: u64) -> IntType {
    // bits: [1:0] size (0 u8,1 u16,2 u32,3 u64), bit 2 signed — as packed by callers.
    IntType {
        bytes: 1 << (v & 3),
        signed: v & 4 != 0,
    }
}

pub fn decode_program(code: &[u8]) -> Vec<Inst> {
    let mut out = Vec::new();
    for (index, chunk) in code.chunks_exact(8).enumerate() {
        if index % 4 == 0 {
            continue;
        }
        let w = u64::from_le_bytes(chunk.try_into().unwrap());
        let addr = (index * 8) as u32;
        let inst = decode(w, addr);
        let end = matches!(inst.op, Op::Exit(_)) && inst.guard.is_always();
        out.push(inst);
        if end {
            // Instructions after an unconditional exit can still be branch targets; keep
            // decoding until the code runs into padding (all-zero bundles).
            continue;
        }
    }
    // Trim trailing data that decodes as nothing useful.
    while matches!(
        out.last(),
        Some(Inst {
            op: Op::Unknown(_),
            ..
        })
    ) {
        out.pop();
    }
    out
}

pub fn decode(w: u64, addr: u32) -> Inst {
    let guard = Pred::at(w, 16, 19);
    let op = decode_op(w, addr);
    let guard = match op {
        // Control-flow setup instructions carry no guard field.
        Op::Ssy { .. } => Pred::TRUE,
        _ => guard,
    };
    Inst { addr, guard, op }
}

fn branch_target(w: u64, addr: u32) -> u32 {
    let off = bits(w, 20, 24) as i32;
    let off = if off & (1 << 23) != 0 {
        off - (1 << 24)
    } else {
        off
    };
    (addr as i32 + 8 + off) as u32
}

fn fsetp_like(w: u64) -> (FMod, FMod) {
    (
        FMod {
            neg: bit(w, 43),
            abs: bit(w, 7),
        },
        FMod {
            neg: bit(w, 6),
            abs: bit(w, 44),
        },
    )
}

fn decode_op(w: u64, addr: u32) -> Op {
    let top16 = (w >> 48) as u16;
    let d = reg(w, 0);
    let a = reg(w, 8);
    let m13 = top16 & 0xfff8;
    match m13 {
        0xe300 | 0xe308 => return Op::Exit(CcTest(bits(w, 0, 5) as u8)),
        0xe330 | 0xe338 => return Op::Kil(CcTest(bits(w, 0, 5) as u8)),
        0xe240 | 0xe248 => {
            return Op::Bra {
                target: branch_target(w, addr),
                cc: CcTest(bits(w, 0, 5) as u8),
            }
        }
        0xe290 | 0xe298 => {
            return Op::Ssy {
                target: branch_target(w, addr),
            }
        }
        0xf0f8 => return Op::Sync(CcTest(bits(w, 0, 5) as u8)),
        0x50b0 => return Op::Nop,
        0xf0c8 => {
            return Op::S2r {
                d,
                sr: bits(w, 20, 8) as u8,
            }
        }
        0xefd8 => {
            return Op::Ald {
                d,
                base: a,
                offset: bits(w, 20, 10) as u32,
                count: bits(w, 47, 2) as u8 + 1,
                output: bit(w, 32),
            }
        }
        0xeff0 => {
            return Op::Ast {
                s: d,
                base: a,
                offset: bits(w, 20, 10) as u32,
                count: bits(w, 47, 2) as u8 + 1,
            }
        }
        0xef90 => {
            let off = bits(w, 20, 16) as i32;
            let off = if off & 0x8000 != 0 {
                off - 0x10000
            } else {
                off
            };
            let size = bits(w, 48, 3);
            return Op::Ldc {
                d,
                base: a,
                bank: bits(w, 36, 5) as u8,
                offset: off,
                bytes: match size {
                    0 | 1 => 1,
                    2 | 3 => 2,
                    5 => 8,
                    _ => 4,
                },
                signed: size == 1 || size == 3,
            };
        }
        0x5cf0 | 0x4cf0 | 0x38f0 | 0x3cf8 => {}
        _ => {}
    }
    if top16 & 0xff00 == 0xe000 && !bit(w, 38) {
        return Op::Ipa {
            d,
            attr: bits(w, 28, 10) as u32,
            base: a,
            mode: bits(w, 54, 2) as u8,
            mul: reg(w, 20),
            sat: bit(w, 51),
        };
    }
    if top16 & 0xf600 == 0xd000 {
        let target = match bits(w, 53, 4) {
            0 => TexTarget::D1LodZero,
            1 => TexTarget::D2,
            2 => TexTarget::D2LodZero,
            3 => TexTarget::D2LodLevel,
            4 => TexTarget::D2DepthCompare,
            5 => TexTarget::D2LodLevelDepthCompare,
            6 => TexTarget::D2LodZeroDepthCompare,
            7 => TexTarget::Array2D,
            8 => TexTarget::Array2DLodZero,
            9 => TexTarget::Array2DLodZeroDepthCompare,
            10 => TexTarget::D3,
            11 => TexTarget::D3LodZero,
            12 => TexTarget::Cube,
            _ => TexTarget::CubeLodLevel,
        };
        return Op::Texs {
            d0: d,
            d1: reg(w, 28),
            a,
            b: reg(w, 20),
            handle: bits(w, 36, 13) as u32,
            target,
            mask: bits(w, 50, 3) as u8,
        };
    }
    if m13 == 0xdf58 {
        return Op::Tmml {
            d,
            a,
            b: reg(w, 20),
            handle: bits(w, 36, 13) as u32,
        };
    }
    // ALU forms: the same operation in register (0x5c..), cbuf (0x4c..) and immediate (0x38..)
    // spellings. Normalise to the register opcode.
    // Immediate forms drop bit 56 (the immediate's sign) from the opcode.
    let base = match w >> 60 {
        0x5 => m13,
        0x4 => m13 + 0x1000,
        0x3 if top16 & 0xfe00 == 0x3800 => (top16 & 0xfef8) + 0x2400,
        _ => 0,
    };
    let sat50 = bit(w, 50);
    match base {
        0x5c98 => {
            return Op::Mov {
                d,
                s: src_b(w, false),
            }
        }
        0x5ca0 => {
            return Op::Sel {
                d,
                a,
                b: src_b(w, false),
                p: Pred::at(w, 39, 42),
            }
        }
        0x5c68 => {
            let scale = match bits(w, 41, 3) {
                1 => -1,
                2 => -2,
                3 => -3,
                4 => 3,
                5 => 2,
                6 => 1,
                _ => 0,
            };
            return Op::Fmul {
                d,
                a,
                b: src_b(w, true),
                neg_b: bit(w, 48),
                sat: sat50,
                scale,
            };
        }
        0x5c58 => {
            return Op::Fadd {
                d,
                a,
                b: src_b(w, true),
                ma: FMod {
                    neg: bit(w, 48),
                    abs: bit(w, 46),
                },
                mb: FMod {
                    neg: bit(w, 45),
                    abs: bit(w, 49),
                },
                sat: sat50,
            }
        }
        0x5c60 => {
            return Op::Fmnmx {
                d,
                a,
                b: src_b(w, true),
                ma: FMod {
                    neg: bit(w, 48),
                    abs: bit(w, 46),
                },
                mb: FMod {
                    neg: bit(w, 45),
                    abs: bit(w, 49),
                },
                p: Pred::at(w, 39, 42),
            }
        }
        0x5c90 => {
            return Op::Rro {
                d,
                s: src_b(w, true),
                m: FMod {
                    neg: bit(w, 45),
                    abs: bit(w, 49),
                },
            }
        }
        0x5c48 => {
            return Op::Shl {
                d,
                a,
                b: src_b(w, false),
                wrap: bit(w, 39),
            }
        }
        0x5c28 => {
            return Op::Shr {
                d,
                a,
                b: src_b(w, false),
                signed: bit(w, 48),
                wrap: bit(w, 39),
            }
        }
        0x5c40 => {
            return Op::Lop {
                d,
                a,
                b: src_b(w, false),
                op: match bits(w, 41, 2) {
                    0 => LogicOp::And,
                    1 => LogicOp::Or,
                    2 => LogicOp::Xor,
                    _ => LogicOp::PassB,
                },
                inv_a: bit(w, 39),
                inv_b: bit(w, 40),
                x: bit(w, 43),
                cc: bit(w, 47),
                p_out: bits(w, 48, 3) as u8,
                p_mode: bits(w, 44, 2) as u8,
            }
        }
        0x5c18 => {
            return Op::Iscadd {
                d,
                a,
                b: src_b(w, false),
                shift: bits(w, 39, 5) as u8,
                neg_a: bit(w, 49),
                neg_b: bit(w, 48),
            }
        }
        0x5c10 => {
            return Op::Iadd {
                d,
                a,
                b: src_b(w, false),
                neg_a: bit(w, 49),
                neg_b: bit(w, 48),
                x: bit(w, 43),
                cc: bit(w, 47),
                sat: sat50,
            }
        }
        0x5cb8 => {
            let from = int_type(bits(w, 10, 2) | if bit(w, 13) { 4 } else { 0 });
            return Op::I2f {
                d,
                s: src_b(w, false),
                from,
                to_bytes: 1 << bits(w, 8, 2),
                m: FMod {
                    neg: bit(w, 45),
                    abs: bit(w, 49),
                },
                round: rounding(bits(w, 39, 2)),
                byte: bits(w, 41, 2) as u8,
            };
        }
        0x5cb0 => {
            let to = int_type(bits(w, 8, 2) | if bit(w, 12) { 4 } else { 0 });
            return Op::F2i {
                d,
                s: src_b(w, true),
                to,
                m: FMod {
                    neg: bit(w, 45),
                    abs: bit(w, 49),
                },
                round: rounding(bits(w, 39, 2)),
            };
        }
        0x5ca8 => {
            let r = bits(w, 39, 3);
            return Op::F2f {
                d,
                s: src_b(w, true),
                m: FMod {
                    neg: bit(w, 45),
                    abs: bit(w, 49),
                },
                round: if bit(w, 42) {
                    Some(rounding(r & 3))
                } else {
                    None
                },
                sat: sat50,
            };
        }
        0x5ce0 => {
            let to = int_type(bits(w, 8, 2) | if bit(w, 12) { 4 } else { 0 });
            let from = int_type(bits(w, 10, 2) | if bit(w, 13) { 4 } else { 0 });
            return Op::I2i {
                d,
                s: src_b(w, false),
                from,
                to,
                neg: bit(w, 45),
                abs: bit(w, 49),
                sat: sat50,
                byte: bits(w, 41, 2) as u8,
            };
        }
        _ => {}
    }
    // Families keyed on fewer bits.
    let t12 = top16 & 0xfff0;
    let t12n = match w >> 60 {
        0x5 => t12,
        0x4 => t12 + 0x1000,
        0x3 if top16 & 0xfe00 == 0x3600 => (top16 & 0xfef0) + 0x2500,
        0x3 if top16 & 0xfe00 == 0x3800 => (top16 & 0xfef0) + 0x2400,
        _ => 0,
    };
    match t12n {
        0x5bb0 => {
            let (ma, mb) = fsetp_like(w);
            return Op::Fsetp {
                p0: bits(w, 3, 3) as u8,
                p1: bits(w, 0, 3) as u8,
                a,
                b: src_b(w, true),
                ma,
                mb,
                cmp: FloatCmp::from(bits(w, 48, 4)),
                bop: BoolOp::from(bits(w, 45, 2)),
                p: Pred::at(w, 39, 42),
            };
        }
        0x5b60 => {
            return Op::Isetp {
                p0: bits(w, 3, 3) as u8,
                p1: bits(w, 0, 3) as u8,
                a,
                b: src_b(w, false),
                cmp: IntCmp::from(bits(w, 49, 3)),
                signed: bit(w, 48),
                bop: BoolOp::from(bits(w, 45, 2)),
                p: Pred::at(w, 39, 42),
                x: bit(w, 43),
            }
        }
        0x5b50 => {
            return Op::Iset {
                d,
                a,
                b: src_b(w, false),
                cmp: IntCmp::from(bits(w, 49, 3)),
                signed: bit(w, 48),
                bop: BoolOp::from(bits(w, 45, 2)),
                p: Pred::at(w, 39, 42),
                bf: bit(w, 44),
                x: bit(w, 43),
            }
        }
        0x5cc0 => {
            return Op::Iadd3 {
                d,
                a,
                b: src_b(w, false),
                c: reg(w, 39),
                neg: [bit(w, 51), bit(w, 50), bit(w, 49)],
                x: bit(w, 48),
                cc: bit(w, 47),
            }
        }
        _ => {}
    }
    match top16 >> 8 {
        // ffma: 0x59 reg/reg, 0x51 reg/cbuf-in-c, 0x49 cbuf b, 0x32/0x33 imm b
        0x59 | 0x51 | 0x49 | 0x32 | 0x33 if top16 & 0x80 != 0 => {
            let (b, c) = match top16 >> 8 {
                0x59 => (Src::Reg(reg(w, 20)), Src::Reg(reg(w, 39))),
                0x51 => (Src::Reg(reg(w, 39)), cbuf(w)),
                0x49 => (cbuf(w), Src::Reg(reg(w, 39))),
                _ => (Src::Imm(f20(w)), Src::Reg(reg(w, 39))),
            };
            return Op::Ffma {
                d,
                a,
                b,
                c,
                neg_b: bit(w, 48),
                neg_c: bit(w, 49),
                sat: sat50,
            };
        }
        0x58 | 0x48 | 0x30 | 0x31 => {
            let b = match top16 >> 8 {
                0x58 => Src::Reg(reg(w, 20)),
                0x48 => cbuf(w),
                _ => Src::Imm(f20(w)),
            };
            return Op::Fset {
                d,
                a,
                b,
                ma: FMod {
                    neg: bit(w, 43),
                    abs: bit(w, 54),
                },
                mb: FMod {
                    neg: bit(w, 53),
                    abs: bit(w, 44),
                },
                cmp: FloatCmp::from(bits(w, 48, 4)),
                bop: BoolOp::from(bits(w, 45, 2)),
                p: Pred::at(w, 39, 42),
                bf: bit(w, 52),
            };
        }
        0x50 if m13 == 0x5080 => {
            let op = match bits(w, 20, 4) {
                0 => MufuOp::Cos,
                1 => MufuOp::Sin,
                2 => MufuOp::Ex2,
                3 => MufuOp::Lg2,
                4 => MufuOp::Rcp,
                5 => MufuOp::Rsq,
                _ => MufuOp::Sqrt,
            };
            return Op::Mufu {
                d,
                a,
                op,
                m: FMod {
                    neg: bit(w, 48),
                    abs: bit(w, 46),
                },
                sat: sat50,
            };
        }
        0x50 if m13 == 0x5090 => {
            return Op::Psetp {
                p0: bits(w, 3, 3) as u8,
                p1: bits(w, 0, 3) as u8,
                a: Pred::at(w, 12, 15),
                b: Pred::at(w, 29, 32),
                op1: BoolOp::from(bits(w, 24, 2)),
                op2: BoolOp::from(bits(w, 45, 2)),
                c: Pred::at(w, 39, 42),
            }
        }
        0x50 if m13 == 0x5088 => {
            return Op::Pset {
                d,
                a: Pred::at(w, 12, 15),
                b: Pred::at(w, 29, 32),
                op1: BoolOp::from(bits(w, 24, 2)),
                op2: BoolOp::from(bits(w, 45, 2)),
                c: Pred::at(w, 39, 42),
                bf: bit(w, 44),
            }
        }
        0x50 if m13 == 0x50a0 => {
            return Op::Csetp {
                p0: bits(w, 3, 3) as u8,
                p1: bits(w, 0, 3) as u8,
                cc: CcTest(bits(w, 8, 5) as u8),
                bop: BoolOp::from(bits(w, 45, 2)),
                p: Pred::at(w, 39, 42),
            }
        }
        0x5c if m13 == 0x5cf0 => {
            return Op::R2p {
                a,
                mask: Src::Reg(reg(w, 20)),
                byte: bits(w, 41, 2) as u8,
                cc: bit(w, 40),
            }
        }
        0x38 if m13 == 0x38f0 => {
            return Op::R2p {
                a,
                mask: Src::Imm(s20(w)),
                byte: bits(w, 41, 2) as u8,
                cc: bit(w, 40),
            }
        }
        0x5b | 0x36 | 0x51 | 0x4e | 0x4f if is_xmad(top16) => return xmad(w),
        0x5b if top16 & 0xfff0 == 0x5be0 => {
            return Op::Lop3 {
                d,
                a,
                b: Src::Reg(reg(w, 20)),
                c: reg(w, 39),
                lut: bits(w, 28, 8) as u8,
                cc: bit(w, 47),
            }
        }
        _ => {}
    }
    match top16 >> 10 {
        // lop3 imm (0x3c..) — 6-bit opcode.
        0x0f => {
            return Op::Lop3 {
                d,
                a,
                b: Src::Imm(s20(w)),
                c: reg(w, 39),
                lut: bits(w, 48, 8) as u8,
                cc: bit(w, 47),
            }
        }
        _ => {}
    }
    match top16 >> 9 {
        // lop3 cbuf (0x02..)
        0x01 => {
            return Op::Lop3 {
                d,
                a,
                b: cbuf(w),
                c: reg(w, 39),
                lut: bits(w, 48, 8) as u8,
                cc: bit(w, 47),
            }
        }
        _ => {}
    }
    match top16 >> 8 {
        0x1e => {
            return Op::Fmul {
                d,
                a,
                b: Src::Imm(bits(w, 20, 32) as u32),
                neg_b: false,
                sat: bit(w, 55),
                scale: 0,
            }
        }
        0x01 if top16 & 0xfff0 == 0x0100 => {
            return Op::Mov {
                d,
                s: Src::Imm(bits(w, 20, 32) as u32),
            }
        }
        _ => {}
    }
    match top16 >> 9 {
        0x0e => {
            // iadd32i (0x1c..)
            return Op::Iadd {
                d,
                a,
                b: Src::Imm(bits(w, 20, 32) as u32),
                neg_a: bit(w, 56),
                neg_b: false,
                x: bit(w, 53),
                cc: bit(w, 52),
                sat: bit(w, 54),
            };
        }
        _ => {}
    }
    match top16 >> 10 {
        0x02 => {
            // fadd32i (0x08..)
            return Op::Fadd {
                d,
                a,
                b: Src::Imm(bits(w, 20, 32) as u32),
                ma: FMod {
                    neg: bit(w, 56),
                    abs: bit(w, 54),
                },
                mb: FMod {
                    neg: bit(w, 53),
                    abs: bit(w, 57),
                },
                sat: false,
            };
        }
        0x01 => {
            // lop32i (0x04..)
            return Op::Lop {
                d,
                a,
                b: Src::Imm(bits(w, 20, 32) as u32),
                op: match bits(w, 53, 2) {
                    0 => LogicOp::And,
                    1 => LogicOp::Or,
                    2 => LogicOp::Xor,
                    _ => LogicOp::PassB,
                },
                inv_a: bit(w, 55),
                inv_b: bit(w, 56),
                x: bit(w, 57),
                cc: bit(w, 52),
                p_out: PT,
                p_mode: 0,
            };
        }
        _ => {}
    }
    Op::Unknown(w)
}

fn rounding(v: u64) -> Rounding {
    match v & 3 {
        0 => Rounding::Nearest,
        1 => Rounding::Floor,
        2 => Rounding::Ceil,
        _ => Rounding::Trunc,
    }
}

fn is_xmad(top16: u16) -> bool {
    (top16 & 0xffc0) == 0x5b00
        || (top16 & 0xfec0) == 0x3600
        || (top16 & 0xff80) == 0x5100
        || (top16 & 0xfe00) == 0x4e00
}

fn xmad(w: u64) -> Op {
    let top16 = (w >> 48) as u16;
    let d = reg(w, 0);
    let a = reg(w, 8);
    let signed_a = bit(w, 48);
    let signed_b = bit(w, 49);
    let cc = bit(w, 47);
    if top16 & 0xffc0 == 0x5b00 {
        Op::Xmad {
            d,
            a,
            b: Src::Reg(reg(w, 20)),
            c: Src::Reg(reg(w, 39)),
            signed_a,
            signed_b,
            hi_a: bit(w, 53),
            hi_b: bit(w, 35),
            mode: bits(w, 50, 3) as u8,
            psl: bit(w, 36),
            mrg: bit(w, 37),
            x: bit(w, 38),
            cc,
        }
    } else if top16 & 0xfec0 == 0x3600 {
        Op::Xmad {
            d,
            a,
            b: Src::Imm(bits(w, 20, 16) as u32),
            c: Src::Reg(reg(w, 39)),
            signed_a,
            signed_b: false,
            hi_a: bit(w, 53),
            hi_b: false,
            mode: bits(w, 50, 3) as u8,
            psl: bit(w, 36),
            mrg: bit(w, 37),
            x: bit(w, 38),
            cc,
        }
    } else if top16 & 0xff80 == 0x5100 {
        // xmad a, r39, c[] : B is the register at 39, C the constant.
        Op::Xmad {
            d,
            a,
            b: Src::Reg(reg(w, 39)),
            c: cbuf(w),
            signed_a,
            signed_b,
            hi_a: bit(w, 53),
            hi_b: bit(w, 52),
            mode: bits(w, 50, 2) as u8,
            psl: false,
            mrg: false,
            x: bit(w, 54),
            cc,
        }
    } else {
        // 0x4e.. xmad a, c[], r39
        Op::Xmad {
            d,
            a,
            b: cbuf(w),
            c: Src::Reg(reg(w, 39)),
            signed_a,
            signed_b,
            hi_a: bit(w, 53),
            hi_b: bit(w, 52),
            mode: bits(w, 50, 2) as u8,
            psl: bit(w, 55),
            mrg: bit(w, 56),
            x: bit(w, 54),
            cc,
        }
    }
}
