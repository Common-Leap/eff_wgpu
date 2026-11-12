//! Maxwell code to WGSL, one stage at a time.
//!
//! Registers become `u32` locals and every float operation goes through bitcasts, so values move
//! between integer and float instructions exactly as on the GPU. Straight-line programs are
//! written as straight-line WGSL. Programs that branch run as a block dispatcher (a loop over a
//! switch on the current block), which handles any control flow the compiler produced.
//!
//! Interface conventions (NVN):
//! - constant buffer bank = uniform block binding + 3; bank `b` is `@group(0) @binding(b)`;
//! - texture handle = 8 + 2 * sampler binding; slot `s` is `@group(1)` bindings `2s` (texture)
//!   and `2s + 1` (sampler);
//! - vertex input `a[0x80 + 16 i]` is `@location(i)`; varying `a[0x80 + 16 i]` is `@location(i)`;
//!   `a[0x70]` is the position; the pixel shader's colour is R0..R3 at exit.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use super::decode::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageKind {
    Vertex,
    Fragment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TexDim {
    D2,
    D3,
    Cube,
    Array2D,
}

/// What the translated stage expects to be bound.
#[derive(Debug, Clone, Default)]
pub struct Interface {
    /// Vertex attribute locations read (vertex) or varying locations read (fragment).
    pub inputs: BTreeSet<u32>,
    /// Varying locations written (vertex).
    pub outputs: BTreeSet<u32>,
    pub banks: BTreeSet<u8>,
    pub textures: BTreeMap<u32, TexDim>,
}

#[derive(Debug, Clone)]
pub struct Translation {
    pub source: String,
    pub interface: Interface,
    /// False when a pixel shader branches: its implicit-LOD samples then read mip 0.
    pub uniform_control_flow: bool,
}

/// Binding of a constant bank. Each stage has its own literal bank (c[1]); the pixel stage's is
/// moved to binding 3 so the two never share a slot.
pub fn binding_of(bank: u8, kind: StageKind) -> u32 {
    if bank == 1 && kind == StageKind::Fragment {
        3
    } else {
        bank as u32
    }
}

/// Size in 16-byte rows of every bound constant buffer (64 KiB, the hardware limit).
pub const CBUF_ROWS: u32 = 4096;
const DONE: u32 = 0xffff_ffff;

pub fn translate(kind: StageKind, insts: &[Inst]) -> Result<Translation, String> {
    let live = live_code(insts);
    let insts = &insts[..live];
    let mut t = Translator::new(kind);
    t.scan(insts)?;
    let structured = !insts
        .iter()
        .any(|i| matches!(i.op, Op::Bra { .. } | Op::Ssy { .. } | Op::Sync(_)));
    let mut body = String::new();
    if structured {
        for inst in insts {
            t.emit(&mut body, inst, "    ", None)?;
        }
        body.push_str("    return finish();\n");
    } else {
        t.emit_dispatch(&mut body, insts)?;
    }
    let source = t.module(&body, structured);
    Ok(Translation {
        source,
        interface: t.iface,
        uniform_control_flow: structured || kind == StageKind::Vertex,
    })
}

/// Everything up to the last instruction any path can reach: code after an unconditional exit
/// is kept only while something branches into it.
fn live_code(insts: &[Inst]) -> usize {
    let targets: BTreeSet<u32> = insts
        .iter()
        .filter_map(|i| match i.op {
            Op::Bra { target, .. } | Op::Ssy { target } => Some(target),
            _ => None,
        })
        .collect();
    let mut end = 0;
    let mut reachable = true;
    for (index, inst) in insts.iter().enumerate() {
        if targets.contains(&inst.addr) {
            reachable = true;
        }
        if !reachable {
            continue;
        }
        end = index + 1;
        let unconditional = inst.guard.is_always();
        match inst.op {
            Op::Exit(cc) if unconditional && cc == CcTest::ALWAYS => reachable = false,
            Op::Bra { target, cc } if unconditional && cc == CcTest::ALWAYS => {
                // A jump to itself parks a finished program.
                reachable = false;
                if target == inst.addr {
                    end = index;
                }
            }
            Op::Sync(_) if unconditional => reachable = false,
            _ => {}
        }
    }
    end
}

struct Translator {
    kind: StageKind,
    iface: Interface,
    regs: BTreeSet<u8>,
    preds: BTreeSet<u8>,
    blocks: BTreeMap<u32, u32>,
}

fn f(bits: &str) -> String {
    format!("bitcast<f32>({bits})")
}

fn u(value: &str) -> String {
    format!("bitcast<u32>({value})")
}

fn fmod(x: String, m: FMod) -> String {
    let x = if m.abs { format!("abs({x})") } else { x };
    if m.neg {
        format!("(-{x})")
    } else {
        x
    }
}

fn sat(x: String, on: bool) -> String {
    if on {
        format!("clamp({x}, 0.0, 1.0)")
    } else {
        x
    }
}

fn bool_op(a: &str, op: BoolOp, b: &str) -> String {
    match op {
        BoolOp::And => format!("({a} && {b})"),
        BoolOp::Or => format!("({a} || {b})"),
        BoolOp::Xor => format!("({a} != {b})"),
    }
}

fn float_cmp(a: &str, b: &str, cmp: FloatCmp) -> String {
    use FloatCmp::*;
    let nan = format!("({a} != {a} || {b} != {b})");
    match cmp {
        F => "false".into(),
        T => "true".into(),
        Lt => format!("({a} < {b})"),
        Eq => format!("({a} == {b})"),
        Le => format!("({a} <= {b})"),
        Gt => format!("({a} > {b})"),
        Ne => format!("({a} != {b} && !{nan})"),
        Ge => format!("({a} >= {b})"),
        Num => format!("(!{nan})"),
        Nan => nan,
        Ltu => format!("({a} < {b} || {nan})"),
        Equ => format!("({a} == {b} || {nan})"),
        Leu => format!("({a} <= {b} || {nan})"),
        Gtu => format!("({a} > {b} || {nan})"),
        Neu => format!("({a} != {b})"),
        Geu => format!("({a} >= {b} || {nan})"),
    }
}

fn int_cmp(a: &str, b: &str, cmp: IntCmp, signed: bool) -> String {
    let (a, b) = if signed {
        (format!("bitcast<i32>({a})"), format!("bitcast<i32>({b})"))
    } else {
        (a.to_string(), b.to_string())
    };
    match cmp {
        IntCmp::F => "false".into(),
        IntCmp::T => "true".into(),
        IntCmp::Lt => format!("({a} < {b})"),
        IntCmp::Eq => format!("({a} == {b})"),
        IntCmp::Le => format!("({a} <= {b})"),
        IntCmp::Gt => format!("({a} > {b})"),
        IntCmp::Ne => format!("({a} != {b})"),
        IntCmp::Ge => format!("({a} >= {b})"),
    }
}

impl Translator {
    fn new(kind: StageKind) -> Self {
        Self {
            kind,
            iface: Interface::default(),
            regs: BTreeSet::new(),
            preds: BTreeSet::new(),
            blocks: BTreeMap::new(),
        }
    }

    fn scan(&mut self, insts: &[Inst]) -> Result<(), String> {
        let note_src = |t: &mut Self, s: &Src| match *s {
            Src::Reg(r) => {
                t.regs.insert(r);
            }
            Src::Cbuf { bank, .. } => {
                t.iface.banks.insert(bank);
            }
            Src::Imm(_) => {}
        };
        for inst in insts {
            self.preds.insert(inst.guard.reg);
            match &inst.op {
                Op::Unknown(w) => {
                    return Err(format!("unknown instruction {w:016x} at {:#x}", inst.addr))
                }
                Op::Mov { s, .. }
                | Op::Rro { s, .. }
                | Op::I2f { s, .. }
                | Op::F2i { s, .. }
                | Op::F2f { s, .. }
                | Op::I2i { s, .. } => note_src(self, s),
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
                | Op::Shr { b, .. } => note_src(self, b),
                Op::Ffma { b, c, .. } | Op::Xmad { b, c, .. } => {
                    note_src(self, b);
                    note_src(self, c);
                }
                Op::R2p { mask, .. } => note_src(self, mask),
                Op::Ldc { bank, .. } => {
                    self.iface.banks.insert(*bank);
                }
                Op::Ald { offset, count, .. } => {
                    for c in 0..*count as u32 {
                        if let Some(loc) = generic(offset + 4 * c) {
                            self.iface.inputs.insert(loc);
                        }
                    }
                }
                Op::Ast { offset, count, .. } => {
                    for c in 0..*count as u32 {
                        if let Some(loc) = generic(offset + 4 * c) {
                            self.iface.outputs.insert(loc);
                        }
                    }
                }
                Op::Ipa { attr, .. } => {
                    if let Some(loc) = generic(*attr) {
                        self.iface.inputs.insert(loc);
                    }
                }
                Op::Texs { handle, target, .. } => {
                    let dim = match target {
                        TexTarget::D3 | TexTarget::D3LodZero => TexDim::D3,
                        TexTarget::Cube | TexTarget::CubeLodLevel => TexDim::Cube,
                        TexTarget::Array2D
                        | TexTarget::Array2DLodZero
                        | TexTarget::Array2DLodZeroDepthCompare => TexDim::Array2D,
                        _ => TexDim::D2,
                    };
                    self.iface.textures.insert(slot(*handle)?, dim);
                }
                Op::Tmml { handle, .. } => {
                    self.iface
                        .textures
                        .entry(slot(*handle)?)
                        .or_insert(TexDim::D2);
                }
                _ => {}
            }
            for r in written(&inst.op) {
                self.regs.insert(r);
            }
            for p in preds_of(&inst.op) {
                self.preds.insert(p);
            }
        }
        self.regs.remove(&RZ);
        self.preds.remove(&PT);
        // Pixel shaders hand their colour over in R0..R3.
        if self.kind == StageKind::Fragment {
            self.regs.extend(0..4);
        }
        Ok(())
    }

    fn r(&self, reg: u8) -> String {
        if reg == RZ {
            "0u".into()
        } else {
            format!("r{reg}")
        }
    }

    fn rf(&self, reg: u8) -> String {
        if reg == RZ {
            "0.0".into()
        } else {
            f(&format!("r{reg}"))
        }
    }

    fn src(&self, s: Src) -> String {
        match s {
            Src::Reg(r) => self.r(r),
            Src::Cbuf { bank, offset } => {
                format!("cb{bank}[{}u][{}u]", offset / 16, (offset / 4) % 4)
            }
            Src::Imm(v) => format!("{v:#x}u"),
        }
    }

    fn srcf(&self, s: Src) -> String {
        match s {
            Src::Imm(v) => {
                let value = f32::from_bits(v);
                if value.is_finite() {
                    format!("{value:?}")
                } else {
                    f(&format!("{v:#x}u"))
                }
            }
            _ => f(&self.src(s)),
        }
    }

    fn p(&self, p: Pred) -> String {
        let base = if p.reg == PT {
            "true".to_string()
        } else {
            format!("p{}", p.reg)
        };
        if p.not {
            format!("(!{base})")
        } else {
            base
        }
    }

    fn set(&self, out: &mut String, ind: &str, d: u8, value: String) {
        if d != RZ {
            let _ = writeln!(out, "{ind}r{d} = {value};");
        }
    }

    fn setp(&self, out: &mut String, ind: &str, p: u8, value: String) {
        if p != PT {
            let _ = writeln!(out, "{ind}p{p} = {value};");
        }
    }

    /// One instruction. `dispatch` is Some(next block index) inside the block dispatcher, where
    /// control-flow instructions set `pc`.
    fn emit(
        &mut self,
        out: &mut String,
        inst: &Inst,
        ind: &str,
        dispatch: Option<u32>,
    ) -> Result<(), String> {
        let guarded = !inst.guard.is_always();
        let inner_ind = if guarded {
            format!("{ind}    ")
        } else {
            ind.to_string()
        };
        let mut body = String::new();
        self.emit_op(&mut body, &inst.op, &inner_ind, dispatch, inst.addr)?;
        // Debug aid (VFX_NAN_TRACE): remember the first instruction that leaves a NaN behind.
        if std::env::var_os("VFX_NAN_TRACE").is_some() && !ends_block(&inst.op) {
            for r in written(&inst.op) {
                let _ = writeln!(
                    body,
                    "{inner_ind}if (dbg_nan == 0u && (r{r} & 0x7f800000u) == 0x7f800000u && (r{r} & 0x7fffffu) != 0u) {{ dbg_nan = {}u; }}",
                    inst.addr
                );
            }
        }
        if guarded {
            let _ = writeln!(out, "{ind}if ({}) {{", self.p(inst.guard));
            out.push_str(&body);
            if let Some(next) = dispatch {
                if ends_block(&inst.op) {
                    let _ = writeln!(out, "{ind}}} else {{\n{ind}    pc = {next}u;");
                }
            }
            let _ = writeln!(out, "{ind}}}");
        } else {
            out.push_str(&body);
        }
        Ok(())
    }

    fn emit_op(
        &mut self,
        out: &mut String,
        op: &Op,
        ind: &str,
        dispatch: Option<u32>,
        addr: u32,
    ) -> Result<(), String> {
        match *op {
            Op::Nop | Op::Unknown(_) => {}
            Op::Exit(cc) => {
                let test = cc_test(cc);
                match dispatch {
                    Some(next) => {
                        let _ = writeln!(
                            out,
                            "{ind}if ({test}) {{ pc = {DONE}u; }} else {{ pc = {next}u; }}"
                        );
                    }
                    None => {
                        let _ = writeln!(out, "{ind}if ({test}) {{ return finish(); }}");
                    }
                }
            }
            Op::Kil(cc) => {
                if self.kind == StageKind::Fragment {
                    let _ = writeln!(out, "{ind}if ({}) {{ discard; }}", cc_test(cc));
                }
            }
            Op::Bra { target, cc } => {
                let next = dispatch.ok_or("branch outside dispatcher")?;
                let to = self.block_of(target)?;
                let _ = writeln!(
                    out,
                    "{ind}if ({}) {{ pc = {to}u; }} else {{ pc = {next}u; }}",
                    cc_test(cc)
                );
            }
            Op::Ssy { target } => {
                let to = self.block_of(target)?;
                let _ = writeln!(out, "{ind}ssy[sp] = {to}u;\n{ind}sp = sp + 1u;");
            }
            Op::Sync(_) => {
                let _ = writeln!(out, "{ind}sp = sp - 1u;\n{ind}pc = ssy[sp];");
            }
            Op::Mov { d, s } => self.set(out, ind, d, self.src(s)),
            Op::S2r { d, sr } => {
                // Lane id, thread ids and clocks: a single invocation sees zeros.
                let _ = sr;
                self.set(out, ind, d, "0u".into());
            }
            Op::Sel { d, a, b, p } => {
                let v = format!("select({}, {}, {})", self.src(b), self.r(a), self.p(p));
                self.set(out, ind, d, v);
            }
            Op::Ffma {
                d,
                a,
                b,
                c,
                neg_b,
                neg_c,
                sat: s,
            } => {
                let b = fmod(
                    self.srcf(b),
                    FMod {
                        neg: neg_b,
                        abs: false,
                    },
                );
                let c = fmod(
                    self.srcf(c),
                    FMod {
                        neg: neg_c,
                        abs: false,
                    },
                );
                let v = sat(format!("fma({}, {b}, {c})", self.rf(a)), s);
                self.set(out, ind, d, u(&v));
            }
            Op::Fmul {
                d,
                a,
                b,
                neg_b,
                sat: s,
                scale,
            } => {
                let b = fmod(
                    self.srcf(b),
                    FMod {
                        neg: neg_b,
                        abs: false,
                    },
                );
                let mut v = format!("({} * {b})", self.rf(a));
                if scale != 0 {
                    v = format!("({v} * {:?})", 2f32.powi(scale as i32));
                }
                self.set(out, ind, d, u(&sat(v, s)));
            }
            Op::Fadd {
                d,
                a,
                b,
                ma,
                mb,
                sat: s,
            } => {
                let v = format!("({} + {})", fmod(self.rf(a), ma), fmod(self.srcf(b), mb));
                self.set(out, ind, d, u(&sat(v, s)));
            }
            Op::Fmnmx { d, a, b, ma, mb, p } => {
                let (a, b) = (fmod(self.rf(a), ma), fmod(self.srcf(b), mb));
                let v = format!("select(max({a}, {b}), min({a}, {b}), {})", self.p(p));
                self.set(out, ind, d, u(&v));
            }
            Op::Mufu {
                d,
                a,
                op,
                m,
                sat: s,
            } => {
                let x = fmod(self.rf(a), m);
                let v = match op {
                    MufuOp::Cos => format!("cos({x})"),
                    MufuOp::Sin => format!("sin({x})"),
                    MufuOp::Ex2 => format!("exp2({x})"),
                    MufuOp::Lg2 => format!("log2({x})"),
                    MufuOp::Rcp => format!("(1.0 / {x})"),
                    MufuOp::Rsq => format!("inverseSqrt({x})"),
                    MufuOp::Sqrt => format!("sqrt({x})"),
                };
                self.set(out, ind, d, u(&sat(v, s)));
            }
            // Range reduction feeds MUFU's sin/cos/ex2, which here take the plain value.
            Op::Rro { d, s, m } => {
                let v = fmod(self.srcf(s), m);
                self.set(out, ind, d, u(&v));
            }
            Op::Fset {
                d,
                a,
                b,
                ma,
                mb,
                cmp,
                bop,
                p,
                bf,
            } => {
                let c = float_cmp(&fmod(self.rf(a), ma), &fmod(self.srcf(b), mb), cmp);
                let c = bool_op(&c, bop, &self.p(p));
                let (t, fv) = if bf {
                    ("0x3f800000u", "0u")
                } else {
                    ("0xffffffffu", "0u")
                };
                self.set(out, ind, d, format!("select({fv}, {t}, {c})"));
            }
            Op::Fsetp {
                p0,
                p1,
                a,
                b,
                ma,
                mb,
                cmp,
                bop,
                p,
            } => {
                let c = float_cmp(&fmod(self.rf(a), ma), &fmod(self.srcf(b), mb), cmp);
                let _ = writeln!(out, "{ind}{{\n{ind}    let c = {c};");
                let inner = format!("{ind}    ");
                self.setp(out, &inner, p0, bool_op("c", bop, &self.p(p)));
                self.setp(out, &inner, p1, bool_op("!c", bop, &self.p(p)));
                let _ = writeln!(out, "{ind}}}");
            }
            Op::Isetp {
                p0,
                p1,
                a,
                b,
                cmp,
                signed,
                bop,
                p,
                ..
            } => {
                let c = int_cmp(&self.r(a), &self.src(b), cmp, signed);
                let _ = writeln!(out, "{ind}{{\n{ind}    let c = {c};");
                let inner = format!("{ind}    ");
                self.setp(out, &inner, p0, bool_op("c", bop, &self.p(p)));
                self.setp(out, &inner, p1, bool_op("!c", bop, &self.p(p)));
                let _ = writeln!(out, "{ind}}}");
            }
            Op::Iset {
                d,
                a,
                b,
                cmp,
                signed,
                bop,
                p,
                bf,
                ..
            } => {
                let c = int_cmp(&self.r(a), &self.src(b), cmp, signed);
                let c = bool_op(&c, bop, &self.p(p));
                let t = if bf { "0x3f800000u" } else { "0xffffffffu" };
                self.set(out, ind, d, format!("select(0u, {t}, {c})"));
            }
            Op::Psetp {
                p0,
                p1,
                a,
                b,
                op1,
                op2,
                c,
            } => {
                let first = bool_op(&self.p(a), op1, &self.p(b));
                let _ = writeln!(out, "{ind}{{\n{ind}    let c = {first};");
                let inner = format!("{ind}    ");
                self.setp(out, &inner, p0, bool_op("c", op2, &self.p(c)));
                self.setp(out, &inner, p1, bool_op("!c", op2, &self.p(c)));
                let _ = writeln!(out, "{ind}}}");
            }
            Op::Pset {
                d,
                a,
                b,
                op1,
                op2,
                c,
                bf,
            } => {
                let v = bool_op(&bool_op(&self.p(a), op1, &self.p(b)), op2, &self.p(c));
                let t = if bf { "0x3f800000u" } else { "0xffffffffu" };
                self.set(out, ind, d, format!("select(0u, {t}, {v})"));
            }
            Op::Csetp { p0, p1, cc, bop, p } => {
                let _ = writeln!(out, "{ind}{{\n{ind}    let c = {};", cc_test(cc));
                let inner = format!("{ind}    ");
                self.setp(out, &inner, p0, bool_op("c", bop, &self.p(p)));
                self.setp(out, &inner, p1, bool_op("!c", bop, &self.p(p)));
                let _ = writeln!(out, "{ind}}}");
            }
            Op::R2p { a, mask, byte, cc } => {
                if !cc {
                    let shift = byte as u32 * 8;
                    let _ = writeln!(out, "{ind}{{\n{ind}    let m = {} >> {shift}u;\n{ind}    let v = {} >> {shift}u;", self.src(mask), self.r(a));
                    for i in 0..7u32 {
                        let _ = writeln!(out, "{ind}    if ((m >> {i}u) & 1u) != 0u {{ p{i} = ((v >> {i}u) & 1u) != 0u; }}");
                        self.preds.insert(i as u8);
                    }
                    let _ = writeln!(out, "{ind}}}");
                }
            }
            Op::I2f {
                d,
                s,
                from,
                m,
                round,
                byte,
                ..
            } => {
                let raw = self.src(s);
                let field = int_field(&raw, from, byte);
                let v = if from.signed {
                    format!("f32(bitcast<i32>({field}))")
                } else {
                    format!("f32({field})")
                };
                let _ = round;
                self.set(out, ind, d, u(&fmod(v, m)));
            }
            Op::F2i { d, s, to, m, round } => {
                let x = rounded(fmod(self.srcf(s), m), Some(round));
                let v = if to.signed {
                    let (lo, hi) = signed_range(to.bytes);
                    format!("bitcast<u32>(i32(clamp({x}, {lo:?}, {hi:?})))")
                } else {
                    let hi = unsigned_max(to.bytes);
                    format!("u32(clamp({x}, 0.0, {hi:?}))")
                };
                self.set(out, ind, d, v);
            }
            Op::F2f {
                d,
                s,
                m,
                round,
                sat: st,
            } => {
                let v = sat(rounded(fmod(self.srcf(s), m), round), st);
                self.set(out, ind, d, u(&v));
            }
            Op::I2i {
                d,
                s,
                from,
                to,
                neg,
                abs,
                sat: st,
                byte,
            } => {
                let field = int_field(&self.src(s), from, byte);
                let mut v = if from.signed {
                    format!("bitcast<i32>({field})")
                } else {
                    format!("i32({field})")
                };
                if abs {
                    v = format!("abs({v})");
                }
                if neg {
                    v = format!("(-{v})");
                }
                let v = if st {
                    if to.signed {
                        let (lo, hi) = signed_range(to.bytes);
                        format!("bitcast<u32>(clamp({v}, {}, {}))", lo as i64, hi as i64)
                    } else {
                        format!("u32(clamp({v}, 0, {}))", unsigned_max(to.bytes) as i64)
                    }
                } else {
                    format!("bitcast<u32>({v})")
                };
                self.set(out, ind, d, v);
            }
            Op::Iadd {
                d,
                a,
                b,
                neg_a,
                neg_b,
                x,
                cc,
                ..
            } => {
                let a = if neg_a {
                    format!("(0u - {})", self.r(a))
                } else {
                    self.r(a)
                };
                let b = if neg_b {
                    format!("(0u - {})", self.src(b))
                } else {
                    self.src(b)
                };
                let _ = writeln!(
                    out,
                    "{ind}{{\n{ind}    let a = {a};\n{ind}    let s = a + {b};"
                );
                let _ = writeln!(out, "{ind}    let r = s + select(0u, 1u, {} && cc_c);", x);
                if cc {
                    let _ = writeln!(out, "{ind}    cc_c = s < a || r < s;\n{ind}    cc_z = r == 0u;\n{ind}    cc_s = (r >> 31u) != 0u;\n{ind}    cc_o = false;");
                }
                if d != RZ {
                    let _ = writeln!(out, "{ind}    r{d} = r;");
                }
                let _ = writeln!(out, "{ind}}}");
            }
            Op::Iadd3 {
                d, a, b, c, neg, ..
            } => {
                let n = |v: String, on: bool| if on { format!("(0u - {v})") } else { v };
                let v = format!(
                    "({} + {} + {})",
                    n(self.r(a), neg[0]),
                    n(self.src(b), neg[1]),
                    n(self.r(c), neg[2])
                );
                self.set(out, ind, d, v);
            }
            Op::Iscadd {
                d,
                a,
                b,
                shift,
                neg_a,
                neg_b,
            } => {
                let a = if neg_a {
                    format!("(0u - {})", self.r(a))
                } else {
                    self.r(a)
                };
                let b = if neg_b {
                    format!("(0u - {})", self.src(b))
                } else {
                    self.src(b)
                };
                self.set(out, ind, d, format!("(({a} << {shift}u) + {b})"));
            }
            Op::Xmad {
                d,
                a,
                b,
                c,
                signed_a,
                signed_b,
                hi_a,
                hi_b,
                mode,
                psl,
                mrg,
                x,
                cc,
            } => {
                let half = |v: String, hi: bool, signed: bool| {
                    let raw = if hi {
                        format!("({v} >> 16u)")
                    } else {
                        format!("({v} & 0xffffu)")
                    };
                    if signed {
                        format!("bitcast<u32>((bitcast<i32>({raw} << 16u)) >> 16u)")
                    } else {
                        format!("({raw} & 0xffffu)")
                    }
                };
                let a16 = half(self.r(a), hi_a, signed_a);
                let b_raw = self.src(b);
                let b16 = half(b_raw.clone(), hi_b, signed_b);
                let c = self.src(c);
                let _ = writeln!(out, "{ind}{{\n{ind}    var p = {a16} * {b16};");
                if psl {
                    let _ = writeln!(out, "{ind}    p = p << 16u;");
                }
                let cv = match mode {
                    1 => format!("({c} & 0xffffu)"),
                    2 => format!("({c} >> 16u)"),
                    4 => format!("({c} + (({b_raw} & 0xffffu) << 16u))"),
                    _ => c,
                };
                let _ = writeln!(
                    out,
                    "{ind}    let s = p + {cv};\n{ind}    var r = s + select(0u, 1u, {x} && cc_c);"
                );
                if cc {
                    let _ = writeln!(out, "{ind}    cc_c = s < p || r < s;\n{ind}    cc_z = r == 0u;\n{ind}    cc_s = (r >> 31u) != 0u;");
                }
                if mrg {
                    let _ = writeln!(
                        out,
                        "{ind}    r = (r & 0xffffu) | (({b_raw} & 0xffffu) << 16u);"
                    );
                }
                if d != RZ {
                    let _ = writeln!(out, "{ind}    r{d} = r;");
                }
                let _ = writeln!(out, "{ind}}}");
            }
            Op::Lop {
                d,
                a,
                b,
                op,
                inv_a,
                inv_b,
                p_out,
                p_mode,
                ..
            } => {
                let a = if inv_a {
                    format!("(~{})", self.r(a))
                } else {
                    self.r(a)
                };
                let b = if inv_b {
                    format!("(~{})", self.src(b))
                } else {
                    self.src(b)
                };
                let v = match op {
                    LogicOp::And => format!("({a} & {b})"),
                    LogicOp::Or => format!("({a} | {b})"),
                    LogicOp::Xor => format!("({a} ^ {b})"),
                    LogicOp::PassB => b,
                };
                let _ = writeln!(out, "{ind}{{\n{ind}    let v = {v};");
                let inner = format!("{ind}    ");
                match p_mode {
                    1 => self.setp(out, &inner, p_out, "true".into()),
                    2 => self.setp(out, &inner, p_out, "v == 0u".into()),
                    3 => self.setp(out, &inner, p_out, "v != 0u".into()),
                    _ => {}
                }
                if d != RZ {
                    let _ = writeln!(out, "{inner}r{d} = v;");
                }
                let _ = writeln!(out, "{ind}}}");
            }
            Op::Lop3 {
                d, a, b, c, lut, ..
            } => {
                let (a, b, c) = (self.r(a), self.src(b), self.r(c));
                let mut terms = Vec::new();
                for i in 0..8u8 {
                    if lut & (1 << i) != 0 {
                        let pick = |bit: u8, v: &str| {
                            if i & bit != 0 {
                                v.to_string()
                            } else {
                                format!("~{v}")
                            }
                        };
                        terms.push(format!(
                            "({} & {} & {})",
                            pick(4, &a),
                            pick(2, &b),
                            pick(1, &c)
                        ));
                    }
                }
                let v = if terms.is_empty() {
                    "0u".into()
                } else {
                    terms.join(" | ")
                };
                self.set(out, ind, d, format!("({v})"));
            }
            Op::Shl { d, a, b, wrap } => {
                let b = self.src(b);
                let v = if wrap {
                    format!("({} << ({b} & 31u))", self.r(a))
                } else {
                    format!("select(0u, {} << min({b}, 31u), {b} < 32u)", self.r(a))
                };
                self.set(out, ind, d, v);
            }
            Op::Shr {
                d,
                a,
                b,
                signed,
                wrap,
            } => {
                let b = self.src(b);
                let amount = if wrap {
                    format!("({b} & 31u)")
                } else {
                    format!("min({b}, 31u)")
                };
                let v = if signed {
                    format!("bitcast<u32>(bitcast<i32>({}) >> {amount})", self.r(a))
                } else if wrap {
                    format!("({} >> {amount})", self.r(a))
                } else {
                    format!("select(0u, {} >> {amount}, {b} < 32u)", self.r(a))
                };
                self.set(out, ind, d, v);
            }
            Op::Ald {
                d,
                base,
                offset,
                count,
                output,
            } => {
                for c in 0..count as u32 {
                    let reg = d.wrapping_add(c as u8);
                    let v = if base != RZ {
                        format!("load_attr({} + {}u)", self.r(base), offset + 4 * c)
                    } else if output {
                        self.output_attr(offset + 4 * c)
                    } else {
                        self.input_attr(offset + 4 * c)
                    };
                    self.set(out, ind, reg, v);
                }
            }
            Op::Ast {
                s,
                base,
                offset,
                count,
            } => {
                if base != RZ {
                    return Err(format!("indexed attribute store at {addr:#x}"));
                }
                for c in 0..count as u32 {
                    if let Some(target) = output_slot(offset + 4 * c) {
                        let _ = writeln!(
                            out,
                            "{ind}{target} = {};",
                            f(&self.r(s.wrapping_add(c as u8)))
                        );
                    }
                }
            }
            Op::Ldc {
                d,
                base,
                bank,
                offset,
                bytes,
                signed,
            } => {
                let at = format!("(({} + {}u) & 0xfffcu)", self.r(base), offset as u32);
                let word = |at: &str| format!("cb{bank}[({at}) >> 4u][(({at}) >> 2u) & 3u]");
                match bytes {
                    8 => {
                        self.set(out, ind, d, word(&at));
                        self.set(out, ind, d.wrapping_add(1), word(&format!("({at} + 4u)")));
                    }
                    4 => self.set(out, ind, d, word(&at)),
                    n => {
                        let shift =
                            format!("((({} + {}u) & 3u) * 8u)", self.r(base), offset as u32);
                        let bits = n as u32 * 8;
                        let v = if signed {
                            format!(
                                "bitcast<u32>(extractBits(bitcast<i32>({}), {shift}, {bits}u))",
                                word(&at)
                            )
                        } else {
                            format!("extractBits({}, {shift}, {bits}u)", word(&at))
                        };
                        self.set(out, ind, d, v);
                    }
                }
            }
            Op::Ipa {
                d,
                attr,
                mode,
                sat: st,
                ..
            } => {
                let v = match attr {
                    0x70 => "frag_coord.x".to_string(),
                    0x74 => "frag_coord.y".to_string(),
                    0x78 => "frag_coord.z".to_string(),
                    0x7c => "frag_coord.w".to_string(),
                    other => match generic(other) {
                        Some(loc) => format!("vin.v{loc}[{}]", (other % 16) / 4),
                        None => "0.0".to_string(),
                    },
                };
                let _ = mode;
                self.set(out, ind, d, u(&sat(v, st)));
            }
            Op::Texs {
                d0,
                d1,
                a,
                b,
                handle,
                target,
                mask,
            } => {
                let s = slot(handle)?;
                let (coords, level): (String, Option<String>) = match target {
                    TexTarget::D1LodZero => (
                        format!("vec2<f32>({}, 0.5)", self.rf(a)),
                        Some("0.0".into()),
                    ),
                    TexTarget::D2 | TexTarget::D2DepthCompare => {
                        (format!("vec2<f32>({}, {})", self.rf(a), self.rf(b)), None)
                    }
                    TexTarget::D2LodZero | TexTarget::D2LodZeroDepthCompare => (
                        format!("vec2<f32>({}, {})", self.rf(a), self.rf(b)),
                        Some("0.0".into()),
                    ),
                    TexTarget::D2LodLevel | TexTarget::D2LodLevelDepthCompare => (
                        format!("vec2<f32>({}, {})", self.rf(a), self.rf(a.wrapping_add(1))),
                        Some(self.rf(b)),
                    ),
                    TexTarget::Array2D
                    | TexTarget::Array2DLodZero
                    | TexTarget::Array2DLodZeroDepthCompare => (
                        format!(
                            "vec2<f32>({}, {}), i32({})",
                            self.rf(a.wrapping_add(1)),
                            self.rf(b),
                            self.r(a)
                        ),
                        Some("0.0".into()),
                    ),
                    TexTarget::D3 | TexTarget::Cube => (
                        format!(
                            "vec3<f32>({}, {}, {})",
                            self.rf(a),
                            self.rf(a.wrapping_add(1)),
                            self.rf(b)
                        ),
                        None,
                    ),
                    TexTarget::D3LodZero => (
                        format!(
                            "vec3<f32>({}, {}, {})",
                            self.rf(a),
                            self.rf(a.wrapping_add(1)),
                            self.rf(b)
                        ),
                        Some("0.0".into()),
                    ),
                    TexTarget::CubeLodLevel => (
                        format!(
                            "vec3<f32>({}, {}, {})",
                            self.rf(a),
                            self.rf(a.wrapping_add(1)),
                            self.rf(b)
                        ),
                        Some(self.rf(b.wrapping_add(1))),
                    ),
                };
                let sample = match level {
                    Some(level) => format!("textureSampleLevel(tex{s}, smp{s}, {coords}, {level})"),
                    None => format!("sample_implicit_{s}({coords})"),
                };
                let comps: &[usize] = if d1 == RZ {
                    match mask {
                        0 => &[0],
                        1 => &[1],
                        2 => &[2],
                        3 => &[3],
                        4 => &[0, 1],
                        5 => &[0, 3],
                        6 => &[1, 3],
                        _ => &[2, 3],
                    }
                } else {
                    match mask {
                        0 => &[0, 1, 2],
                        1 => &[0, 1, 3],
                        2 => &[0, 2, 3],
                        3 => &[1, 2, 3],
                        _ => &[0, 1, 2, 3],
                    }
                };
                let _ = writeln!(out, "{ind}{{\n{ind}    let t = {sample};");
                let inner = format!("{ind}    ");
                for (i, comp) in comps.iter().enumerate() {
                    let reg = match i {
                        0 => d0,
                        1 => d0.wrapping_add(1),
                        2 => d1,
                        _ => d1.wrapping_add(1),
                    };
                    if reg != RZ {
                        let _ = writeln!(out, "{inner}r{reg} = bitcast<u32>(t[{comp}]);");
                    }
                }
                let _ = writeln!(out, "{ind}}}");
            }
            Op::Tmml { d, a, handle, .. } => {
                let s = slot(handle)?;
                let v = format!(
                    "query_lod_{s}(vec2<f32>({}, {}))",
                    self.rf(a),
                    self.rf(a.wrapping_add(1))
                );
                let _ = writeln!(
                    out,
                    "{ind}{{\n{ind}    let l = u32(clamp({v}, 0.0, 255.0) * 256.0);"
                );
                if d != RZ {
                    let _ = writeln!(out, "{ind}    r{d} = l;");
                }
                if d.wrapping_add(1) != RZ {
                    let _ = writeln!(out, "{ind}    r{} = l;", d.wrapping_add(1));
                }
                let _ = writeln!(out, "{ind}}}");
            }
        }
        Ok(())
    }

    fn input_attr(&self, offset: u32) -> String {
        match offset {
            0x2f8 => "instance_id".into(),
            0x2fc => "vertex_id".into(),
            other => match generic(other) {
                Some(loc) if self.kind == StageKind::Vertex => {
                    u(&format!("vin.a{loc}[{}]", (other % 16) / 4))
                }
                _ => "0u".into(),
            },
        }
    }

    fn output_attr(&self, offset: u32) -> String {
        match output_slot(offset) {
            Some(slot) => u(&slot),
            None => "0u".into(),
        }
    }

    fn block_of(&self, addr: u32) -> Result<u32, String> {
        self.blocks
            .get(&addr)
            .copied()
            .ok_or_else(|| format!("branch to {addr:#x} which starts no block"))
    }

    fn emit_dispatch(&mut self, out: &mut String, insts: &[Inst]) -> Result<(), String> {
        let mut leaders = BTreeSet::new();
        leaders.insert(insts[0].addr);
        for (i, inst) in insts.iter().enumerate() {
            match inst.op {
                Op::Bra { target, .. } | Op::Ssy { target } => {
                    leaders.insert(target);
                }
                _ => {}
            }
            if ends_block(&inst.op) {
                if let Some(next) = insts.get(i + 1) {
                    leaders.insert(next.addr);
                }
            }
        }
        for (index, addr) in leaders.iter().enumerate() {
            self.blocks.insert(*addr, index as u32);
        }
        let _ = writeln!(out, "    var pc = 0u;\n    var ssy: array<u32, 16>;\n    var sp = 0u;\n    dbg_steps = 0u;\n    loop {{\n        dbg_steps = dbg_steps + 1u;\n        dbg_pc = pc;\n        if (dbg_steps > 100000u) {{ break; }}\n        switch pc {{");
        let mut i = 0;
        while i < insts.len() {
            let start = insts[i].addr;
            let index = self.blocks[&start];
            let _ = writeln!(out, "            case {index}u: {{");
            let mut closed = false;
            while i < insts.len() {
                let inst = &insts[i];
                let next_addr = insts.get(i + 1).map(|n| n.addr);
                let next_block = next_addr
                    .and_then(|a| self.blocks.get(&a).copied())
                    .unwrap_or(DONE);
                if ends_block(&inst.op) {
                    self.emit(out, inst, "                ", Some(next_block))?;
                    i += 1;
                    closed = true;
                    break;
                }
                self.emit(out, inst, "                ", Some(next_block))?;
                i += 1;
                if next_addr.is_some_and(|a| self.blocks.contains_key(&a)) {
                    let _ = writeln!(out, "                pc = {next_block}u;");
                    closed = true;
                    break;
                }
            }
            if !closed {
                let _ = writeln!(out, "                pc = {DONE}u;");
            }
            let _ = writeln!(out, "            }}");
        }
        let _ = writeln!(out, "            default: {{ pc = {DONE}u; }}\n        }}\n        if (pc == {DONE}u) {{ break; }}\n    }}\n    return finish();");
        Ok(())
    }

    fn module(&self, body: &str, structured: bool) -> String {
        let mut m = String::new();
        for bank in &self.iface.banks {
            let _ = writeln!(
                m,
                "@group(0) @binding({}) var<uniform> cb{bank}: array<vec4<u32>, {CBUF_ROWS}>;",
                binding_of(*bank, self.kind)
            );
        }
        for (slot, dim) in &self.iface.textures {
            let ty = match dim {
                TexDim::D2 => "texture_2d<f32>",
                TexDim::D3 => "texture_3d<f32>",
                TexDim::Cube => "texture_cube<f32>",
                TexDim::Array2D => "texture_2d_array<f32>",
            };
            let _ = writeln!(
                m,
                "@group(1) @binding({}) var tex{slot}: {ty};\n@group(1) @binding({}) var smp{slot}: sampler;",
                slot * 2,
                slot * 2 + 1
            );
            let coord = match dim {
                TexDim::D3 | TexDim::Cube => "vec3<f32>",
                _ => "vec2<f32>",
            };
            let implicit = if self.kind == StageKind::Fragment && structured {
                format!("textureSample(tex{slot}, smp{slot}, uv)")
            } else {
                format!("textureSampleLevel(tex{slot}, smp{slot}, uv, 0.0)")
            };
            if *dim != TexDim::Array2D {
                let _ = writeln!(
                    m,
                    "fn sample_implicit_{slot}(uv: {coord}) -> vec4<f32> {{ return {implicit}; }}"
                );
            }
            if *dim == TexDim::D2 {
                let lod = if self.kind == StageKind::Fragment && structured {
                    format!("let size = vec2<f32>(textureDimensions(tex{slot}));\n    let dx = dpdx(uv) * size;\n    let dy = dpdy(uv) * size;\n    return max(0.5 * log2(max(dot(dx, dx), dot(dy, dy))), 0.0);")
                } else {
                    "return 0.0;".to_string()
                };
                let _ = writeln!(
                    m,
                    "fn query_lod_{slot}(uv: vec2<f32>) -> f32 {{\n    {lod}\n}}"
                );
            }
        }
        let regs: String = self
            .regs
            .iter()
            .map(|r| format!("var<private> r{r}: u32;\n"))
            .collect();
        let preds: String = self
            .preds
            .iter()
            .map(|p| format!("var<private> p{p}: bool;\n"))
            .collect();
        m.push_str(&regs);
        m.push_str(&preds);
        m.push_str(
            "var<private> dbg_pc: u32;\nvar<private> dbg_steps: u32;\nvar<private> dbg_nan: u32;\n",
        );
        m.push_str("var<private> cc_c: bool;\nvar<private> cc_z: bool;\nvar<private> cc_s: bool;\nvar<private> cc_o: bool;\n");
        m.push_str(CC_TEST_FN);
        match self.kind {
            StageKind::Vertex => {
                m.push_str("struct VertexIn {\n");
                for loc in &self.iface.inputs {
                    let _ = writeln!(m, "    @location({loc}) a{loc}: vec4<f32>,");
                }
                if self.iface.inputs.is_empty() {
                    m.push_str("    @builtin(vertex_index) unused_vertex: u32,\n");
                }
                m.push_str("};\nstruct VertexOut {\n    @builtin(position) pos: vec4<f32>,\n");
                for loc in &self.iface.outputs {
                    let _ = writeln!(m, "    @location({loc}) v{loc}: vec4<f32>,");
                }
                m.push_str("};\nvar<private> vin: VertexIn;\nvar<private> vout: VertexOut;\nvar<private> vertex_id: u32;\nvar<private> instance_id: u32;\n");
                m.push_str("fn finish() -> VertexOut { return vout; }\n");
                m.push_str(&self.load_attr_fn());
                m.push_str("fn run() -> VertexOut {\n");
                m.push_str(body);
                m.push_str("}\n@vertex\nfn main(input: VertexIn, @builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VertexOut {\n    vin = input;\n    vertex_id = vid;\n    instance_id = iid;\n    return run();\n}\n");
            }
            StageKind::Fragment => {
                m.push_str("struct FragmentIn {\n    @builtin(position) pos: vec4<f32>,\n");
                for loc in &self.iface.inputs {
                    let _ = writeln!(m, "    @location({loc}) v{loc}: vec4<f32>,");
                }
                m.push_str(
                    "};\nvar<private> vin: FragmentIn;\nvar<private> frag_coord: vec4<f32>;\n",
                );
                m.push_str("fn finish() -> vec4<f32> { return vec4<f32>(bitcast<f32>(r0), bitcast<f32>(r1), bitcast<f32>(r2), bitcast<f32>(r3)); }\n");
                m.push_str("fn run() -> vec4<f32> {\n");
                m.push_str(body);
                m.push_str("}\n@fragment\nfn main(input: FragmentIn) -> @location(0) vec4<f32> {\n    vin = input;\n    frag_coord = vec4<f32>(input.pos.xyz, 1.0 / input.pos.w);\n    return run();\n}\n");
            }
        }
        m
    }

    fn load_attr_fn(&self) -> String {
        let mut s = String::from("fn load_attr(at: u32) -> u32 {\n    switch at {\n");
        for loc in &self.iface.inputs {
            for c in 0..4u32 {
                let _ = writeln!(
                    s,
                    "        case {}u: {{ return bitcast<u32>(vin.a{loc}[{c}]); }}",
                    0x80 + loc * 16 + c * 4
                );
            }
        }
        s.push_str("        default: { return 0u; }\n    }\n}\n");
        s
    }
}

const CC_TEST_FN: &str = "fn sp_dbg() -> u32 { return dbg_nan; }
fn cc(test: u32) -> bool {
    switch test {
        case 0u: { return false; }
        case 1u: { return cc_s != cc_o; }
        case 2u: { return cc_z; }
        case 3u: { return cc_z || (cc_s != cc_o); }
        case 4u: { return !cc_z && (cc_s == cc_o); }
        case 5u: { return !cc_z; }
        case 6u: { return cc_s == cc_o; }
        case 17u: { return !cc_c; }
        case 18u: { return !cc_s; }
        case 19u: { return !cc_c || cc_z; }
        case 20u: { return cc_c && !cc_z; }
        case 21u: { return cc_s; }
        case 22u: { return cc_c; }
        case 23u: { return cc_o; }
        case 16u: { return !cc_o; }
        default: { return true; }
    }
}
";

fn cc_test(cc: CcTest) -> String {
    if cc == CcTest::ALWAYS {
        "true".into()
    } else {
        format!("cc({}u)", cc.0)
    }
}

fn ends_block(op: &Op) -> bool {
    matches!(op, Op::Bra { .. } | Op::Exit(_) | Op::Sync(_))
}

fn written(op: &Op) -> Vec<u8> {
    match *op {
        Op::Mov { d, .. }
        | Op::S2r { d, .. }
        | Op::Sel { d, .. }
        | Op::Ffma { d, .. }
        | Op::Fmul { d, .. }
        | Op::Fadd { d, .. }
        | Op::Fmnmx { d, .. }
        | Op::Mufu { d, .. }
        | Op::Rro { d, .. }
        | Op::Fset { d, .. }
        | Op::Iset { d, .. }
        | Op::Pset { d, .. }
        | Op::I2f { d, .. }
        | Op::F2i { d, .. }
        | Op::F2f { d, .. }
        | Op::I2i { d, .. }
        | Op::Iadd { d, .. }
        | Op::Iadd3 { d, .. }
        | Op::Iscadd { d, .. }
        | Op::Xmad { d, .. }
        | Op::Lop { d, .. }
        | Op::Lop3 { d, .. }
        | Op::Shl { d, .. }
        | Op::Shr { d, .. }
        | Op::Ipa { d, .. } => vec![d],
        Op::Ald { d, count, .. } => (0..count).map(|c| d.wrapping_add(c)).collect(),
        Op::Ldc { d, bytes, .. } => {
            if bytes == 8 {
                vec![d, d.wrapping_add(1)]
            } else {
                vec![d]
            }
        }
        Op::Texs { d0, d1, .. } => vec![d0, d0.wrapping_add(1), d1, d1.wrapping_add(1)],
        Op::Tmml { d, .. } => vec![d, d.wrapping_add(1)],
        _ => Vec::new(),
    }
    .into_iter()
    .filter(|r| *r != RZ)
    .collect()
}

fn preds_of(op: &Op) -> Vec<u8> {
    match *op {
        Op::Fsetp { p0, p1, p, .. } | Op::Isetp { p0, p1, p, .. } | Op::Csetp { p0, p1, p, .. } => {
            vec![p0, p1, p.reg]
        }
        Op::Psetp {
            p0, p1, a, b, c, ..
        } => vec![p0, p1, a.reg, b.reg, c.reg],
        Op::Pset { a, b, c, .. } => vec![a.reg, b.reg, c.reg],
        Op::Sel { p, .. } | Op::Fmnmx { p, .. } | Op::Fset { p, .. } | Op::Iset { p, .. } => {
            vec![p.reg]
        }
        Op::Lop { p_out, .. } => vec![p_out],
        _ => Vec::new(),
    }
}

/// Generic attribute / varying location of an attribute byte offset.
fn generic(offset: u32) -> Option<u32> {
    (0x80..0x280)
        .contains(&offset)
        .then(|| (offset - 0x80) / 16)
}

fn output_slot(offset: u32) -> Option<String> {
    match offset {
        0x70..=0x7c => Some(format!("vout.pos[{}]", (offset - 0x70) / 4)),
        other => generic(other).map(|loc| format!("vout.v{loc}[{}]", (other % 16) / 4)),
    }
}

fn slot(handle: u32) -> Result<u32, String> {
    if handle < 8 || handle % 2 != 0 {
        return Err(format!(
            "texture handle {handle} is not a bound sampler slot"
        ));
    }
    Ok((handle - 8) / 2)
}

fn int_field(raw: &str, ty: IntType, byte: u8) -> String {
    let bits = ty.bytes as u32 * 8;
    if bits >= 32 {
        raw.to_string()
    } else {
        let shift = byte as u32 * 8;
        if ty.signed {
            format!("bitcast<u32>(extractBits(bitcast<i32>({raw}), {shift}u, {bits}u))")
        } else {
            format!("extractBits({raw}, {shift}u, {bits}u)")
        }
    }
}

fn rounded(x: String, round: Option<Rounding>) -> String {
    match round {
        None => x,
        Some(Rounding::Nearest) => format!("round({x})"),
        Some(Rounding::Floor) => format!("floor({x})"),
        Some(Rounding::Ceil) => format!("ceil({x})"),
        Some(Rounding::Trunc) => format!("trunc({x})"),
    }
}

fn signed_range(bytes: u8) -> (f32, f32) {
    match bytes {
        1 => (-128.0, 127.0),
        2 => (-32768.0, 32767.0),
        _ => (-2147483648.0, 2147483520.0),
    }
}

fn unsigned_max(bytes: u8) -> f32 {
    match bytes {
        1 => 255.0,
        2 => 65535.0,
        _ => 4294967040.0,
    }
}

/// A translated vertex stage rewritten as a compute shader, for checking what it computes:
/// invocation `i` runs vertex `i % 4` of instance `i / 4` with inputs from `dbg_in` (one vec4
/// per input location per instance, in `interface.inputs` order) and writes the position then
/// each output varying to `dbg_out`. Bound at `@group(2)`.
pub fn vertex_as_compute(t: &Translation) -> String {
    let source = &t.source;
    let cut = source.find("@vertex\nfn main").unwrap_or(source.len());
    let mut m = source[..cut].to_string();
    let inputs: Vec<u32> = t.interface.inputs.iter().copied().collect();
    let outputs: Vec<u32> = t.interface.outputs.iter().copied().collect();
    let _ = writeln!(
        m,
        "@group(2) @binding(0) var<storage, read> dbg_in: array<vec4<f32>>;\n@group(2) @binding(1) var<storage, read_write> dbg_out: array<vec4<f32>>;\n@compute @workgroup_size(1)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let i = gid.x;\n    vertex_id = i % 4u;\n    instance_id = i / 4u;"
    );
    for (k, loc) in inputs.iter().enumerate() {
        let _ = writeln!(
            m,
            "    vin.a{loc} = dbg_in[instance_id * {}u + {k}u];",
            inputs.len()
        );
    }
    let _ = writeln!(
        m,
        "    let o = run();\n    let base = i * {}u;\n    dbg_out[base] = o.pos;",
        outputs.len() + 2
    );
    let _ = writeln!(
        m,
        "    dbg_out[base + {}u] = vec4<f32>(f32(dbg_pc), f32(dbg_steps), f32(sp_dbg()), 0.0);",
        outputs.len() + 1
    );
    for (k, loc) in outputs.iter().enumerate() {
        let _ = writeln!(m, "    dbg_out[base + {}u] = o.v{loc};", k + 1);
    }
    m.push_str("}\n");
    m
}
