use vfx_runtime::maxwell::decode::{decode_program, Op};
fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let (mut vs, mut fs, mut vb, mut fb, mut unk) = (0, 0, 0, 0, std::collections::BTreeMap::new());
    let mut handles = std::collections::BTreeMap::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let code = std::fs::read(&p).unwrap();
        if code.len() <= 0x50 {
            continue;
        }
        let insts = decode_program(&code[0x50..]);
        let frag = name.ends_with("_4.bin");
        let mut exit_seen = false;
        let mut branches = false;
        for i in &insts {
            match &i.op {
                Op::Exit(_) if i.guard.is_always() => exit_seen = true,
                Op::Bra { target, .. } if !(exit_seen && *target == i.addr) => branches = true,
                Op::Unknown(w) if !exit_seen => {
                    *unk.entry(format!("{:04x}", w >> 48)).or_insert(0) += 1
                }
                Op::Texs { handle, .. } | Op::Tmml { handle, .. } => {
                    *handles.entry(*handle).or_insert(0) += 1
                }
                _ => {}
            }
        }
        if frag {
            fs += 1;
            if branches {
                fb += 1;
            }
        } else {
            vs += 1;
            if branches {
                vb += 1;
            }
        }
    }
    println!("vertex {vs} (branching {vb}), fragment {fs} (branching {fb})");
    println!("unknown before exit: {unk:?}");
    println!("texture handles: {handles:?}");
}
