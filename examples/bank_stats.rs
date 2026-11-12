use vfx_runtime::maxwell::{decode::decode_program, wgsl};
fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let mut banks = std::collections::BTreeMap::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        let n = p.file_name().unwrap().to_string_lossy().to_string();
        let kind = if n.ends_with("_4.bin") {
            wgsl::StageKind::Fragment
        } else if n.ends_with("_0.bin") {
            wgsl::StageKind::Vertex
        } else {
            continue;
        };
        let code = std::fs::read(&p).unwrap();
        if let Ok(t) = wgsl::translate(kind, &decode_program(&code[0x50..])) {
            for b in t.interface.banks {
                *banks.entry(b).or_insert(0) += 1;
            }
        }
    }
    println!("{banks:?}");
}
