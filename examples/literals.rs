//! cargo run --example literals -- ef_x.eff PROGRAM : each stage's bank-1 literals as floats
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let bin = eff
        .ptcl_file
        .unwrap()
        .shader_info
        .unwrap()
        .binary_data
        .unwrap();
    let file = vfx_runtime::bnsh::ShaderFile::read(&bin).unwrap();
    let p = &file.programs[a[2].parse::<usize>().unwrap()];
    for (i, st) in p.stages.iter().enumerate() {
        if let Some(st) = st {
            let l = st.literals();
            let f: Vec<String> = l
                .chunks_exact(4)
                .take(16)
                .map(|c| format!("{}", f32::from_le_bytes(c.try_into().unwrap())))
                .collect();
            println!("stage {i} ({} bytes): {}", l.len(), f.join(" "));
        }
    }
}
