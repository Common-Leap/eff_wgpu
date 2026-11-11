//! cargo run --example dump_shaders -- path/to/ef_x.eff [program index]
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("eff path");
    let only: Option<usize> = args.next().and_then(|a| a.parse().ok());
    let data = std::fs::read(&path).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let ptcl = eff.ptcl_file.expect("no ptcl");
    let info = ptcl.shader_info.expect("no shaders");
    let bin = info.binary_data.expect("no binary");
    let file = vfx_runtime::bnsh::ShaderFile::read(&bin).unwrap();
    println!("{} programs", file.programs.len());
    for (i, p) in file.programs.iter().enumerate() {
        if only.is_some_and(|o| o != i) {
            continue;
        }
        for (s, stage) in p.stages.iter().enumerate() {
            let Some(stage) = stage else { continue };
            println!("program {i} stage {s}: {} bytes", stage.code.len());
            let r = &stage.reflection;
            println!(
                "  in  {:?}",
                r.inputs
                    .iter()
                    .map(|b| format!("{}@{}", b.name, b.slot))
                    .collect::<Vec<_>>()
            );
            println!(
                "  out {:?}",
                r.outputs
                    .iter()
                    .map(|b| format!("{}@{}", b.name, b.slot))
                    .collect::<Vec<_>>()
            );
            println!(
                "  smp {:?}",
                r.samplers
                    .iter()
                    .map(|b| format!("{}@{}", b.name, b.slot))
                    .collect::<Vec<_>>()
            );
            println!(
                "  cb  {:?}",
                r.constant_buffers
                    .iter()
                    .map(|b| format!("{}@{}", b.name, b.slot))
                    .collect::<Vec<_>>()
            );
            if only.is_some() {
                for (k, w) in stage.code.chunks(4).take(24).enumerate() {
                    print!(
                        "{:08x}{}",
                        u32::from_le_bytes(w.try_into().unwrap()),
                        if k % 8 == 7 { "\n" } else { " " }
                    );
                }
            }
        }
    }
}
