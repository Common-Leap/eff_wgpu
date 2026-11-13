//! cargo run --release --example sampler_names -- EFFECT_ROOT : every pixel-stage sampler name and slot, counted
fn walk(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out)
        } else if p.extension().is_some_and(|x| x == "eff") {
            out.push(p)
        }
    }
}
fn main() {
    let mut list = Vec::new();
    walk(
        std::path::Path::new(&std::env::args().nth(1).unwrap()),
        &mut list,
    );
    let mut counts = std::collections::BTreeMap::new();
    for path in list {
        let Ok(data) = std::fs::read(&path) else {
            continue;
        };
        let Ok(eff) = effect_library::namco_file::NamcoEffectFile::load(&data) else {
            continue;
        };
        let Some(bin) = eff
            .ptcl_file
            .and_then(|p| p.shader_info)
            .and_then(|s| s.binary_data)
        else {
            continue;
        };
        let Ok(file) = vfx_runtime::bnsh::ShaderFile::read(&bin) else {
            continue;
        };
        for prog in &file.programs {
            for stage in prog.stages.iter().flatten() {
                for s in &stage.reflection.samplers {
                    *counts.entry((s.name.clone(), s.slot)).or_insert(0) += 1;
                }
            }
        }
    }
    for ((n, s), c) in counts {
        println!("{c:6} slot {s:2} {n}");
    }
}
