//! cargo run --release --example tex_dims -- EFFECT_ROOT : texture slot dimensions the translated shaders declare
use vfx_runtime::maxwell::{decode::decode_program, wgsl};
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
            for (i, kind) in [(0, wgsl::StageKind::Vertex), (4, wgsl::StageKind::Fragment)] {
                let Some(st) = prog.stages[i].as_ref() else {
                    continue;
                };
                let Ok(t) = wgsl::translate(
                    kind,
                    &decode_program(st.code.get(0x50..).unwrap_or_default()),
                ) else {
                    continue;
                };
                for (slot, dim) in t.interface.textures {
                    *counts.entry((i, slot, format!("{dim:?}"))).or_insert(0) += 1;
                }
            }
        }
    }
    for ((st, slot, d), c) in counts {
        println!("{c:6} stage {st} slot {slot} {d}");
    }
}
