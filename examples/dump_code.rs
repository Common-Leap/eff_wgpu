//! Write every program's stage code as raw files: OUT/<file>_<program>_<stage>.bin
fn main() {
    let mut args = std::env::args().skip(1);
    let out = std::path::PathBuf::from(args.next().unwrap());
    std::fs::create_dir_all(&out).unwrap();
    for path in args {
        let data = std::fs::read(&path).unwrap();
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
        let file = vfx_runtime::bnsh::ShaderFile::read(&bin).unwrap();
        let stem = std::path::Path::new(&path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        for (i, p) in file.programs.iter().enumerate() {
            for (s, st) in p.stages.iter().enumerate() {
                if let Some(st) = st {
                    std::fs::write(out.join(format!("{stem}_{i}_{s}.bin")), &st.code).unwrap();
                }
            }
        }
    }
}
