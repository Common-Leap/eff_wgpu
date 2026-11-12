//! cargo run --example reflect -- ef_x.eff PROGRAM : each stage's reflected bindings
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
    let program = &file.programs[a[2].parse::<usize>().unwrap()];
    for (i, stage) in program.stages.iter().enumerate() {
        if let Some(stage) = stage {
            println!("stage {i}: {:#?}", stage.reflection);
        }
    }
}
