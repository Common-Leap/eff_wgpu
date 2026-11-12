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
    let f = vfx_runtime::bnsh::ShaderFile::read(&bin).unwrap();
    let st = f.programs[a[2].parse::<usize>().unwrap()].stages[a[3].parse::<usize>().unwrap()]
        .as_ref()
        .unwrap();
    println!("control {} bytes", st.control.len());
    for (i, w) in st.control.chunks(16).enumerate() {
        let ws: Vec<String> = w
            .chunks(4)
            .map(|x| format!("{:08x}", u32::from_le_bytes(x.try_into().unwrap_or([0; 4]))))
            .collect();
        if ws.iter().any(|x| x != "00000000") {
            println!("{:04x}: {}", i * 16, ws.join(" "));
        }
    }
}
