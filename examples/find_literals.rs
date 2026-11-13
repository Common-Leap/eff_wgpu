//! cargo run --example find_literals -- ef_x.eff PROGRAM HEX : where HEX sits relative to the program's stage code
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
    let pat: Vec<u8> = (0..a[3].len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&a[3][i..i + 2], 16).unwrap())
        .collect();
    let at_bin = bin.windows(pat.len()).position(|w| w == pat.as_slice());
    let file = vfx_runtime::bnsh::ShaderFile::read(&bin).unwrap();
    let st = file.programs[a[2].parse::<usize>().unwrap()].stages[0]
        .as_ref()
        .unwrap();
    let code_at = bin.windows(64).position(|w| w == &st.code[..64]);
    let ctl_at = bin.windows(64).position(|w| w == &st.control[..64]);
    println!("pattern in shader binary at {at_bin:x?}; vertex code at {code_at:x?} (len {:#x}); control at {ctl_at:x?} (len {:#x})", st.code.len(), st.control.len());
    if let (Some(p), Some(c)) = (at_bin, code_at) {
        println!("pattern - code = {:#x}", p as i64 - c as i64);
    }
    let words: Vec<String> = st
        .control
        .chunks_exact(4)
        .enumerate()
        .filter(|(_, w)| {
            let v = u32::from_le_bytes((*w).try_into().unwrap()) as usize;
            at_bin
                .zip(code_at)
                .is_some_and(|(p, c)| v == p - c || v == p)
        })
        .map(|(i, _)| format!("{:#x}", i * 4))
        .collect();
    println!("control words equal to that offset: {words:?}");
    for off in [0x6f0usize, 0x6f4, 0x6f8, 0x6fc, 0x700, 0x704] {
        println!(
            "ctl[{off:#x}] = {:#x}",
            u32::from_le_bytes(st.control[off..off + 4].try_into().unwrap())
        );
    }
}
