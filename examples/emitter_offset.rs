//! cargo run --example emitter_offset -- ef_x.eff ENTRY : file offset of each emitter's data
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let set = eff
        .entry_names
        .iter()
        .zip(&eff.entries)
        .find(|(n, _)| n.eq_ignore_ascii_case(&a[2]))
        .map(|(_, e)| e.emitter_set_id as usize - 1)
        .unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let base = data
        .windows(64)
        .position(|w| w == &p.base_bytes[..64])
        .unwrap();
    for e in &vfx_runtime::ptcl::emitter_sets(&p.base_bytes)[set].emitters {
        let at = p
            .base_bytes
            .windows(e.data.len().min(0xa00))
            .position(|w| w == &e.data[..e.data.len().min(0xa00)])
            .unwrap();
        println!("{:#x}", base + at);
    }
}
