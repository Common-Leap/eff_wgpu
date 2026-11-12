//! cargo run --example static_hex -- ef_x.eff ENTRY : each emitter's static block as hex, one per line
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
    for e in &vfx_runtime::ptcl::emitter_sets(&p.base_bytes)[set].emitters {
        let fixed = vfx_runtime::static_block::fixed_emitter_data(e, &|_| None, None);
        let block = vfx_runtime::static_block::static_block(&fixed);
        println!(
            "{}",
            block.iter().map(|b| format!("{b:02x}")).collect::<String>()
        );
    }
}
