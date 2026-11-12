//! cargo run --example entries -- ef_x.eff [FILTER] : Namco entry headers
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let sets = &eff.ptcl_file.as_ref().unwrap().emitter_list.emitter_sets;
    for (n, e) in eff.entry_names.iter().zip(&eff.entries) {
        if a.len() > 2 && !n.to_lowercase().contains(&a[2].to_lowercase()) {
            continue;
        }
        let set = (e.emitter_set_id as usize)
            .checked_sub(1)
            .and_then(|i| sets.get(i))
            .map(|s| s.name.clone())
            .unwrap_or_default();
        println!(
            "{n}: kind {} unknown {:#x} set {} ({set}) [set id is 1-based] model {} variants {}+{}",
            e.kind,
            e.unknown,
            e.emitter_set_id,
            e.external_model_idx,
            e.variant_start_idx,
            e.variant_count
        );
    }
}
