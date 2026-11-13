//! cargo run --example emitter_names -- ef_x.eff [ENTRY_SUBSTRING] : entry -> emitter names
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    for (n, e) in eff.entry_names.iter().zip(&eff.entries) {
        if a.len() >= 3 && !n.to_lowercase().contains(&a[2].to_lowercase()) {
            continue;
        }
        let Some(set) = (e.emitter_set_id as usize).checked_sub(1) else { continue };
        let Some(s) = p.emitter_list.emitter_sets.get(set) else { continue };
        let names: Vec<_> = s.emitters.iter().map(|e| e.data.name.clone().unwrap_or_default()).collect();
        println!("{n}: {}", names.join(" "));
    }
}
