fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    for (i, s) in p.emitter_list.emitter_sets.iter().enumerate() {
        if a.len() < 3 || s.name.to_lowercase().contains(&a[2].to_lowercase()) {
            println!("{i}: {} ({} emitters)", s.name, s.emitters.len());
        }
    }
    for (i, n) in eff.entry_names.iter().enumerate() {
        if a.len() >= 3 && n.to_lowercase().contains(&a[2].to_lowercase()) {
            println!("entry {i}: {n}");
        }
    }
}
