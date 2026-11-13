//! cargo run --example emission -- ef_x.eff ENTRY : each emitter's emission timing and life
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
    for e in &eff.ptcl_file.as_ref().unwrap().emitter_list.emitter_sets[set].emitters {
        println!("{:?}\n  {:?}", e.data.name, e.data.emission);
    }
}
