//! cargo run --example sampler_ids -- ef_x.eff ENTRY : each emitter's sampler texture ids and names
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
    let names: Vec<(u64, String)> = p
        .texture_info
        .as_ref()
        .unwrap()
        .descriptors
        .iter()
        .map(|d| (d.id, d.name.clone()))
        .collect();
    for e in &p.emitter_list.emitter_sets[set].emitters {
        let ids: Vec<String> = [&e.data.sampler0, &e.data.sampler1, &e.data.sampler2]
            .iter()
            .map(|s| match s {
                Some(s) => format!(
                    "{:#x}={}",
                    s.texture_id,
                    names
                        .iter()
                        .find(|(i, _)| *i == s.texture_id)
                        .map(|(_, n)| n.as_str())
                        .unwrap_or("?")
                ),
                None => "none".into(),
            })
            .collect();
        println!("{:?} {ids:?}", e.data.name);
    }
}
