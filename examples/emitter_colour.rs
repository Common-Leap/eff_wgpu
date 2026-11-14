//! cargo run --example emitter_colour -- ef_x.eff ENTRY EMITTER : colour and alpha settings of one emitter
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let set = eff.entry_names.iter().zip(&eff.entries)
        .find(|(n, _)| n.eq_ignore_ascii_case(&a[2])).map(|(_, e)| e.emitter_set_id as usize - 1).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let e = p.emitter_list.emitter_sets[set].emitters.iter().find(|e| e.data.name.as_deref() == Some(a[3].as_str())).unwrap();
    println!("{:#?}", e.data.particle_color);
    println!("{:#?}", e.data.emitter_info.color0_r);
}
