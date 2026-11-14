//! cargo run --example emitter_debug -- ef_x.eff ENTRY EMITTER : one emitter's parsed velocity,
//! emission and shape, and the raw life/emission fields the simulation reads.
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
    let raws = vfx_runtime::ptcl::emitter_sets(&p.base_bytes);
    for (index, e) in p.emitter_list.emitter_sets[set].emitters.iter().enumerate() {
        if e.data.name.as_deref() != Some(a[3].as_str()) {
            continue;
        }
        println!("velocity {:#?}", e.data.particle_velocity);
        println!("emission {:#?}", e.data.emission);
        println!("shape {:#?}", e.data.shape_info);
        let fixed =
            vfx_runtime::static_block::fixed_emitter_data(&raws[set].emitters[index], &|_| None, None);
        let i = |o: usize| i32::from_le_bytes(fixed[o..o + 4].try_into().unwrap());
        let f = |o: usize| f32::from_le_bytes(fixed[o..o + 4].try_into().unwrap());
        println!(
            "raw 0x7f4 {} 0x7f8 {} 0x7fc {} life {} life_rand {} infinite {} dist_flag {} unit {} min {} max {} margin {} interval {} interval_rand {} rate_rand {}",
            i(0x7f4), i(0x7f8), i(0x7fc), i(0x8b8), i(0x8bc), fixed[0x8a8], fixed[0x7f2], f(0x824), f(0x828), f(0x82c), f(0x830),
            i(0x808), i(0x80c), i(0x804)
        );
    }
}
