//! cargo run --example emitter_floats -- ef_x.eff ENTRY OFFSET COUNT [i] : floats (or i32 with i) of the entry's emitters
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let raw = vfx_runtime::ptcl::emitter_sets(&p.base_bytes);
    let at = usize::from_str_radix(a[3].trim_start_matches("0x"), 16).unwrap();
    let n: usize = a[4].parse().unwrap();
    let ints = a.get(5).is_some();
    let set = eff
        .entry_names
        .iter()
        .zip(&eff.entries)
        .find(|(name, _)| name.eq_ignore_ascii_case(&a[2]))
        .map(|(_, e)| e.emitter_set_id as usize - 1)
        .unwrap();
    for (j, (e, r)) in p.emitter_list.emitter_sets[set]
        .emitters
        .iter()
        .zip(&raw[set].emitters)
        .enumerate()
    {
        let fixed = vfx_runtime::static_block::fixed_emitter_data(r, &|_| None, None);
        let v: Vec<String> = (0..n)
            .map(|i| {
                let b: [u8; 4] = fixed[at + 4 * i..at + 4 * i + 4].try_into().unwrap();
                if ints {
                    i32::from_le_bytes(b).to_string()
                } else {
                    format!("{:.3}", f32::from_le_bytes(b))
                }
            })
            .collect();
        println!(
            "{j} {:<12} {}",
            e.data.name.clone().unwrap_or_default(),
            v.join(" ")
        );
    }
}
