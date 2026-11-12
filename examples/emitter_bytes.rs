//! cargo run --example emitter_bytes -- ef_x.eff NAME OFFSET LEN : raw emitter bytes of sets matching NAME
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let raw = vfx_runtime::ptcl::emitter_sets(&p.base_bytes);
    let at = usize::from_str_radix(a[3].trim_start_matches("0x"), 16).unwrap();
    let len: usize = a[4].parse().unwrap();
    let wanted: Vec<usize> = eff
        .entry_names
        .iter()
        .zip(&eff.entries)
        .filter(|(n, _)| n.to_lowercase().contains(&a[2].to_lowercase()))
        .map(|(_, e)| e.emitter_set_id as usize)
        .collect();
    for (i, s) in p.emitter_list.emitter_sets.iter().enumerate() {
        if !wanted.contains(&i) && !s.name.to_lowercase().contains(&a[2].to_lowercase()) {
            continue;
        }
        println!("{i}: {}", s.name);
        for (j, (e, r)) in s.emitters.iter().zip(&raw[i].emitters).enumerate() {
            let bytes: Vec<String> = r.data[at..at + len]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            let attrs: Vec<String> = r
                .attributes
                .iter()
                .map(|(m, d)| format!("{}:{}", String::from_utf8_lossy(m), d.len()))
                .collect();
            println!("  {j} {:?} {} {attrs:?}", e.data.name, bytes.join(" "));
        }
    }
}
