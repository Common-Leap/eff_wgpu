//! cargo run --example emitters -- ef_x.eff : sets, emitters, data sizes, attribute sections
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let data = std::fs::read(&path).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let ptcl = eff.ptcl_file.unwrap().base_bytes;
    for h in eff_wgpu::ptcl::top_level(&ptcl) {
        println!(
            "top {} size {:#x}",
            String::from_utf8_lossy(&h.magic),
            h.size
        );
    }
    let sets = eff_wgpu::ptcl::emitter_sets(&ptcl);
    println!("{} sets", sets.len());
    for (i, s) in sets.iter().enumerate().take(3) {
        for (j, e) in s.emitters.iter().enumerate() {
            let attrs: Vec<_> = e
                .attributes
                .iter()
                .map(|(m, d)| format!("{}:{}", String::from_utf8_lossy(m), d.len()))
                .collect();
            println!(
                "set {i} emitter {j}: data {:#x} attrs {attrs:?} children {}",
                e.data.len(),
                e.children.len()
            );
        }
    }
}
