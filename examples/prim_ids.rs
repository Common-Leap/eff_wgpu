fn main() {
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).unwrap();
        let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
        let p = eff.ptcl_file.unwrap().base_bytes;
        let mut counts = std::collections::BTreeMap::new();
        fn walk(e: &vfx_runtime::ptcl::Emitter, c: &mut std::collections::BTreeMap<u64, usize>) {
            let v = u64::from_le_bytes(e.data[0x8c8..0x8d0].try_into().unwrap());
            *c.entry(v).or_insert(0) += 1;
            for k in &e.children {
                walk(k, c);
            }
        }
        for s in vfx_runtime::ptcl::emitter_sets(&p) {
            for e in &s.emitters {
                walk(e, &mut counts);
            }
        }
        println!("{path}: {:x?}", counts);
    }
}
