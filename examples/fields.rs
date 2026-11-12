fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.unwrap();
    let sets = vfx_runtime::ptcl::emitter_sets(&p.base_bytes);
    let set: usize = a[2].parse().unwrap();
    for (j, e) in sets[set].emitters.iter().enumerate() {
        let d = &e.data;
        let u64at = |o: usize| u64::from_le_bytes(d[o..o + 8].try_into().unwrap());
        println!("8c8={:#x}", u64at(0x8c8));
        println!(
            "emitter {j}: 8a9={} 8d0={:#x} 878={:#x} 838 shape={} 9e7={:#x} shader_idx_parsed={}",
            d[0x8a9],
            u64at(0x8d0),
            u64at(0x878),
            d[0x838],
            d[0x9e7],
            p.emitter_list.emitter_sets[set].emitters[j]
                .data
                .shader_references
                .shader_index
        );
    }
}
