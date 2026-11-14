//! cargo run --example flag_offsets -- ef_x.eff : which raw bytes at res+0x7f0..0x7f3 agree with
//! the parsed emission flags over every emitter.
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let raws = vfx_runtime::ptcl::emitter_sets(&p.base_bytes);
    let mut agree = [[0usize; 4]; 4];
    let mut total = 0;
    for (set, parsed) in p.emitter_list.emitter_sets.iter().enumerate() {
        for (index, e) in parsed.emitters.iter().enumerate() {
            let fixed = vfx_runtime::static_block::fixed_emitter_data(&raws[set].emitters[index], &|_| None, None);
            let em = &e.data.emission;
            let flags = [em.is_one_time, em.is_world_gravity, em.is_emit_dist_enabled, em.is_world_oriented_velocity];
            for (o, row) in agree.iter_mut().enumerate() {
                for (k, cell) in row.iter_mut().enumerate() {
                    if (fixed[0x7f0 + o] != 0) == flags[k] { *cell += 1; }
                }
            }
            total += 1;
        }
    }
    println!("{total} emitters; rows res+0x7f0..0x7f3, columns one_time world_gravity emit_dist world_oriented");
    for (o, row) in agree.iter().enumerate() { println!("0x{:x} {:?}", 0x7f0 + o, row); }
}
