fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.unwrap();
    let (s, j): (usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let raw = &vfx_runtime::ptcl::emitter_sets(&p.base_bytes)[s].emitters[j].data;
    let e = &p.emitter_list.emitter_sets[s].emitters[j].data;
    let find = |name: &str, v: f32| {
        let hits: Vec<String> = (0..raw.len() - 4)
            .step_by(4)
            .filter(|&o| f32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) == v)
            .map(|o| format!("{o:#x}"))
            .collect();
        println!("{name} = {v}: {hits:?}");
    };
    let sc = &e.particle_scale;
    let v = &e.particle_velocity;
    find("scale_x", sc.scale_x);
    find("scale_y", sc.scale_y);
    find("scale_random_x", sc.scale_random_x);
    find("all_direction", v.all_direction);
    find("designated_dir_scale", v.designated_dir_scale);
    find("vel_random", v.vel_random);
    find("air_res", e.emitter_static.air_res);
    find("gravity_scale", e.emitter_static.gravity_scale);
    println!(
        "life {} life_random {}",
        e.particle_data.life, e.particle_data.life_random
    );
    println!("{:?}", sc);
}
