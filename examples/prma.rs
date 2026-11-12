fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.unwrap().base_bytes;
    for h in vfx_runtime::ptcl::top_level(&p) {
        if &h.magic != b"PRMA" {
            continue;
        }
        println!("{h:?}");
        let end = h.at + h.size as usize;
        for (i, w) in p[h.at..end].chunks(16).enumerate() {
            let ws: Vec<String> = w
                .chunks(4)
                .map(|x| {
                    let v = u32::from_le_bytes(x.try_into().unwrap_or([0; 4]));
                    format!("{v:08x}({:.3})", f32::from_bits(v))
                })
                .collect();
            println!("{:04x}: {}", i * 16, ws.join(" "));
        }
        for c in vfx_runtime::ptcl::chain(&p, h.first_child()) {
            println!("child {:?}", c);
        }
    }
}
