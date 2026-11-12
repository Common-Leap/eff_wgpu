//! cargo run --example prim_attrs -- ef_x.eff ID_HEX : a primitive's descriptor and its model's vertex attributes
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.unwrap();
    let info = p.primitive_info.unwrap();
    let want = u64::from_str_radix(&a[2], 16).unwrap();
    let res =
        effect_library::bfres::load::load_from_bytes(info.binary_data.as_ref().unwrap()).unwrap();
    for (d, (name, m)) in info.descriptors.iter().zip(res.models.iter()) {
        if d.id != want {
            continue;
        }
        println!("{d:x?}\nmodel {name}");
        for vb in &m.vertex_buffers {
            for (i, (an, at)) in vb.attributes.iter().enumerate() {
                println!(
                    "  [{i}] {an} buf {} off {} fmt {:#x}",
                    at.buffer_index, at.offset, at.format
                );
            }
        }
    }
}
