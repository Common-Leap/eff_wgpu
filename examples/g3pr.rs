fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.unwrap();
    let info = p.primitive_info.unwrap();
    println!("{} descriptors", info.descriptors.len());
    for d in info.descriptors.iter().take(3) {
        println!("{:x?}", d);
    }
    let res =
        effect_library::bfres::load::load_from_bytes(info.binary_data.as_ref().unwrap()).unwrap();
    println!("{} models", res.models.len());
    let mut formats = std::collections::BTreeMap::new();
    for (i, (name, m)) in res.models.iter().enumerate() {
        for vb in &m.vertex_buffers {
            for (an, at) in &vb.attributes {
                *formats.entry((an.clone(), at.format)).or_insert(0) += 1;
                if i < 2 {
                    println!(
                        "model {i} {name}: attr {an} buf {} off {} fmt {:#x} stride {:?}",
                        at.buffer_index, at.offset, at.format, vb.buffer_strides
                    );
                }
            }
        }
        if i < 2 {
            for (sn, s) in &m.shapes {
                for mesh in &s.meshes {
                    println!(
                        "  shape {sn} prim {} idxfmt {} count {}",
                        mesh.primitive_type, mesh.index_format, mesh.index_count
                    );
                }
            }
        }
    }
    println!("{formats:?}");
}
