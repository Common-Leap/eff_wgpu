//! cargo run --example emitter_texture -- ef_x.eff ENTRY EMITTER : an emitter's sampler-0 texture and its channel ranges
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let set = eff.entry_names.iter().zip(&eff.entries)
        .find(|(n, _)| n.eq_ignore_ascii_case(&a[2])).map(|(_, e)| e.emitter_set_id as usize - 1).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let e = p.emitter_list.emitter_sets[set].emitters.iter().find(|e| e.data.name.as_deref() == Some(a[3].as_str())).unwrap();
    let s = e.data.sampler0.as_ref().unwrap();
    let ids: Vec<u64> = p.texture_info.as_ref().unwrap().descriptors.iter().map(|d| d.id).collect();
    let index = ids.iter().position(|id| *id == s.texture_id).unwrap();
    let d = &p.texture_info.as_ref().unwrap().descriptors[index];
    println!("texture {index} {:?}", d);
    let t = vfx_runtime::texture::decode(p, index).unwrap();
    println!("{}x{} layers {} srgb {}", t.width, t.height, t.layers, t.srgb);
    let px = &t.mips[0];
    let (mut mn, mut mx) = ([255u8; 4], [0u8; 4]);
    for c in px.chunks(4) { for k in 0..4 { mn[k] = mn[k].min(c[k]); mx[k] = mx[k].max(c[k]); } }
    println!("channel min {mn:?} max {mx:?}");
}
