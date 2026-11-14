//! cargo run --example blend_info -- ef_x.eff ENTRY : each emitter's blend, shader and draw flags
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let set = eff.entry_names.iter().zip(&eff.entries)
        .find(|(n, _)| n.eq_ignore_ascii_case(&a[2])).map(|(_, e)| e.emitter_set_id as usize - 1).unwrap();
    let p = eff.ptcl_file.as_ref().unwrap();
    let raws = vfx_runtime::ptcl::emitter_sets(&p.base_bytes);
    for (i, e) in p.emitter_list.emitter_sets[set].emitters.iter().enumerate() {
        let fixed = vfx_runtime::static_block::fixed_emitter_data(&raws[set].emitters[i], &|_| None, None);
        let rs = &e.data.render_state;
        println!("{} blend_enable {} blend_type {} res898 {} depth_test {} shader {} mesh {:x}",
            e.data.name.clone().unwrap_or_default(), rs.is_blend_enable, rs.blend_type, fixed[0x898],
            rs.is_depth_test, e.data.shader_references.shader_index,
            u64::from_le_bytes(fixed[0x8c8..0x8d0].try_into().unwrap()));
    }
}
