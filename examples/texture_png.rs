//! cargo run --example texture_png -- ef_x.eff NAME OUT_PREFIX : the texture as sampled, plus each channel as grey
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.unwrap();
    let index = p
        .texture_info
        .as_ref()
        .unwrap()
        .descriptors
        .iter()
        .position(|d| d.name == a[2])
        .unwrap();
    let t = eff_wgpu::texture::decode(&p, index).unwrap();
    let m = &t.mips[0];
    for c in 0..4 {
        let grey: Vec<u8> = m.chunks_exact(4).map(|px| px[c]).collect();
        let file =
            std::fs::File::create(format!("{}_{}.png", a[3], ["r", "g", "b", "a"][c])).unwrap();
        let mut enc = png::Encoder::new(file, t.width, t.height);
        enc.set_color(png::ColorType::Grayscale);
        enc.write_header().unwrap().write_image_data(&grey).unwrap();
    }
}
