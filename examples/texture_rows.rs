//! cargo run --example texture_rows -- ef_x.eff NAME : alpha of the first and last rows and columns, as sampled
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
    let t = vfx_runtime::texture::decode(&p, index).unwrap();
    let (w, h) = (t.width as usize, t.height as usize);
    let m = &t.mips[0];
    let alpha = |x: usize, y: usize| m[(y * w + x) * 4 + 3];
    let row = |y: usize| {
        (0..w)
            .step_by(w / 16)
            .map(|x| alpha(x, y).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    let col = |x: usize| {
        (0..h)
            .step_by(h / 16)
            .map(|y| alpha(x, y).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    println!("{w}x{h} mips {}", t.mips.len());
    println!("row 0:    {}", row(0));
    println!("row last: {}", row(h - 1));
    println!("col 0:    {}", col(0));
    println!("col last: {}", col(w - 1));
    println!("middle row: {}", row(h / 2));
}
