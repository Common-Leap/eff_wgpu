//! cargo run --example atlas_cells -- ef_x.eff NAME N : mean alpha of each cell of an N x N atlas (row 0 = first rows in memory)
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
    let n: usize = a[3].parse().unwrap();
    let t = vfx_runtime::texture::decode(&p, index).unwrap();
    let (w, h) = (t.width as usize, t.height as usize);
    println!(
        "{w}x{h} mips {:?}",
        t.mips.iter().map(|m| m.len()).collect::<Vec<_>>()
    );
    for cy in 0..n {
        let mut row = String::new();
        for cx in 0..n {
            let mut sum = 0u64;
            for y in cy * h / n..(cy + 1) * h / n {
                for x in cx * w / n..(cx + 1) * w / n {
                    sum += t.mips[0][(y * w + x) * 4 + 3] as u64;
                }
            }
            row += &format!("{:4} ", sum / ((w / n) * (h / n)) as u64);
        }
        println!("{row}");
    }
}
