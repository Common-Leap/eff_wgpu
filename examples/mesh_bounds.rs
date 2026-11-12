//! cargo run --example mesh_bounds -- ef_x.eff [ID_HEX] : each mesh's vertex count, bounds, and attribute ranges
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let eff = effect_library::namco_file::NamcoEffectFile::load(&data).unwrap();
    let p = eff.ptcl_file.unwrap();
    let want = a
        .get(2)
        .map(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap());
    for (id, m) in vfx_runtime::mesh::meshes(&p) {
        if want.is_some_and(|w| w != id) {
            continue;
        }
        let mut lo = [[f32::MAX; 4]; 4];
        let mut hi = [[f32::MIN; 4]; 4];
        for v in &m.vertices {
            for s in 0..4 {
                for c in 0..4 {
                    lo[s][c] = lo[s][c].min(v[s][c]);
                    hi[s][c] = hi[s][c].max(v[s][c]);
                }
            }
        }
        println!(
            "{id:#x}: {} vertices, {} indices",
            m.vertices.len(),
            m.indices.len()
        );
        if std::env::var_os("RADIAL").is_some() {
            let mut rows: Vec<(f32, f32, f32, f32)> = m
                .vertices
                .iter()
                .map(|v| {
                    (
                        (v[0][0].powi(2) + v[0][1].powi(2)).sqrt(),
                        v[3][3],
                        v[1][0],
                        v[1][1],
                    )
                })
                .collect();
            rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for (r, a, u, v) in rows.iter().step_by(9) {
                println!("  r {r:.2} alpha {a:.2} uv {u:.3} {v:.3}");
            }
        }
        for (s, name) in ["pos", "uv", "normal", "colour"].iter().enumerate() {
            println!("  {name} {:?} .. {:?}", lo[s], hi[s]);
        }
    }
}
