//! The camera block every effect shader reads (uniform slot 5), laid out the way the game's
//! renderer fills nn::vfx::ViewParam (seen in sead::ptcl::PtclSystem::beginRender).

/// Row-major 4x4 matrices (`m[row][col]`).
pub type Mat4 = [[f32; 4]; 4];

pub const VIEW_SIZE: usize = 0x140;

fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = (0..4).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    out
}

/// `view` maps world to camera space (last row 0,0,0,1); `proj` is the projection.
/// `extra` is the value the game passes as its third argument (row 19.y).
pub fn view_block(
    view: &Mat4,
    proj: &Mat4,
    eye: [f32; 3],
    near: f32,
    far: f32,
    extra: f32,
) -> Vec<u8> {
    let view_proj = mul(proj, view);
    let mut billboard = [[0.0; 4]; 4];
    for r in 0..3 {
        for c in 0..3 {
            billboard[r][c] = view[c][r];
        }
    }
    billboard[3][3] = 1.0;
    let mut rows: Vec<[f32; 4]> = Vec::with_capacity(20);
    rows.extend_from_slice(view);
    rows.extend_from_slice(proj);
    rows.extend_from_slice(&view_proj);
    rows.extend_from_slice(&billboard);
    rows.push([view[2][0], view[2][1], view[2][2], 0.0]);
    rows.push([eye[0], eye[1], eye[2], 0.0]);
    rows.push([near, far, near * far, far - near]);
    rows.push([f32::from_bits(0x3fc8f5c3), extra, 0.0, 0.0]);
    rows.iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}
