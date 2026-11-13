//! The camera block every effect shader reads (uniform slot 5), laid out the way Smash fills
//! nn::vfx::ViewParam: checked against the block a captured game frame bound (13.0.4, Ryujinx +
//! RenderDoc); the layout first came from SMO's sead::ptcl::PtclSystem::beginRender.

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
/// `extra` is row 19's second value (1 in the captured game frame).
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
    rows.push([1.0, extra, 0.01, 0.0]);
    rows.iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The camera block from a captured training-stage frame, rebuilt from its parts.
    #[test]
    fn rebuilds_a_captured_game_camera_block() {
        let view = [
            [1.0, 0.0, 0.0, 17.330_957],
            [0.0, 1.0, 0.0, -8.396_974],
            [0.0, 0.0, 1.0, -446.101_65],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let proj = [
            [0.028_879_892, 0.0, 0.0, 0.0],
            [0.0, 0.051_342_033, 0.0, 0.0],
            [0.0, 0.0, -0.000_447_828_03, -0.095_432_89],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let block = view_block(
            &view,
            &proj,
            [-17.33096, 8.39697, 446.10165],
            213.10165,
            2446.10156,
            1.0,
        );
        let f: Vec<f32> = block
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let row = |r: usize| [f[r * 4], f[r * 4 + 1], f[r * 4 + 2], f[r * 4 + 3]];
        let close = |a: [f32; 4], b: [f32; 4]| {
            a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() < 1e-5 * y.abs().max(1.0))
        };
        // Rows 8-11 as captured: projection times view.
        assert!(
            close(row(8), [0.028_879_892, 0.0, 0.0, 0.500_516_2]),
            "{:?}",
            row(8)
        );
        assert!(
            close(row(9), [0.0, 0.051_342_033, 0.0, -0.431_117_7]),
            "{:?}",
            row(9)
        );
        assert!(
            close(row(10), [0.0, 0.0, -0.000_447_828_03, 0.104_343_94]),
            "{:?}",
            row(10)
        );
        assert_eq!(row(12), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(row(16), [0.0, 0.0, 1.0, 0.0]);
        assert!(
            close(row(18), [213.10165, 2446.10156, 521268.28, 2233.0]),
            "{:?}",
            row(18)
        );
        assert_eq!(row(19), [1.0, 1.0, 0.01, 0.0]);
    }
}
