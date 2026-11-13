//! Effect textures (the GRTF section, a BNTX pool) decoded the way the shader samples them:
//! every mip level as straight RGBA8, with the texture's channel swizzle (BRTI `comp_sel`)
//! applied, since that is part of what a texture means to the shader (a BC4 mask broadcasts
//! red into alpha, a BC5 one reads alpha from green).

use binrw::BinRead;
use effect_library::ptcl_file::PtclFile;

pub struct Texture {
    pub width: u32,
    pub height: u32,
    /// Array layers (1 for most; pattern-animation textures keep one frame per layer).
    pub layers: u32,
    /// Stored as sRGB: the GPU converts these to linear when sampling (BC1/2/3/7 Srgb).
    pub srgb: bool,
    /// Mip 0 first; each level holds every layer in turn, each width*height*4 bytes at that
    /// level's size.
    pub mips: Vec<Vec<u8>>,
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn u64_at(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

/// The BRTI block of texture `index` in a pool, and its channel swizzle (one byte per output
/// channel naming its source: 0 zero, 1 one, 2..5 red, green, blue, alpha).
fn comp_sel(pool: &[u8], index: usize) -> Option<[u8; 4]> {
    let table = u64_at(pool, 0x28)? as usize;
    let brti = u64_at(pool, table + index * 8)? as usize;
    (pool.get(brti..brti + 4)? == b"BRTI").then_some(())?;
    Some(u32_at(pool, brti + 0x58)?.to_le_bytes())
}

/// Texture `index` of the file's pool, or `None` when it is missing or in a format the decoder
/// does not know.
pub fn decode(ptcl: &PtclFile, index: usize) -> Option<Texture> {
    let info = ptcl.texture_info.as_ref()?;
    let pool = info.binary_data.as_ref()?;
    let name = &info.descriptors.get(index)?.name;
    let single = effect_library::bntx::build_single_texture_bntx_public(pool, index, name).ok()?;
    let bntx = bntx::Bntx::read_le(&mut std::io::Cursor::new(single)).ok()?;
    let stored = bntx.to_surface().ok()?;
    let srgb = format!("{:?}", stored.image_format).contains("Srgb");
    let surface = stored.decode_rgba8().ok()?;
    let sel = comp_sel(pool, index).unwrap_or([2, 3, 4, 5]);
    let (width, height, layers) = (surface.width, surface.height, surface.layers.max(1));
    let mut mips = Vec::new();
    'levels: for level in 0..surface.mipmaps {
        let (w, h) = ((width >> level).max(1), (height >> level).max(1));
        let mut out = Vec::with_capacity((w * h * 4 * layers) as usize);
        for layer in 0..layers {
            let Some(data) = surface.get(layer, 0, level) else {
                break 'levels;
            };
            out.extend_from_slice(&data[..((w * h * 4) as usize).min(data.len())]);
        }
        for px in out.chunks_exact_mut(4) {
            let src = [px[0], px[1], px[2], px[3]];
            for (c, s) in sel.iter().enumerate() {
                px[c] = match s {
                    0 => 0,
                    1 => 255,
                    2..=5 => src[(s - 2) as usize],
                    _ => src[c],
                };
            }
        }
        mips.push(out);
    }
    (!mips.is_empty()).then_some(Texture {
        width,
        height,
        layers,
        srgb,
        mips,
    })
}
