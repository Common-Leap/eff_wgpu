//! Mesh primitives (the G3PR section, a BFRES) decoded into the four per-vertex inputs the
//! effect shaders read: sysPosAttr, sysTexCoordAttr, sysNormalAttr, sysVertexColor0Attr.

use effect_library::ptcl_file::PtclFile;

#[derive(Debug, Clone)]
pub struct Mesh {
    /// Per vertex: position (w 1), texcoord, normal, colour.
    pub vertices: Vec<[[f32; 4]; 4]>,
    pub indices: Vec<u32>,
}

fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let man = (bits & 0x3ff) as f32;
    sign * match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// One attribute value as four floats (missing components 0, w 1 for positions).
fn decode(format: u16, b: &[u8]) -> Option<[f32; 4]> {
    let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let f32_at = |i: usize| f32::from_le_bytes(b[i..i + 4].try_into().unwrap());
    Some(match format {
        0x1805 => [f32_at(0), f32_at(4), f32_at(8), 1.0],
        0x1705 => [f32_at(0), f32_at(4), 0.0, 0.0],
        0x1905 => [f32_at(0), f32_at(4), f32_at(8), f32_at(12)],
        0x1205 => [half(u16_at(0)), half(u16_at(2)), 0.0, 0.0],
        0x1505 => [
            half(u16_at(0)),
            half(u16_at(2)),
            half(u16_at(4)),
            half(u16_at(6)),
        ],
        0x1201 => [
            u16_at(0) as f32 / 65535.0,
            u16_at(2) as f32 / 65535.0,
            0.0,
            0.0,
        ],
        0x1202 => [
            (u16_at(0) as i16 as f32 / 32767.0).max(-1.0),
            (u16_at(2) as i16 as f32 / 32767.0).max(-1.0),
            0.0,
            0.0,
        ],
        0x0901 => [b[0] as f32 / 255.0, b[1] as f32 / 255.0, 0.0, 0.0],
        0x0b01 => [
            b[0] as f32 / 255.0,
            b[1] as f32 / 255.0,
            b[2] as f32 / 255.0,
            b[3] as f32 / 255.0,
        ],
        0x0b02 => std::array::from_fn(|i| (b[i] as i8 as f32 / 127.0).max(-1.0)),
        0x0e02 => {
            let w = u32::from_le_bytes(b[0..4].try_into().unwrap());
            let s10 = |v: u32| (((v << 22) as i32) >> 22) as f32 / 511.0;
            [
                s10(w & 0x3ff),
                s10((w >> 10) & 0x3ff),
                s10((w >> 20) & 0x3ff),
                ((w >> 30) as i32 - if w >> 31 != 0 { 4 } else { 0 }) as f32,
            ]
        }
        _ => return None,
    })
}

/// Every mesh in the file by primitive id.
pub fn meshes(ptcl: &PtclFile) -> std::collections::HashMap<u64, Mesh> {
    let mut out = std::collections::HashMap::new();
    let Some(info) = ptcl.primitive_info.as_ref() else {
        return out;
    };
    let Some(binary) = info.binary_data.as_ref() else {
        return out;
    };
    let Ok(res) = effect_library::bfres::load::load_from_bytes(binary) else {
        return out;
    };
    for (descriptor, (_, model)) in info.descriptors.iter().zip(res.models.iter()) {
        let Some(vb) = model.vertex_buffers.first() else {
            continue;
        };
        let Some(shape) = model.shapes.values().next() else {
            continue;
        };
        let Some(mesh) = shape.meshes.first() else {
            continue;
        };
        let attrs: Vec<_> = vb.attributes.values().collect();
        let pick = |index: i8| {
            (index >= 0)
                .then(|| attrs.get(index as usize).copied())
                .flatten()
        };
        let slots = [
            pick(descriptor.position_index),
            pick(descriptor.tex_coord0_index),
            pick(descriptor.normal_index),
            pick(descriptor.color_index),
        ];
        let mut vertices = Vec::with_capacity(vb.vertex_count as usize);
        for v in 0..vb.vertex_count as usize {
            let mut vertex = [
                [0.0, 0.0, 0.0, 1.0],
                [0.0; 4],
                [0.0, 0.0, 1.0, 0.0],
                [1.0; 4],
            ];
            for (slot, attr) in slots.iter().enumerate() {
                let Some(attr) = attr else { continue };
                let buffer = &vb.buffers[attr.buffer_index as usize];
                let stride = vb.buffer_strides[attr.buffer_index as usize] as usize;
                let at = v * stride + attr.offset as usize;
                if let Some(value) = buffer.get(at..).and_then(|b| decode(attr.format, b)) {
                    vertex[slot] = value;
                    if slot == 0 {
                        vertex[0][3] = 1.0;
                    }
                }
            }
            vertices.push(vertex);
        }
        let indices: Vec<u32> = match mesh.index_format {
            1 => mesh
                .index_data
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
                .collect(),
            2 => mesh
                .index_data
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            _ => mesh.index_data.iter().map(|b| *b as u32).collect(),
        };
        let first = mesh.first_vertex;
        let indices: Vec<u32> = indices
            .into_iter()
            .map(|i: u32| i + first)
            .take(mesh.index_count as usize)
            .collect();
        out.insert(descriptor.id, Mesh { vertices, indices });
    }
    out
}
