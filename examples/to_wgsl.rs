//! cargo run --example to_wgsl -- code.bin vertex|fragment
use vfx_runtime::maxwell::{decode::decode_program, wgsl};
fn main() {
    let mut a = std::env::args().skip(1);
    let code = std::fs::read(a.next().unwrap()).unwrap();
    let kind = if a.next().as_deref() == Some("fragment") {
        wgsl::StageKind::Fragment
    } else {
        wgsl::StageKind::Vertex
    };
    let t = wgsl::translate(kind, &decode_program(&code[0x50..])).unwrap();
    println!("{}", t.source);
}
