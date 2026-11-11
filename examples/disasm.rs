//! cargo run --example disasm -- code.bin  (raw stage code as written by dump_code)
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let code = std::fs::read(path).unwrap();
    let body = if code.len() > 0x50 {
        &code[0x50..]
    } else {
        &code[..]
    };
    for inst in vfx_runtime::maxwell::decode::decode_program(body) {
        println!("{:05x}: {inst}", inst.addr);
    }
}
