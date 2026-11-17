//! Every program dumped from real effect files translates to WGSL that naga accepts.
//! Point VFX_SHADER_DUMP at a folder written by `examples/dump_code` (skipped when unset).
use eff_wgpu::maxwell::{decode::decode_program, wgsl};

#[test]
fn every_dumped_program_translates_and_validates() {
    let Some(dir) = std::env::var_os("VFX_SHADER_DUMP") else {
        eprintln!("VFX_SHADER_DUMP not set; skipped");
        return;
    };
    let mut failures = Vec::new();
    let mut count = 0;
    let mut nonuniform = 0;
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    let limit: usize = std::env::var("VFX_SHADER_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    for path in paths.into_iter().take(limit) {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let kind = if name.ends_with("_4.bin") {
            wgsl::StageKind::Fragment
        } else if name.ends_with("_0.bin") {
            wgsl::StageKind::Vertex
        } else {
            continue;
        };
        let code = std::fs::read(&path).unwrap();
        let insts = decode_program(&code[0x50..]);
        count += 1;
        let t = match wgsl::translate(kind, &insts) {
            Ok(t) => t,
            Err(e) => {
                failures.push(format!("{name}: translate: {e}"));
                continue;
            }
        };
        if !t.uniform_control_flow {
            nonuniform += 1;
        }
        let module = match naga::front::wgsl::parse_str(&t.source) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("{name}: parse: {}", e.emit_to_string(&t.source)));
                continue;
            }
        };
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        if let Err(e) = v.validate(&module) {
            failures.push(format!("{name}: validate: {}", e.emit_to_string(&t.source)));
        }
    }
    eprintln!(
        "{count} programs, {} failed, {nonuniform} pixel shaders branch",
        failures.len()
    );
    for f in failures.iter().take(5) {
        eprintln!("{f}\n---");
    }
    assert!(failures.is_empty());
}
