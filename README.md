# eff_wgpu

Simulates and draws Super Smash Bros. Ultimate effect files (`.eff`) with
[wgpu](https://wgpu.rs), the way the game does.

- Particles follow the rules of the game's own effect code: emission, life,
  movement, fading, children.
- Each emitter is drawn with its own shader, translated from the game's
  compiled shader in the file to WGSL.
- Textures, meshes, blending and the camera block match the game.

[Visionary](https://github.com/Common-Leap/Visionary) uses it for its live
effect preview.

## Use

```toml
[dependencies]
eff_wgpu = { git = "https://github.com/Common-Leap/eff_wgpu" }
```

The library is the pieces the preview is made of. A frame is drawn in four
steps:

1. Read the file with `effect_library`, its shaders with `bnsh::ShaderFile`,
   and its emitter sets with `ptcl::emitter_sets`.
2. Make one `sim::EmitterSim` per emitter, and step it once per frame.
3. Make an `render::EffectRenderer` for your target format, give it a camera
   with `set_view`, and upload the textures with `render::upload_texture`.
4. Each frame, `prepare` an `EmitterDraw` per emitter, then `draw` it in your
   render pass.

[`examples/render_frame.rs`](examples/render_frame.rs) does all of this in one
file and writes a PNG:

```sh
cargo run --example render_frame -- ef_mario.eff 0 10 out.png
```

Effect files come from the game's data. Dump them with a tool like
ArcExplorer.

## Build

Needs Rust 1.87 or newer.

```sh
cargo build
cargo test
```

The tests that read the game's files are skipped unless `VFX_GAME_EFFECTS`
points at a dumped `effect` folder (and `VFX_SHADER_DUMP` at dumped shader
programs, for the shader test).

The other examples:

- `emitters`: lists a file's sets, emitters and their data.
- `texture_png`: writes a texture as the shaders sample it.
- `sweep`: prepares and draws every set of every file in a folder, and reports
  any that fail.

## License

MIT
