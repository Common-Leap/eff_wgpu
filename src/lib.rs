//! Smash Ultimate's effect runtime, rebuilt to match the game: effect files are read with
//! `effect_library`, particles are simulated by the rules of the game's own nn::vfx code, and
//! each emitter is drawn with its own shader, translated from the Maxwell code in the file.

pub mod bnsh;
pub mod maxwell;
