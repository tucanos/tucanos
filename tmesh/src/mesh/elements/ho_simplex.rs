#[derive(Debug, Clone, Copy)]
pub enum HOType {
    Lagrange,
    Bezier,
}

pub const FAST_PROJ_TOLERANCE: f64 = 0.1;
pub const FAST_PROJ_MAX_ITERS: u64 = 1;
