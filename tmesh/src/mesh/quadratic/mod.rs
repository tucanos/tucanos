pub mod nelder_mead;
pub mod optimize;
pub mod to_quadratic;

pub use nelder_mead::{NelderMeadParams, minimize_nelder_mead};
pub use optimize::optimize_quadratic_mesh;
pub use to_quadratic::{
    to_quadratic_edge_mesh, to_quadratic_tetrahedron_mesh, to_quadratic_triangle_mesh,
};
