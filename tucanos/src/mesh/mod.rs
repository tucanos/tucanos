mod autotag;
mod gradient_l2proj;
mod gradient_ls;
#[allow(dead_code)]
#[allow(clippy::enum_variant_names)]
mod simplex_mesh;
pub mod test_meshes;
mod topology;
mod vector;

pub use simplex_mesh::{SimplexMesh, SubSimplexMesh};
pub use topology::Topology;
