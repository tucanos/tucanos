mod autotag;
#[allow(dead_code)]
#[allow(clippy::enum_variant_names)]
pub mod test_meshes;
mod topology;

pub use autotag::{autotag, autotag_bdy, transfer_tags};
pub use topology::{MeshTopology, Topology};
