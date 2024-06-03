mod autotag;
mod geom_elems;
mod gradient_l2proj;
mod gradient_ls;
mod graph;
mod interpolate;
mod ordering;
mod partition;
#[allow(dead_code)]
#[allow(clippy::enum_variant_names)]
mod simplex_mesh;
mod split;
mod to_simplices;
mod topo_elems;
mod topology;
mod twovec;
mod vector;

pub mod io;
pub mod test_meshes;

pub use geom_elems::{AsSliceF64, GEdge, GElem, GTetrahedron, GTriangle};
pub use graph::ConnectedComponents;
pub use partition::PartitionType;
pub use simplex_mesh::{Point, SimplexMesh, SubSimplexMesh};
pub use topo_elems::{Edge, Elem, Tetrahedron, Triangle, get_face_to_elem};
pub use topology::Topology;
