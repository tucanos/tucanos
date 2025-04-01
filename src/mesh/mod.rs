mod autotag;
mod geom_elems;
mod geom_quad_elems;
mod gradient_l2proj;
mod gradient_ls;
mod graph;
mod interpolate;
mod ordering;
mod partition;
mod quadratic_mesh;
#[allow(dead_code)]
#[allow(clippy::enum_variant_names)]
mod simplex_mesh;
mod split;
mod to_simplices;
mod topo_elems;
mod topo_elems_quadratic;
mod topology;
mod twovec;
mod vector;

pub mod io;
pub mod test_meshes;

pub use geom_elems::{AsSliceF64, GEdge, GElem, GTetrahedron, GTriangle};
pub use geom_quad_elems::{GQuadraticEdge, GQuadraticTriangle};
pub use partition::PartitionType;
pub use quadratic_mesh::QuadraticMesh;
pub use simplex_mesh::{Point, SimplexMesh, SubSimplexMesh};
pub use topo_elems::{get_face_to_elem, Edge, Elem, Tetrahedron, Triangle};
pub use topo_elems_quadratic::{QuadraticEdge, QuadraticElem, QuadraticTriangle};
pub use topology::Topology;
