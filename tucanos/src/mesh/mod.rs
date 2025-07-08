mod autotag;
mod geom_elems;
mod gradient_l2proj;
mod gradient_ls;
#[allow(dead_code)]
#[allow(clippy::enum_variant_names)]
mod simplex_mesh;
pub mod test_meshes;
mod tmesh;
mod topo_elems;
mod topology;
mod twovec;
mod vector;

pub use geom_elems::{AsSliceF64, GEdge, GElem, GTetrahedron, GTriangle};
pub use simplex_mesh::{HasTmeshImpl, Point, SimplexMesh, SubSimplexMesh};
pub use topo_elems::{Edge, Elem, Tetrahedron, Triangle, Vertex, get_face_to_elem};
pub use topology::Topology;
