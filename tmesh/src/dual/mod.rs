//! Dual of simplex meshes

mod dual_mesh;
mod dual_mesh_2d;
mod dual_mesh_3d;
mod poly_mesh;

pub(crate) use dual_mesh::{circumcenter_bcoords, DualCellCenter};
pub use dual_mesh::{DualMesh, DualType};
pub use dual_mesh_2d::DualMesh2d;
pub use dual_mesh_3d::DualMesh3d;
#[allow(unused_imports)]
pub(crate) use poly_mesh::{merge_polygons, merge_polylines};
pub use poly_mesh::{PolyMesh, PolyMeshType, SimplePolyMesh};
