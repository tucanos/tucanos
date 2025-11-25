use crate::{
    Idx,
    mesh::{PyBoundaryMesh2d, PyBoundaryMesh3d, PyMesh2d, PyMesh3d},
    to_numpy_2d,
};
use numpy::PyArray2;
use pyo3::{Bound, PyResult, Python, exceptions::PyRuntimeError, pyclass, pymethods};
use tmesh::mesh::{Edge, GenericMesh, Mesh, Triangle};
use tucanos::{
    geometry::{Geometry, MeshedGeometry, orient_geometry},
    mesh::MeshTopology,
};

macro_rules! create_geometry {
    ($name: ident, $dim: expr, $etype: ident, $mesh: ident, $geom: ident) => {
        #[doc = concat!("Piecewise linear geometry consisting of ", stringify!($etype), " in ", stringify!($dim), "D")]
        #[pyclass]
        // #[derive(Clone)]
        pub struct $name {
            pub geom: MeshedGeometry<$dim, $etype<Idx>, GenericMesh<$dim, $etype<Idx>>>,
        }
        #[pymethods]
        impl $name {
            /// Create a new geometry
            #[new]
            #[must_use]
            #[pyo3(signature = (mesh, geom=None))]
            pub fn new(mesh: &$mesh, geom: Option<&$geom>) -> Self {
                let mut gmesh = if let Some(geom) = geom {
                    geom.0.clone()
                } else {
                    mesh.0.boundary::<GenericMesh<$dim, $etype::<Idx>>>().0
                };
                orient_geometry(&mesh.0, &mut gmesh);
                let topo= MeshTopology::new(&mesh.0);
                let geom = MeshedGeometry::new(&mesh.0, &topo, gmesh).unwrap();

                Self { geom }
            }

            /// Compute the max distance between the face centers and the geometry normals
            #[must_use]
            pub fn max_distance(&self, mesh: &$mesh) -> f64 {
                self.geom.max_distance(&mesh.0)
            }

            /// Compute the max angle between the face normals and the geometry normals
            #[must_use]
            pub fn max_normal_angle(&self, mesh: &$mesh) -> f64 {
                self.geom.max_normal_angle(&mesh.0)
            }

            /// Export the curvature to a vtk file
            pub fn write_curvature_vtk(&self, fname: &str) -> PyResult<()> {
                self.geom
                    .write_curvature(fname)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }

            /// Project vertices
            pub fn project<'py>(
                &self,
                py: Python<'py>,
                mesh: &$mesh,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                let topo = MeshTopology::new(&mesh.0);
                let mut coords = Vec::with_capacity(mesh.0.n_verts() * $dim);

                for (mut pt, tag) in mesh.0.verts().zip(topo.vtags().iter()) {
                    if tag.0 < $dim {
                        self.geom.project(&mut pt, tag);
                    }
                    coords.extend(pt.iter().copied());
                }

                Ok(to_numpy_2d(py, coords, $dim))
            }
        }
    };
}

create_geometry!(LinearGeometry3d, 3, Triangle, PyMesh3d, PyBoundaryMesh3d);
create_geometry!(LinearGeometry2d, 2, Edge, PyMesh2d, PyBoundaryMesh2d);
