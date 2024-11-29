use crate::{
    mesh::{Mesh21, Mesh22, Mesh32, Mesh33},
    to_numpy_2d,
};
use numpy::PyArray2;
use pyo3::{exceptions::PyRuntimeError, pyclass, pymethods, Bound, PyResult, Python};
use tucanos::{
    geometry::{Geometry, LinearGeometry},
    mesh::{io::orient_stl, Edge, Triangle},
};
macro_rules! create_geometry {
    ($name: ident, $dim: expr, $etype: ident, $mesh: ident, $geom: ident) => {
        #[doc = concat!("Piecewise linear geometry consisting of ", stringify!($etype), " in ", stringify!($dim), "D")]
        #[pyclass]
        // #[derive(Clone)]
        pub struct $name {
            pub geom: LinearGeometry<$dim, $etype>,
        }
        #[pymethods]
        impl $name {
            /// Create a new geometry
            #[new]
            #[must_use]
            pub fn new(
                mesh: &$mesh,
                geom: Option<&$geom>,
            ) -> Self {

                let mut gmesh = if let Some(geom) = geom {
                    geom.mesh.clone()
                } else {
                    mesh.mesh.boundary().0
                };
                orient_stl(&mesh.mesh, &mut gmesh);
                let geom = LinearGeometry::new(&mesh.mesh, gmesh).unwrap();

                Self { geom }
            }

            /// Compute the max distance between the face centers and the geometry normals
            #[must_use]
            pub fn max_distance(&self, mesh: &$mesh) -> f64 {
                self.geom.max_distance(&mesh.mesh)
            }

            /// Compute the max angle between the face normals and the geometry normals
            #[must_use]
            pub fn max_normal_angle(&self, mesh: &$mesh) -> f64 {
                self.geom.max_normal_angle(&mesh.mesh)
            }

            /// Compute the curvature
            pub fn compute_curvature(&mut self)  {
               self.geom.compute_curvature()
            }

            /// Export the curvature to a vtk file
            pub fn write_curvature_vtk(&self, fname: &str) -> PyResult<()> {
               self.geom
                    .write_curvature(fname)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }

            /// Project vertices
            pub fn project<'py>(&self, py: Python<'py>, mesh: &$mesh) -> PyResult<Bound<'py, PyArray2<f64>>> {
                let vtags = mesh.mesh.get_vertex_tags().unwrap();
                let mut coords = Vec::with_capacity(mesh.mesh.n_verts() as usize * $dim);

                for (mut pt, tag) in mesh.mesh.verts().zip(vtags.iter()) {
                    if tag.0 < $dim {
                        self.geom.project(&mut pt, tag);
                    }
                    coords.extend(pt.iter().copied());
                }

                Ok(to_numpy_2d(
                    py,
                    coords,
                    $dim,
                ))
            }
        }
    }
}

create_geometry!(LinearGeometry3d, 3, Triangle, Mesh33, Mesh32);
create_geometry!(LinearGeometry2d, 2, Edge, Mesh22, Mesh21);
