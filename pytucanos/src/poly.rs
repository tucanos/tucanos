//! Python bindings for general poly meshes
use numpy::{
    PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{
    Bound, PyResult, Python,
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods,
};
use tmesh::{
    Tag, Vertex,
    dual::{PolyMesh, PolyMeshType, SimplePolyMesh},
};

/// Type of poly cells (mapping of `tmesh::PolyMeshType`)
#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq, Eq)]
pub enum PyPolyMeshType {
    /// Polylines
    Polylines,
    /// Polygons
    Polygons,
    /// Polyhedra
    Polyhedra,
}

macro_rules! create_poly_mesh {
    ($pyname: ident, $dim: expr) => {
        #[doc = concat!("Python binding for ", stringify!($name))]
        #[pyclass]
        pub struct $pyname(pub(crate) SimplePolyMesh<$dim>);

        #[pymethods]
        impl $pyname {
            /// Create a new mesh from coordinates, connectivities and tags
            #[new]
            #[allow(clippy::too_many_arguments)]
            fn new(
                poly_type: PyPolyMeshType,
                coords: PyReadonlyArray2<f64>,
                face_to_node_ptr: PyReadonlyArray1<usize>,
                face_to_node: PyReadonlyArray1<usize>,
                ftags: PyReadonlyArray1<Tag>,
                elem_to_face_ptr: PyReadonlyArray1<usize>,
                elem_to_face: PyReadonlyArray1<usize>,
                elem_to_face_orient: PyReadonlyArray1<bool>,
                etags: PyReadonlyArray1<Tag>,
            ) -> PyResult<Self> {
                if coords.shape()[1] != $dim {
                    return Err(PyValueError::new_err("Invalid dimension 1 for coords"));
                }

                let poly_type = match poly_type {
                    PyPolyMeshType::Polylines => PolyMeshType::Polylines,
                    PyPolyMeshType::Polygons => PolyMeshType::Polygons,
                    PyPolyMeshType::Polyhedra => PolyMeshType::Polyhedra,
                };

                let coords = coords.as_slice()?;
                let coords = coords
                    .chunks($dim)
                    .map(|p| {
                        let mut vx = Vertex::<$dim>::zeros();
                        vx.copy_from_slice(p);
                        vx
                    })
                    .collect();

                let elem_to_face = elem_to_face.as_slice()?;
                let elem_to_face_orient = elem_to_face_orient.as_slice()?;
                let elem_to_face = elem_to_face
                    .iter()
                    .cloned()
                    .zip(elem_to_face_orient.iter().cloned())
                    .collect::<Vec<_>>();

                Ok(Self(SimplePolyMesh::<$dim>::new(
                    poly_type,
                    coords,
                    face_to_node_ptr.to_vec().unwrap(),
                    face_to_node.to_vec().unwrap(),
                    ftags.to_vec().unwrap(),
                    elem_to_face_ptr.to_vec().unwrap(),
                    elem_to_face,
                    etags.to_vec().unwrap(),
                )))
            }

            /// Number of vertices
            fn n_verts(&self) -> usize {
                self.0.n_verts()
            }

            /// Get a copy of the vertices
            fn get_verts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
                let mut res = Vec::with_capacity($dim * self.0.n_verts());
                for v in self.0.verts() {
                    for &x in v.as_slice() {
                        res.push(x);
                    }
                }
                PyArray::from_vec(py, res).reshape([self.0.n_verts(), $dim])
            }

            /// Number of elements
            fn n_elems(&self) -> usize {
                self.0.n_elems()
            }

            /// Get a copy of the elements as 3 arrays:
            /// `(ptr, face_indices, faces_orientation)`
            fn get_elems<'py>(
                &self,
                py: Python<'py>,
            ) -> (
                Bound<'py, PyArray1<usize>>,
                Bound<'py, PyArray1<usize>>,
                Bound<'py, PyArray1<bool>>,
            ) {
                let n = self.0.n_elems();
                let m = self.0.elems().map(|x| x.len()).sum::<usize>();
                let mut ptr = Vec::with_capacity(n + 1);
                let mut e2f = Vec::with_capacity(m);
                let mut e2f_orient = Vec::with_capacity(m);
                ptr.push(0);

                for e in self.0.elems() {
                    for (f, o) in e {
                        e2f.push(f);
                        e2f_orient.push(o);
                    }
                    ptr.push(e2f.len());
                }

                (
                    PyArray::from_vec(py, ptr),
                    PyArray::from_vec(py, e2f),
                    PyArray::from_vec(py, e2f_orient),
                )
            }

            /// Get a copy of the element tags
            fn get_etags<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Tag>>> {
                Ok(PyArray::from_vec(py, self.0.etags().collect()))
            }

            /// Numver of faces
            fn n_faces(&self) -> usize {
                self.0.n_faces()
            }

            /// Get a copy of the elements as 3 arrays:
            /// `(ptr, vertex_indices)`
            fn get_faces<'py>(
                &self,
                py: Python<'py>,
            ) -> (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>) {
                let n = self.0.n_elems();
                let m = self.0.faces().map(|x| x.len()).sum::<usize>();
                let mut ptr = Vec::with_capacity(n + 1);
                let mut f2n = Vec::with_capacity(m);

                ptr.push(0);

                for f in self.0.faces() {
                    for i in f {
                        f2n.push(i);
                    }
                    ptr.push(f2n.len());
                }

                (PyArray::from_vec(py, ptr), PyArray::from_vec(py, f2n))
            }

            /// Get a copy of the face tags
            fn get_ftags<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Tag>>> {
                Ok(PyArray::from_vec(py, self.0.ftags().collect()))
            }

            /// Export the mesh to a `.vtu` file
            fn write_vtk(&self, file_name: &str) -> PyResult<()> {
                self.0
                    .write_vtk(file_name)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }
        }
    };
}

create_poly_mesh!(PyPolyMesh2d, 2);
create_poly_mesh!(PyPolyMesh3d, 3);
