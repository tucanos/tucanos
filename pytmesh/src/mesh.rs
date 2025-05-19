//! Python bindings for simplex meshes
use numpy::{
    PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{
    Bound, PyResult, Python,
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods,
    types::PyType,
};
use tmesh::{
    Tag, Vertex,
    boundary_mesh_2d::BoundaryMesh2d,
    boundary_mesh_3d::BoundaryMesh3d,
    mesh::Mesh,
    mesh_2d::{Mesh2d, nonuniform_rectangle_mesh},
    mesh_3d::{Mesh3d, nonuniform_box_mesh},
};

macro_rules! create_mesh {
    ($pyname: ident, $name: ident, $dim: expr, $cell_dim: expr, $face_dim: expr) => {
        #[doc = concat!("Python binding for ", stringify!($name))]
        #[pyclass]
        pub struct $pyname(pub(crate) $name);

        #[pymethods]
        impl $pyname {
            /// Create a new mesh from coordinates, connectivities and tags
            #[new]
            pub fn new(
                coords: PyReadonlyArray2<f64>,
                elems: PyReadonlyArray2<usize>,
                etags: PyReadonlyArray1<Tag>,
                faces: PyReadonlyArray2<usize>,
                ftags: PyReadonlyArray1<Tag>,
            ) -> PyResult<Self> {
                if coords.shape()[1] != $dim {
                    return Err(PyValueError::new_err("Invalid dimension 1 for coords"));
                }
                let n = elems.shape()[0];
                if elems.shape()[1] != $cell_dim {
                    return Err(PyValueError::new_err("Invalid dimension 1 for elems"));
                }
                if etags.shape()[0] != n {
                    return Err(PyValueError::new_err("Invalid dimension 0 for etags"));
                }
                let n = faces.shape()[0];

                if faces.shape()[1] != $face_dim {
                    return Err(PyValueError::new_err("Invalid dimension 1 for faces"));
                }
                if ftags.shape()[0] != n {
                    return Err(PyValueError::new_err("Invalid dimension 0 for ftags"));
                }

                let coords = coords.as_slice()?;
                let coords = coords
                    .chunks($dim)
                    .map(|p| {
                        let mut vx = Vertex::<$dim>::zeros();
                        vx.copy_from_slice(p);
                        vx
                    })
                    .collect();

                let elems = elems.as_slice()?;
                let elems = elems
                    .chunks($cell_dim)
                    .map(|x| x.try_into().unwrap())
                    .collect();

                let faces = faces.as_slice()?;
                let faces = faces
                    .chunks($face_dim)
                    .map(|x| x.try_into().unwrap())
                    .collect();

                Ok(Self($name::new(
                    coords,
                    elems,
                    etags.to_vec().unwrap(),
                    faces,
                    ftags.to_vec().unwrap(),
                )))
            }

            /// Number of vertices
            pub fn n_verts(&self) -> usize {
                self.0.n_verts()
            }

            /// Get a copy of the vertices
            fn get_verts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
                PyArray::from_vec(py, self.0.verts().flatten().cloned().collect())
                    .reshape([self.0.n_verts(), $dim])
            }

            /// Number of elements
            fn n_elems(&self) -> usize {
                self.0.n_elems()
            }

            /// Get a copy of the elements
            fn get_elems<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<usize>>> {
                PyArray::from_vec(py, self.0.elems().flatten().cloned().collect())
                    .reshape([self.0.n_elems(), $cell_dim])
            }

            /// Get a copy of the element tags
            fn get_etags<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Tag>>> {
                Ok(PyArray::from_vec(py, self.0.etags().collect()))
            }

            /// Number of faces
            fn n_faces(&self) -> usize {
                self.0.n_faces()
            }

            /// Get a copy of the faces
            fn get_faces<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<usize>>> {
                PyArray::from_vec(py, self.0.faces().flatten().cloned().collect())
                    .reshape([self.0.n_faces(), $face_dim])
            }

            /// Get a copy of the face tags
            fn get_ftags<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Tag>>> {
                Ok(PyArray::from_vec(py, self.0.ftags().collect()))
            }

            /// Fix the element & face orientation (if possible) and tag internal faces (if needed)
            fn fix(&mut self) -> PyResult<()> {
                let all_faces = self.0.compute_faces();
                self.0.fix_orientation(&all_faces);
                self.0.tag_internal_faces(&all_faces);
                self.0
                    .check(&all_faces)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }

            /// Export the mesh to a `.vtu` file
            fn write_vtk(&self, file_name: &str) -> PyResult<()> {
                self.0
                    .write_vtk(file_name)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }

            /// Export the mesh to a `.meshb` file
            fn write_meshb(&self, file_name: &str) -> PyResult<()> {
                self.0
                    .write_meshb(file_name)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }

            /// Read a `.meshb` file
            #[classmethod]
            fn from_meshb(_cls: &Bound<'_, PyType>, file_name: &str) -> PyResult<Self> {
                Ok(Self(
                    $name::from_meshb(file_name)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                ))
            }

            /// Split and add quandrangles
            fn add_quadrangles(
                &mut self,
                elems: PyReadonlyArray2<usize>,
                etags: PyReadonlyArray1<Tag>,
            ) -> PyResult<()> {
                let n = elems.shape()[0];
                if elems.shape()[1] != 4 {
                    return Err(PyValueError::new_err("Invalid dimension 1 for elems"));
                }
                if etags.shape()[0] != n {
                    return Err(PyValueError::new_err("Invalid dimension 0 for etags"));
                }

                self.0.add_quadrangles(
                    elems.as_slice()?.chunks(4).map(|x| x.try_into().unwrap()),
                    etags.as_slice()?.iter().cloned(),
                );

                Ok(())
            }

            /// Split and add pyramids
            fn add_pyramids(
                &mut self,
                elems: PyReadonlyArray2<usize>,
                etags: PyReadonlyArray1<Tag>,
            ) -> PyResult<()> {
                let n = elems.shape()[0];
                if elems.shape()[1] != 5 {
                    return Err(PyValueError::new_err("Invalid dimension 1 for elems"));
                }
                if etags.shape()[0] != n {
                    return Err(PyValueError::new_err("Invalid dimension 0 for etags"));
                }

                self.0.add_pyramids(
                    elems.as_slice()?.chunks(5).map(|x| x.try_into().unwrap()),
                    etags.as_slice()?.iter().cloned(),
                );

                Ok(())
            }

            /// Split and add prisms
            fn add_prisms(
                &mut self,
                elems: PyReadonlyArray2<usize>,
                etags: PyReadonlyArray1<Tag>,
            ) -> PyResult<()> {
                let n = elems.shape()[0];
                if elems.shape()[1] != 6 {
                    return Err(PyValueError::new_err("Invalid dimension 1 for elems"));
                }
                if etags.shape()[0] != n {
                    return Err(PyValueError::new_err("Invalid dimension 0 for etags"));
                }

                self.0.add_prisms(
                    elems.as_slice()?.chunks(6).map(|x| x.try_into().unwrap()),
                    etags.as_slice()?.iter().cloned(),
                );

                Ok(())
            }

            /// Split and add hexahedra
            fn add_hexahedra<'py>(
                &mut self,
                py: Python<'py>,
                elems: PyReadonlyArray2<usize>,
                etags: PyReadonlyArray1<Tag>,
            ) -> PyResult<Bound<'py, PyArray1<usize>>> {
                let n = elems.shape()[0];
                if elems.shape()[1] != 8 {
                    return Err(PyValueError::new_err("Invalid dimension 1 for elems"));
                }
                if etags.shape()[0] != n {
                    return Err(PyValueError::new_err("Invalid dimension 0 for etags"));
                }

                let ids = self.0.add_hexahedra(
                    elems.as_slice()?.chunks(8).map(|x| x.try_into().unwrap()),
                    etags.as_slice()?.iter().cloned(),
                );

                Ok(PyArray1::from_vec(py, ids))
            }

            /// Reorder the mesh using RCM
            fn reorder_rcm<'py>(
                &mut self,
                py: Python<'py>,
            ) -> (
                Self,
                Bound<'py, PyArray1<usize>>,
                Bound<'py, PyArray1<usize>>,
                Bound<'py, PyArray1<usize>>,
            ) {
                let (new_mesh, vert_ids, face_ids, elem_ids) = self.0.reorder_rcm();
                (
                    Self(new_mesh),
                    PyArray1::from_vec(py, vert_ids),
                    PyArray1::from_vec(py, face_ids),
                    PyArray1::from_vec(py, elem_ids),
                )
            }

            /// Check that two meshes are equal
            fn check_equals(&self, other: &Self, tol: f64) -> PyResult<()> {
                self.0
                    .check_equals(&other.0, tol)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }
        }
    };
}

create_mesh!(PyMesh2d, Mesh2d, 2, 3, 2);
create_mesh!(PyBoundaryMesh2d, BoundaryMesh2d, 2, 2, 1);
create_mesh!(PyMesh3d, Mesh3d, 3, 4, 3);
create_mesh!(PyBoundaryMesh3d, BoundaryMesh3d, 3, 3, 2);

#[pymethods]
impl PyMesh2d {
    /// Build a nonuniform rectangle mesh by splitting a structured
    /// mesh
    #[classmethod]
    pub fn rectangle_mesh(
        _cls: &Bound<'_, PyType>,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let x = x.as_slice()?;
        let y = y.as_slice()?;
        Ok(Self(nonuniform_rectangle_mesh(x, y)))
    }

    /// Get the mesh boundary
    pub fn boundary<'py>(
        &self,
        py: Python<'py>,
    ) -> (PyBoundaryMesh2d, Bound<'py, PyArray1<usize>>) {
        let (bdy, ids) = self.0.boundary();
        (PyBoundaryMesh2d(bdy), PyArray1::from_vec(py, ids))
    }
}

#[pymethods]
impl PyMesh3d {
    /// Build a nonuniform box mesh by splitting a structured
    /// mesh
    #[classmethod]
    pub fn box_mesh(
        _cls: &Bound<'_, PyType>,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
        z: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let x = x.as_slice()?;
        let y = y.as_slice()?;
        let z = z.as_slice()?;
        Ok(Self(nonuniform_box_mesh(x, y, z)))
    }

    /// Get the mesh boundary
    pub fn boundary<'py>(
        &self,
        py: Python<'py>,
    ) -> (PyBoundaryMesh3d, Bound<'py, PyArray1<usize>>) {
        let (bdy, ids) = self.0.boundary();
        (PyBoundaryMesh3d(bdy), PyArray1::from_vec(py, ids))
    }
}
