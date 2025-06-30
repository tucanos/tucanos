#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]
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
#[cfg(feature = "metis")]
use tmesh::mesh::partition::{MetisKWay, MetisPartitioner, MetisRecursive};
use tmesh::{
    Tag, Vertex,
    interpolate::{InterpolationMethod, Interpolator},
    mesh::{
        BoundaryMesh2d, BoundaryMesh3d, Mesh, Mesh2d, Mesh3d, nonuniform_box_mesh,
        nonuniform_rectangle_mesh,
        partition::{HilbertPartitioner, KMeansPartitioner2d, KMeansPartitioner3d, RCMPartitioner},
        read_stl,
    },
};

/// Partitionner type
#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq, Eq)]
pub enum PyPartitionerType {
    /// Hilbert
    Hilbert,
    /// RCM
    #[allow(clippy::upper_case_acronyms)]
    RCM,
    /// KMeans
    KMeans,
    #[cfg(feature = "metis")]
    /// Metis - Recursive
    MetisRecursive,
    #[cfg(feature = "metis")]
    /// Metis - KWay
    MetisKWay,
}

macro_rules! create_mesh {
    ($pyname: ident, $name: ident, $dim: expr, $cell_dim: expr, $face_dim: expr) => {
        #[doc = concat!("Python binding for ", stringify!($name))]
        #[pyclass]
        pub struct $pyname(pub(crate) $name);

        #[pymethods]
        impl $pyname {
            /// Create a new mesh from coordinates, connectivities and tags
            #[new]
            fn new(
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
                let coords = coords.chunks($dim).map(|p| {
                    let mut vx = Vertex::<$dim>::zeros();
                    vx.copy_from_slice(p);
                    vx
                });

                let elems = elems.as_slice()?;
                let elems = elems.chunks($cell_dim).map(|x| x.try_into().unwrap());

                let faces = faces.as_slice()?;
                let faces = faces.chunks($face_dim).map(|x| x.try_into().unwrap());

                let mut res = $name::empty();

                res.add_verts(coords);
                res.add_elems(elems, etags.to_vec().unwrap().iter().cloned());
                res.add_faces(faces, ftags.to_vec().unwrap().iter().cloned());

                Ok(Self(res))
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

            /// Get a copy of the elements
            fn get_elems<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<usize>>> {
                PyArray::from_vec(py, self.0.elems().flatten().collect())
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
                PyArray::from_vec(py, self.0.faces().flatten().collect())
                    .reshape([self.0.n_faces(), $face_dim])
            }

            /// Get a copy of the face tags
            fn get_ftags<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Tag>>> {
                Ok(PyArray::from_vec(py, self.0.ftags().collect()))
            }

            /// Fix the element & face orientation (if possible) and tag internal faces (if needed)
            fn fix(&mut self) -> PyResult<()> {
                let all_faces = self.0.all_faces();
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

            /// Reorder the mesh using a Hilbert curve
            fn reorder_hilbert<'py>(
                &mut self,
                py: Python<'py>,
            ) -> (
                Self,
                Bound<'py, PyArray1<usize>>,
                Bound<'py, PyArray1<usize>>,
                Bound<'py, PyArray1<usize>>,
            ) {
                let (new_mesh, vert_ids, face_ids, elem_ids) = self.0.reorder_hilbert();
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

            /// Partition the mesh
            #[pyo3(signature = (n_parts, method=PyPartitionerType::Hilbert, weights=None))]
            fn partition(
                &mut self,
                n_parts: usize,
                method: PyPartitionerType,
                weights: Option<PyReadonlyArray1<f64>>,
            ) -> PyResult<(f64, f64)> {
                let weights = weights.map(|x| x.as_slice().unwrap().to_vec());
                match method {
                    PyPartitionerType::Hilbert => self
                        .0
                        .partition::<HilbertPartitioner>(n_parts, weights)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string())),
                    PyPartitionerType::RCM => self
                        .0
                        .partition::<RCMPartitioner>(n_parts, weights)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string())),
                    PyPartitionerType::KMeans => match $dim {
                        3 => self
                            .0
                            .partition::<KMeansPartitioner3d>(n_parts, weights)
                            .map_err(|e| PyRuntimeError::new_err(e.to_string())),
                        2 => self
                            .0
                            .partition::<KMeansPartitioner2d>(n_parts, weights)
                            .map_err(|e| PyRuntimeError::new_err(e.to_string())),
                        _ => unimplemented!(),
                    },
                    #[cfg(feature = "metis")]
                    PyPartitionerType::MetisRecursive => self
                        .0
                        .partition::<MetisPartitioner<MetisRecursive>>(n_parts, weights)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string())),

                    #[cfg(feature = "metis")]
                    PyPartitionerType::MetisKWay => self
                        .0
                        .partition::<MetisPartitioner<MetisKWay>>(n_parts, weights)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string())),
                }
            }

            /// Uniform mesh split
            fn split(&self) -> Self {
                let msh = self.0.split();
                Self(msh)
            }

            /// Split and add prisms
            #[pyo3(signature = (data, verts, tol=1e-3, nearest=false))]
            fn interpolate<'py>(
                &mut self,
                py: Python<'py>,
                data: PyReadonlyArray2<f64>,
                verts: PyReadonlyArray2<f64>,
                tol: f64,
                nearest: bool,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if data.shape()[0] != self.0.n_verts() {
                    return Err(PyValueError::new_err("Invalid dimension 0 for data"));
                }
                let m = data.shape()[1];

                if verts.shape()[1] != $dim {
                    return Err(PyValueError::new_err("Invalid dimension 1 for verts"));
                }

                let method = if nearest {
                    InterpolationMethod::Nearest
                } else {
                    InterpolationMethod::Linear(tol)
                };

                let interp = Interpolator::new(&self.0, method);

                let res = interp.interpolate(
                    data.as_slice()?,
                    verts
                        .as_slice()?
                        .chunks($dim)
                        .map(|x| Vertex::<$dim>::from_column_slice(x)),
                );

                PyArray::from_vec(py, res).reshape([verts.shape()[0], m])
            }

            // Compute the skewness for all internal faces in the mesh
            /// Skewness is the normalized distance between a line that connects two
            /// adjacent cell centroids and the distance from that line to the shared
            /// face’s center.
            fn face_skewnesses<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<(Bound<'py, PyArray2<usize>>, Bound<'py, PyArray1<f64>>)> {
                let all_faces = self.0.all_faces();
                let res = self.0.face_skewnesses(&all_faces);

                let mut ids = Vec::new();
                let mut vals = Vec::new();
                for (i, j, v) in res {
                    ids.push(i);
                    ids.push(j);
                    vals.push(v);
                }
                Ok((
                    PyArray::from_vec(py, ids).reshape([vals.len(), 2])?,
                    PyArray::from_vec(py, vals),
                ))
            }

            /// Compute the edge ratio for all the elements in the mesh
            #[must_use]
            fn edge_length_ratios<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                let res = self.0.edge_length_ratios().collect::<Vec<_>>();
                PyArray::from_vec(py, res)
            }

            /// Compute the ratio of inscribed radius to circumradius
            /// (normalized to be between 0 and 1) for all the elements in the mesh
            #[must_use]
            fn elem_gammas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                let res = self.0.elem_gammas().collect::<Vec<_>>();
                PyArray::from_vec(py, res)
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
    #[allow(clippy::needless_pass_by_value)]
    fn rectangle_mesh(
        _cls: &Bound<'_, PyType>,
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let x = x.as_slice()?;
        let y = y.as_slice()?;
        Ok(Self(nonuniform_rectangle_mesh(x, y)))
    }

    /// Get the mesh boundary
    fn boundary<'py>(&self, py: Python<'py>) -> (PyBoundaryMesh2d, Bound<'py, PyArray1<usize>>) {
        let (bdy, ids) = self.0.boundary();
        (PyBoundaryMesh2d(bdy), PyArray1::from_vec(py, ids))
    }
}

#[pymethods]
impl PyBoundaryMesh3d {
    /// Read a stl file
    #[classmethod]
    fn read_stl(_cls: &Bound<'_, PyType>, file_name: &str) -> PyResult<Self> {
        let msh = read_stl(file_name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self(msh))
    }
}

#[pymethods]
impl PyMesh3d {
    /// Build a nonuniform box mesh by splitting a structured
    /// mesh
    #[classmethod]
    #[allow(clippy::needless_pass_by_value)]
    fn box_mesh(
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
    fn boundary<'py>(&self, py: Python<'py>) -> (PyBoundaryMesh3d, Bound<'py, PyArray1<usize>>) {
        let (bdy, ids) = self.0.boundary();
        (PyBoundaryMesh3d(bdy), PyArray1::from_vec(py, ids))
    }
}
