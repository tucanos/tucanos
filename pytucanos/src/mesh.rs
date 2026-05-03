#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]
//! Python bindings for simplex meshes
use super::Idx;
use numpy::{
    PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{
    Bound, PyResult, Python,
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods,
    types::{PyDict, PyDictMethods, PyType},
};
use std::collections::HashMap;
#[cfg(feature = "metis")]
use tmesh::mesh::partition::{MetisKWay, MetisPartitioner, MetisRecursive};
use tmesh::{
    Tag, Vertex,
    interpolate::{InterpolationMethod, Interpolator},
    io::VTUFile,
    mesh::{
        AdativeBoundsQuadraticTetrahedron, AdativeBoundsQuadraticTriangle, Edge, GSimplex,
        GenericMesh, GradientMethod, Hexahedron, Mesh, Prism, Pyramid, Quadrangle, QuadraticEdge,
        QuadraticTetrahedron, QuadraticTriangle, Simplex, SolutionLocation, Tetrahedron, Triangle,
        ball_mesh, circle_mesh, nonuniform_box_mesh, nonuniform_rectangle_mesh,
        partition::{HilbertPartitioner, KMeansPartitioner2d, KMeansPartitioner3d, RCMPartitioner},
        quadratic_circle_mesh, quadratic_sphere_mesh, read_stl, sphere_mesh,
        to_quadratic::{to_quadratic_tetrahedron_mesh, to_quadratic_triangle_mesh},
    },
};
use tucanos::geometry::orient_geometry;
/// Partitionner type
#[pyclass(eq, eq_int, from_py_object)]
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

/// Validate that coords array has correct shape for dimension D
fn validate_coords_shape<const D: usize>(shape: &[usize]) -> PyResult<()> {
    if shape.len() < 2 || shape[1] != D {
        return Err(PyValueError::new_err(format!(
            "Invalid dimension 1 for coords (expecting {}, got {})",
            D,
            shape.get(1).copied().unwrap_or(0)
        )));
    }
    Ok(())
}

/// Validate that elems array has correct shape for cell type C
fn validate_elems_shape<C: Simplex>(shape: &[usize]) -> PyResult<()> {
    if shape.len() < 2 || shape[1] != C::N_VERTS {
        return Err(PyValueError::new_err(format!(
            "Invalid dimension 1 for elems (expecting {}, got {})",
            C::N_VERTS,
            shape.get(1).copied().unwrap_or(0)
        )));
    }
    Ok(())
}

/// Validate that faces array has correct shape for face type
fn validate_faces_shape<C: Simplex>(shape: &[usize]) -> PyResult<()> {
    if shape.len() < 2 || shape[1] != C::FACE::N_VERTS {
        return Err(PyValueError::new_err(format!(
            "Invalid dimension 1 for faces (expecting {}, got {})",
            C::FACE::N_VERTS,
            shape.get(1).copied().unwrap_or(0)
        )));
    }
    Ok(())
}

/// Validate that tags array has correct length
fn validate_tags_length(tags_len: usize, expected: usize, name: &str) -> PyResult<()> {
    if tags_len != expected {
        return Err(PyValueError::new_err(format!(
            "Invalid dimension 0 for {name} (expecting {expected}, got {tags_len})"
        )));
    }
    Ok(())
}

/// Convert numpy array to iterator of Vertex<D>
fn coords_to_vertices<const D: usize>(
    coords: &[f64],
) -> impl ExactSizeIterator<Item = Vertex<D>> + '_ {
    coords.chunks_exact(D).map(|p| {
        let mut vx = Vertex::<D>::zeros();
        vx.copy_from_slice(p);
        vx
    })
}

/// Convert numpy array to iterator of cells
fn slice_to_cells<C: Simplex>(elems: &[Idx]) -> impl ExactSizeIterator<Item = C> + '_ {
    let it = elems.chunks_exact(C::N_VERTS);
    #[cfg(not(feature = "32bit-ints"))]
    let r = it.map(|x| C::from_iter(x.iter().copied()));
    #[cfg(feature = "32bit-ints")]
    let r = it.map(|x| C::from_iter(x.iter().copied().map(|i| i.try_into().unwrap())));
    r
}

/// Convert numpy array to iterator of faces
fn slice_to_faces<C: Simplex>(faces: &[Idx]) -> impl ExactSizeIterator<Item = C::FACE> + '_ {
    let it = faces.chunks_exact(C::FACE::N_VERTS);
    #[cfg(not(feature = "32bit-ints"))]
    let r = it.map(|x| C::FACE::from_iter(x.iter().copied()));
    #[cfg(feature = "32bit-ints")]
    let r = it.map(|x| C::FACE::from_iter(x.iter().copied().map(|i| i.try_into().unwrap())));
    r
}

/// Create a PyArray2 from vertices with automatic 32bit-int conversion
fn verts_to_pyarray<const D: usize>(
    py: Python<'_>,
    verts: impl Iterator<Item = impl AsRef<[f64]>>,
    n_verts: usize,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let mut res = Vec::with_capacity(D * n_verts);
    for v in verts {
        for &x in v.as_ref() {
            res.push(x);
        }
    }
    PyArray::from_vec(py, res).reshape([n_verts, D])
}

/// Create a PyArray2 from elements with automatic 32bit-int conversion
fn elems_to_pyarray<C: Simplex>(
    py: Python<'_>,
    elems: impl Iterator<Item = C>,
    n_elems: usize,
) -> PyResult<Bound<'_, PyArray2<Idx>>> {
    #[cfg(not(feature = "32bit-ints"))]
    let res = PyArray::from_vec(py, elems.flatten().collect()).reshape([n_elems, C::N_VERTS]);

    #[cfg(feature = "32bit-ints")]
    let res = PyArray::from_vec(py, elems.flatten().map(|x| x.try_into().unwrap()).collect())
        .reshape([n_elems, C::N_VERTS]);

    res
}

/// Create a PyArray2 from faces with automatic 32bit-int conversion
fn faces_to_pyarray<C: Simplex>(
    py: Python<'_>,
    faces: impl Iterator<Item = C::FACE>,
    n_faces: usize,
) -> PyResult<Bound<'_, PyArray2<Idx>>> {
    #[cfg(not(feature = "32bit-ints"))]
    let res = PyArray::from_vec(py, faces.flatten().collect()).reshape([n_faces, C::FACE::N_VERTS]);

    #[cfg(feature = "32bit-ints")]
    let res = PyArray::from_vec(py, faces.flatten().map(|x| x.try_into().unwrap()).collect())
        .reshape([n_faces, C::FACE::N_VERTS]);

    res
}

/// Map Result error to PyRuntimeError
fn to_py_err<T>(result: Result<T, impl std::fmt::Display>) -> PyResult<T> {
    result.map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

macro_rules! create_mesh {
    ($pyname: ident, $dim: expr, $cell: ident) => {
        #[doc = concat!("Python binding for ", stringify!($pyname))]
        #[pyclass]
        pub struct $pyname(pub GenericMesh<$dim, $cell<Idx>>);
    };
}

#[macro_export]
macro_rules! impl_mesh {
    ($pyname: ident, $dim: expr, $cell: ident) => {
        #[pymethods]
        impl $pyname {
            /// Create a new mesh from coordinates, connectivities and tags
            #[new]
            fn new(
                coords: PyReadonlyArray2<f64>,
                elems: PyReadonlyArray2<Idx>,
                etags: PyReadonlyArray1<Tag>,
                faces: PyReadonlyArray2<Idx>,
                ftags: PyReadonlyArray1<Tag>,
            ) -> PyResult<Self> {
                validate_coords_shape::<$dim>(coords.shape())?;
                validate_elems_shape::<$cell<Idx>>(elems.shape())?;
                validate_tags_length(etags.shape()[0], elems.shape()[0], "etags")?;
                validate_faces_shape::<$cell<Idx>>(faces.shape())?;
                validate_tags_length(ftags.shape()[0], faces.shape()[0], "ftags")?;

                let coords_iter = coords_to_vertices::<$dim>(coords.as_slice()?);
                let elems_iter = slice_to_cells::<$cell<Idx>>(elems.as_slice()?);
                let faces_iter = slice_to_faces::<$cell<Idx>>(faces.as_slice()?);

                let mut res = GenericMesh::empty();
                res.add_verts(coords_iter);
                res.add_elems(elems_iter, etags.to_vec().unwrap().iter().cloned());
                res.add_faces(faces_iter, ftags.to_vec().unwrap().iter().cloned());

                Ok(Self(res))
            }

            /// Number of vertices
            pub fn n_verts(&self) -> usize {
                self.0.n_verts()
            }

            /// Get a copy of the vertices
            fn get_verts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
                verts_to_pyarray::<$dim>(py, self.0.verts(), self.n_verts())
            }

            /// Number of elements
            fn n_elems(&self) -> usize {
                self.0.n_elems()
            }

            /// Get a copy of the elements
            fn get_elems<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Idx>>> {
                elems_to_pyarray::<$cell<Idx>>(py, self.0.elems(), self.n_elems())
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
            fn get_faces<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Idx>>> {
                faces_to_pyarray::<$cell<Idx>>(py, self.0.faces(), self.n_faces())
            }

            /// Get a copy of the face tags
            fn get_ftags<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<Tag>>> {
                Ok(PyArray::from_vec(py, self.0.ftags().collect()))
            }

            /// Fix the element & face orientation (if possible) and tag internal faces (if needed)
            fn fix<'py>(
                &mut self,
                py: Python<'py>,
            ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
                let (bdy, ifc) = to_py_err(self.0.fix())?;
                let dict_bdy = PyDict::new(py);
                for (k, v) in bdy.iter() {
                    dict_bdy.set_item(k, v)?;
                }
                let dict_ifc = PyDict::new(py);
                for (k, v) in ifc.iter() {
                    dict_ifc.set_item((k[0], k[1]), v)?;
                }

                Ok((dict_bdy, dict_ifc))
            }

            /// Export the mesh to a `.vtu` file
            #[pyo3(signature = (file_name, vert_data=None, elem_data=None))]
            fn write_vtk(
                &self,
                file_name: &str,
                vert_data: Option<HashMap<String, PyReadonlyArray2<f64>>>,
                elem_data: Option<HashMap<String, PyReadonlyArray2<f64>>>,
            ) -> PyResult<()> {
                let mut writer = VTUFile::from_mesh(&self.0);
                if let Some(data) = vert_data.as_ref() {
                    for (name, arr) in data.iter() {
                        let n = arr.shape()[1];
                        let arr = arr.as_slice()?;
                        writer.add_point_data(name, n, arr.iter().copied())
                    }
                }
                if let Some(data) = elem_data.as_ref() {
                    for (name, arr) in data.iter() {
                        let n = arr.shape()[1];
                        let arr = arr.as_slice()?;
                        writer.add_cell_data(name, n, arr.iter().copied())
                    }
                }
                to_py_err(writer.export(file_name))
            }

            /// Export the mesh to a `.meshb` file
            fn write_meshb(&self, file_name: &str) -> PyResult<()> {
                to_py_err(self.0.write_meshb(file_name))
            }

            /// Write a solution defined at the vertices to a .sol(b) file
            pub fn write_solb(&self, fname: &str, arr: PyReadonlyArray2<f64>) -> PyResult<()> {
                to_py_err(self.0.write_solb(
                    &arr.to_vec().unwrap(),
                    fname,
                    SolutionLocation::Vertices,
                ))
            }

            /// Write a solution defined at the elements to a .sol(b) file
            pub fn write_elem_solb(&self, fname: &str, arr: PyReadonlyArray2<f64>) -> PyResult<()> {
                to_py_err(self.0.write_solb(
                    &arr.to_vec().unwrap(),
                    fname,
                    SolutionLocation::Elements,
                ))
            }

            /// Write a solution defined at the faces to a .sol(b) file
            pub fn write_face_solb(&self, fname: &str, arr: PyReadonlyArray2<f64>) -> PyResult<()> {
                to_py_err(
                    self.0
                        .write_solb(&arr.to_vec().unwrap(), fname, SolutionLocation::Faces),
                )
            }

            /// Read a `.meshb` file
            #[classmethod]
            fn from_meshb(_cls: &Bound<'_, PyType>, file_name: &str) -> PyResult<Self> {
                Ok(Self(to_py_err(GenericMesh::from_meshb(file_name))?))
            }

            /// Read a solution stored in a .sol(b) file
            #[classmethod]
            pub fn read_solb<'py>(
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                fname: &str,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                use pyo3::exceptions::PyRuntimeError;

                let res = GenericMesh::<$dim, $cell<Idx>>::read_solb(fname);
                match res {
                    Ok((sol, m)) => {
                        let n = sol.len() / m;
                        Ok(PyArray::from_vec(py, sol).reshape([n, m])?)
                    }
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }

            #[doc = concat!("Create an empty ", stringify!($pyname))]
            #[classmethod]
            pub fn empty(_cls: &Bound<'_, PyType>) -> Self {
                Self(GenericMesh::empty())
            }

            /// Add vertices, faces and elements from another mesh
            /// If `tol` is not None, vertices on the boundaries of `self`
            /// and `other` are merged if closer than the tolerance.
            pub fn add(&mut self, other: &Self, tol: Option<f64>) {
                self.0.add(&other.0, |_| true, |_| true, tol);
            }

            /// Add vertices
            pub fn add_verts(&mut self, coords: PyReadonlyArray2<f64>) -> PyResult<()> {
                validate_coords_shape::<$dim>(coords.shape())?;
                let coords_iter = coords_to_vertices::<$dim>(coords.as_slice()?);
                self.0.add_verts(coords_iter);
                Ok(())
            }

            /// Clear all faces
            pub fn clear_faces(&mut self) {
                self.0.clear_faces()
            }

            /// Add faces
            pub fn add_faces(
                &mut self,
                faces: PyReadonlyArray2<Idx>,
                ftags: PyReadonlyArray1<Tag>,
            ) -> PyResult<()> {
                validate_faces_shape::<$cell<Idx>>(faces.shape())?;
                let faces_iter = slice_to_faces::<$cell<Idx>>(faces.as_slice()?);
                self.0
                    .add_faces(faces_iter, ftags.as_slice()?.iter().copied());
                Ok(())
            }

            /// Add elements
            pub fn add_elems(
                &mut self,
                elems: PyReadonlyArray2<Idx>,
                etags: PyReadonlyArray1<Tag>,
            ) -> PyResult<()> {
                validate_elems_shape::<$cell<Idx>>(elems.shape())?;
                let elems_iter = slice_to_cells::<$cell<Idx>>(elems.as_slice()?);
                self.0
                    .add_elems(elems_iter, etags.as_slice()?.iter().copied());
                Ok(())
            }

            /// Split and add quandrangles
            fn add_quadrangles(
                &mut self,
                elems: PyReadonlyArray2<Idx>,
                etags: PyReadonlyArray1<Tag>,
            ) -> PyResult<()> {
                validate_tags_length(etags.shape()[0], elems.shape()[0], "etags")?;
                if elems.shape()[1] != 4 {
                    return Err(PyValueError::new_err("Invalid dimension 1 for elems"));
                }
                self.0.add_quadrangles(
                    elems.as_slice()?.chunks(4).map(|x| {
                        let quad: [Idx; 4] = x.try_into().unwrap();
                        #[cfg(not(feature = "32bit-ints"))]
                        let res = Quadrangle::<Idx>::from_iter(quad);
                        #[cfg(feature = "32bit-ints")]
                        let res = Quadrangle::<Idx>::from_iter(
                            quad.iter().map(|&x| x.try_into().unwrap()),
                        );
                        res
                    }),
                    etags.as_slice()?.iter().cloned(),
                );
                Ok(())
            }

            /// Split and add pyramids
            fn add_pyramids(
                &mut self,
                elems: PyReadonlyArray2<Idx>,
                etags: PyReadonlyArray1<Tag>,
            ) -> PyResult<()> {
                validate_tags_length(etags.shape()[0], elems.shape()[0], "etags")?;
                if elems.shape()[1] != 5 {
                    return Err(PyValueError::new_err("Invalid dimension 1 for elems"));
                }
                self.0.add_pyramids(
                    elems.as_slice()?.chunks(5).map(|x| {
                        let pyr: [Idx; 5] = x.try_into().unwrap();
                        #[cfg(not(feature = "32bit-ints"))]
                        let res = Pyramid::<Idx>::from_iter(pyr);
                        #[cfg(feature = "32bit-ints")]
                        let res =
                            Pyramid::<Idx>::from_iter(pyr.iter().map(|&x| x.try_into().unwrap()));
                        res
                    }),
                    etags.as_slice()?.iter().cloned(),
                );
                Ok(())
            }

            /// Split and add prisms
            fn add_prisms(
                &mut self,
                elems: PyReadonlyArray2<Idx>,
                etags: PyReadonlyArray1<Tag>,
            ) -> PyResult<()> {
                validate_tags_length(etags.shape()[0], elems.shape()[0], "etags")?;
                if elems.shape()[1] != 6 {
                    return Err(PyValueError::new_err("Invalid dimension 1 for elems"));
                }
                self.0.add_prisms(
                    elems.as_slice()?.chunks(6).map(|x| {
                        let pri: [Idx; 6] = x.try_into().unwrap();
                        #[cfg(not(feature = "32bit-ints"))]
                        let res = Prism::<Idx>::from_iter(pri);
                        #[cfg(feature = "32bit-ints")]
                        let res =
                            Prism::<Idx>::from_iter(pri.iter().map(|&x| x.try_into().unwrap()));
                        res
                    }),
                    etags.as_slice()?.iter().cloned(),
                );
                Ok(())
            }

            /// Split and add hexahedra
            fn add_hexahedra<'py>(
                &mut self,
                py: Python<'py>,
                elems: PyReadonlyArray2<Idx>,
                etags: PyReadonlyArray1<Tag>,
            ) -> PyResult<Bound<'py, PyArray1<usize>>> {
                validate_tags_length(etags.shape()[0], elems.shape()[0], "etags")?;
                if elems.shape()[1] != 8 {
                    return Err(PyValueError::new_err("Invalid dimension 1 for elems"));
                }
                let ids = self.0.add_hexahedra(
                    elems.as_slice()?.chunks(8).map(|x| {
                        let hex: [Idx; 8] = x.try_into().unwrap();
                        #[cfg(not(feature = "32bit-ints"))]
                        let res = Hexahedron::<Idx>::from_iter(hex);
                        #[cfg(feature = "32bit-ints")]
                        let res = Hexahedron::<Idx>::from_iter(
                            hex.iter().map(|&x| x.try_into().unwrap()),
                        );
                        res
                    }),
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
                to_py_err(self.0.check_equals(&other.0, tol))
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
                    PyPartitionerType::Hilbert => {
                        to_py_err(self.0.partition::<HilbertPartitioner>(n_parts, weights))
                    }
                    PyPartitionerType::RCM => {
                        to_py_err(self.0.partition::<RCMPartitioner>(n_parts, weights))
                    }
                    PyPartitionerType::KMeans => match $dim {
                        3 => to_py_err(self.0.partition::<KMeansPartitioner3d>(n_parts, weights)),
                        2 => to_py_err(self.0.partition::<KMeansPartitioner2d>(n_parts, weights)),
                        _ => unimplemented!(),
                    },
                    #[cfg(feature = "metis")]
                    PyPartitionerType::MetisRecursive => to_py_err(
                        self.0
                            .partition::<MetisPartitioner<MetisRecursive>>(n_parts, weights),
                    ),
                    #[cfg(feature = "metis")]
                    PyPartitionerType::MetisKWay => to_py_err(
                        self.0
                            .partition::<MetisPartitioner<MetisKWay>>(n_parts, weights),
                    ),
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
                if data.shape()[0] != self.n_verts() {
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

            /// Smooth a field defined at the mesh vertices using a 1st order least-square
            /// approximation
            #[pyo3(signature = (arr, weight_exp=2, order=1))]
            pub fn smooth<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
                weight_exp: i32,
                order: i32,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if arr.shape()[0] != self.0.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if arr.shape()[1] != 1 {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let method = match order {
                    1 => GradientMethod::LinearLeastSquares(weight_exp),
                    2 => GradientMethod::LinearLeastSquares(weight_exp),
                    _ => unreachable!("Invalid order {order}"),
                };

                let res = self.0.smooth(method, arr.as_slice().unwrap());

                Ok(PyArray::from_vec(py, res)
                    .reshape([self.0.n_verts(), 1])
                    .unwrap())
            }

            /// Compute the gradient of a field defined at the mesh vertices using a 1st order
            /// least-square approximation
            #[pyo3(signature = (arr, weight_exp=2, order=1))]
            pub fn gradient<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
                weight_exp: i32,
                order: i32,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if arr.shape()[0] != self.0.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if arr.shape()[1] != 1 {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let method = match order {
                    1 => GradientMethod::LinearLeastSquares(weight_exp),
                    2 => GradientMethod::LinearLeastSquares(weight_exp),
                    _ => unreachable!("Invalid order {order}"),
                };
                let res = self.0.gradient(method, arr.as_slice().unwrap());

                Ok(PyArray::from_vec(py, res)
                    .reshape([self.0.n_verts(), $dim])
                    .unwrap())
            }

            /// Compute the hessian of a field defined at the mesh vertices using a 2nd order
            /// least-square approximation
            #[pyo3(signature = (arr, weight_exp=2))]
            pub fn hessian<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
                weight_exp: i32,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if arr.shape()[0] != self.0.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if arr.shape()[1] != 1 {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let res = self.0.hessian(
                    GradientMethod::QuadraticLeastSquares(weight_exp),
                    arr.as_slice().unwrap(),
                );

                Ok(PyArray::from_vec(py, res)
                    .reshape([self.0.n_verts(), $dim * ($dim + 1) / 2])
                    .unwrap())
            }

            /// Compute the hessian of a field defined at the mesh vertices using a 2nd order
            /// least-square approximation
            #[pyo3(signature = (arr))]
            pub fn hessian_l2proj<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if arr.shape()[0] != self.0.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if arr.shape()[1] != 1 {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let res = self
                    .0
                    .hessian(GradientMethod::L2Projection, arr.as_slice().unwrap());

                Ok(PyArray::from_vec(py, res)
                    .reshape([self.0.n_verts(), $dim * ($dim + 1) / 2])
                    .unwrap())
            }

            /// Convert a (scalar or vector) field defined at the element centers (P0) to a field
            /// defined at the vertices (P1) using a weighted average.
            pub fn elem_data_to_vertex_data<'py>(
                &mut self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if arr.shape()[0] != self.0.n_elems() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                let v2e = self.0.vertex_to_elems();
                let res = self
                    .0
                    .elem_data_to_vertex_data(&v2e, arr.as_slice().unwrap());

                Ok(PyArray::from_vec(py, res)
                    .reshape([self.0.n_verts(), arr.shape()[1]])
                    .unwrap())
            }

            /// Convert a field (scalar or vector) defined at the vertices (P1) to a field defined
            /// at the element centers (P0)
            pub fn vertex_data_to_elem_data<'py>(
                &mut self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if arr.shape()[0] != self.0.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                let res = self.0.vertex_data_to_elem_data(arr.as_slice().unwrap());

                Ok(PyArray::from_vec(py, res)
                    .reshape([self.0.n_elems(), arr.shape()[1]])
                    .unwrap())
            }

            /// Get the mesh volume
            pub fn vol(&self) -> f64 {
                self.0.vol()
            }

            /// Get the volume of all the mesh elements
            pub fn vols<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
                let vols = self.0.gelems().map(|ge| ge.vol()).collect();
                Ok(PyArray::from_vec(py, vols))
            }

            /// Check that the mesh is valid
            pub fn check(&self) -> PyResult<()> {
                to_py_err(self.0.check(&self.0.all_faces()))
            }
        }
    };
}

create_mesh!(PyMesh2d, 2, Triangle);
impl_mesh!(PyMesh2d, 2, Triangle);

create_mesh!(PyQuadraticMesh2d, 2, QuadraticTriangle);
impl_mesh!(PyQuadraticMesh2d, 2, QuadraticTriangle);

create_mesh!(PyBoundaryMesh2d, 2, Edge);
impl_mesh!(PyBoundaryMesh2d, 2, Edge);

create_mesh!(PyQuadraticBoundaryMesh2d, 2, QuadraticEdge);
impl_mesh!(PyQuadraticBoundaryMesh2d, 2, QuadraticEdge);

create_mesh!(PyMesh3d, 3, Tetrahedron);
impl_mesh!(PyMesh3d, 3, Tetrahedron);

create_mesh!(PyQuadraticMesh3d, 3, QuadraticTetrahedron);
impl_mesh!(PyQuadraticMesh3d, 3, QuadraticTetrahedron);

create_mesh!(PyBoundaryMesh3d, 3, Triangle);
impl_mesh!(PyBoundaryMesh3d, 3, Triangle);

create_mesh!(PyQuadraticBoundaryMesh3d, 3, QuadraticTriangle);
impl_mesh!(PyQuadraticBoundaryMesh3d, 3, QuadraticTriangle);

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

    fn to_quadratic(&self) -> PyQuadraticMesh2d {
        PyQuadraticMesh2d(to_quadratic_triangle_mesh(&self.0))
    }
}

#[pymethods]
impl PyQuadraticMesh2d {
    /// Get the mesh boundary
    fn boundary<'py>(
        &self,
        py: Python<'py>,
    ) -> (PyQuadraticBoundaryMesh2d, Bound<'py, PyArray1<usize>>) {
        let (bdy, ids) = self.0.boundary();
        (PyQuadraticBoundaryMesh2d(bdy), PyArray1::from_vec(py, ids))
    }

    /// Compute the distortion for all the elements in the mesh
    fn distortion<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let d = AdativeBoundsQuadraticTriangle::element_distortion(&self.0);
        PyArray1::from_vec(py, d)
    }
}

#[pymethods]
impl PyBoundaryMesh2d {
    /// Create a circle_mesh
    #[classmethod]
    fn circle_mesh(_cls: &Bound<'_, PyType>, r: f64, n: usize) -> Self {
        Self(circle_mesh(r, n))
    }

    fn fix_orientation(&mut self, mesh: &PyMesh2d) -> (usize, f64) {
        orient_geometry(&mesh.0, &mut self.0)
    }
}

#[pymethods]
impl PyQuadraticBoundaryMesh2d {
    /// Create a circle_mesh
    #[classmethod]
    fn circle_mesh(_cls: &Bound<'_, PyType>, r: f64, n: usize) -> Self {
        Self(quadratic_circle_mesh(r, n))
    }

    fn fix_orientation(&mut self, mesh: &PyMesh2d) -> (usize, f64) {
        orient_geometry(&mesh.0, &mut self.0)
    }
}

#[pymethods]
impl PyBoundaryMesh3d {
    /// Read a stl file
    #[classmethod]
    fn read_stl(_cls: &Bound<'_, PyType>, file_name: &str) -> PyResult<Self> {
        Ok(Self(to_py_err(read_stl(file_name))?))
    }

    /// Create a sphere
    #[classmethod]
    fn sphere_mesh(_cls: &Bound<'_, PyType>, r: f64, n: usize) -> Self {
        Self(sphere_mesh(r, n))
    }

    fn fix_orientation(&mut self, mesh: &PyMesh3d) -> (usize, f64) {
        orient_geometry(&mesh.0, &mut self.0)
    }
}

#[pymethods]
impl PyQuadraticBoundaryMesh3d {
    /// Create a sphere
    #[classmethod]
    fn sphere_mesh(_cls: &Bound<'_, PyType>, r: f64, n: usize) -> Self {
        Self(quadratic_sphere_mesh(r, n))
    }

    fn fix_orientation(&mut self, mesh: &PyMesh3d) -> (usize, f64) {
        orient_geometry(&mesh.0, &mut self.0)
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

    /// Build a ball mesh
    #[classmethod]
    #[allow(clippy::needless_pass_by_value)]
    fn ball_mesh(_cls: &Bound<'_, PyType>, r: f64, n: usize) -> Self {
        Self(ball_mesh(r, n))
    }

    /// Get the mesh boundary
    fn boundary<'py>(&self, py: Python<'py>) -> (PyBoundaryMesh3d, Bound<'py, PyArray1<usize>>) {
        let (bdy, ids) = self.0.boundary();
        (PyBoundaryMesh3d(bdy), PyArray1::from_vec(py, ids))
    }

    fn to_quadratic(&self) -> PyQuadraticMesh3d {
        PyQuadraticMesh3d(to_quadratic_tetrahedron_mesh(&self.0))
    }
}

#[pymethods]
impl PyQuadraticMesh3d {
    /// Get the mesh boundary
    fn boundary<'py>(
        &self,
        py: Python<'py>,
    ) -> (PyQuadraticBoundaryMesh3d, Bound<'py, PyArray1<usize>>) {
        let (bdy, ids) = self.0.boundary();
        (PyQuadraticBoundaryMesh3d(bdy), PyArray1::from_vec(py, ids))
    }

    /// Compute the distortion for all the elements in the mesh
    fn distortion<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let d = AdativeBoundsQuadraticTetrahedron::element_distortion(&self.0);
        PyArray1::from_vec(py, d)
    }
}
