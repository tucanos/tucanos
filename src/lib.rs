use log::info;
use numpy::{PyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pyfunction, pymethods, pymodule,
    types::{PyModule, PyType},
    wrap_pyfunction, PyResult, Python,
};
#[cfg(feature = "libmeshb-sys")]
use tucanos::meshb_io::{GmfElementTypes, GmfReader, GmfWriter};
use tucanos::{
    geometry::LinearGeometry,
    mesh::SimplexMesh,
    mesh_stl::orient_stl,
    metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric, Metric},
    remesher::{Remesher, RemesherParams, SmoothingType},
    topo_elems::{Edge, Elem, Tetrahedron, Triangle},
    FieldType, Idx, Mesh, Tag,
};

fn to_numpy_1d<T: numpy::Element>(py: Python<'_>, vec: Vec<T>) -> &'_ PyArray1<T> {
    PyArray::from_vec(py, vec)
}

fn to_numpy_2d<T: numpy::Element>(py: Python<'_>, vec: Vec<T>, m: usize) -> &'_ PyArray2<T> {
    let n = vec.len();
    PyArray::from_vec(py, vec).reshape([n / m, m]).unwrap()
}

fn to_numpy_1d_copy<'py, T: numpy::Element>(py: Python<'py>, vec: &[T]) -> &'py PyArray1<T> {
    PyArray::from_slice(py, vec)
}

fn to_numpy_2d_copy<'py, T: numpy::Element>(
    py: Python<'py>,
    vec: &[T],
    m: usize,
) -> &'py PyArray2<T> {
    PyArray::from_slice(py, vec)
        .reshape([vec.len() / m, m])
        .unwrap()
}

macro_rules! create_mesh {
    ($name: ident, $dim: expr, $etype: ident) => {
        #[doc = concat!("Mesh consisting of ", stringify!($etype), " in ", stringify!($dim), "D")]
        #[pyclass]
        pub struct $name {
            mesh: SimplexMesh<$dim, $etype>,
        }
        #[pymethods]
        impl $name {
            /// Create a new mesh from numpy arrays
            /// The data is copied
            #[new]
            pub fn new(
                coords: PyReadonlyArray2<f64>,
                elems: PyReadonlyArray2<Idx>,
                etags: PyReadonlyArray1<Tag>,
                faces: PyReadonlyArray2<Idx>,
                ftags: PyReadonlyArray1<Tag>,
            ) -> PyResult<Self> {
                if coords.shape()[1] != $dim {
                    return Err(PyValueError::new_err("Invalid dimension 1 for coords"));
                }
                let n = elems.shape()[0];
                if elems.shape()[1] != <$etype as Elem>::N_VERTS as usize {
                    return Err(PyValueError::new_err("Invalid dimension 1 for elems"));
                }
                if etags.shape()[0] != n {
                    return Err(PyValueError::new_err("Invalid dimension 0 for etags"));
                }
                let n = faces.shape()[0];

                if faces.shape()[1] != <$etype as Elem>::Face::N_VERTS as usize {
                    return Err(PyValueError::new_err("Invalid dimension 1 for faces"));
                }
                if ftags.shape()[0] != n {
                    return Err(PyValueError::new_err("Invalid dimension 0 for ftags"));
                }

                info!(
                    "Create a {} mesh in {}D",
                    stringify!($etype),
                    stringify!($dim)
                );
                Ok(Self {
                    mesh: SimplexMesh::<$dim, $etype>::new(
                        coords.to_vec().unwrap(),
                        elems.to_vec().unwrap(),
                        etags.to_vec().unwrap(),
                        faces.to_vec().unwrap(),
                        ftags.to_vec().unwrap(),
                    ),
                })
            }

            #[doc = concat!("Read a ", stringify!($name), " from a .mesh(b) file")]
            #[classmethod]
            #[allow(unused_variables)]
            pub fn from_meshb(_cls: &PyType, fname: &str) -> PyResult<Self> {
                #[cfg(feature = "libmeshb-sys")]
                {
                    let reader = GmfReader::new(fname);
                    if reader.is_invalid() {
                        return Err(PyRuntimeError::new_err("Cannot open the file"));
                    }
                    if reader.dim() != $dim {
                        return Err(PyRuntimeError::new_err("Invalid dimension"));
                    }

                    let coords = reader.read_vertices();
                    let etype = match <$etype as Elem>::N_VERTS {
                        3 => GmfElementTypes::Triangle,
                        4 => GmfElementTypes::Tetrahedron,
                        _ => unreachable!(),
                    };
                    let (elems, etags) = reader.read_elements(etype);

                    let etype = match <$etype as Elem>::Face::N_VERTS {
                        2 => GmfElementTypes::Edge,
                        3 => GmfElementTypes::Triangle,
                        _ => unreachable!(),
                    };
                    let (faces, ftags) = reader.read_elements(etype);

                    return Ok(Self {
                        mesh: SimplexMesh::<$dim, $etype>::new(coords, elems, etags, faces, ftags),
                    });
                }
                Err(PyRuntimeError::new_err(
                    "The meshb interface is not available",
                ))
            }

            /// Write the mesh to a .mesh(b) file
            #[allow(unused_variables)]
            pub fn write_meshb(&self, fname: &str) -> PyResult<()> {
                #[cfg(feature = "libmeshb-sys")]
                {
                    let mut writer = GmfWriter::new(fname, $dim);

                    if writer.is_invalid() {
                        return Err(PyRuntimeError::new_err("Cannot write the meshb file"));
                    }

                    writer.write_mesh(&self.mesh);

                    return Ok(());
                }
                Err(PyRuntimeError::new_err(
                    "The meshb interface is not available",
                ))
            }

            /// Get the number of vertices in the mesh
            #[must_use]
            pub fn n_verts(&self) -> Idx {
                self.mesh.n_verts()
            }

            /// Get the number of vertices in the mesh
            #[must_use]
            pub fn n_elems(&self) -> Idx {
                self.mesh.n_elems()
            }

            /// Get the number of faces in the mesh
            #[must_use]
            pub fn n_faces(&self) -> Idx {
                self.mesh.n_faces()
            }

            /// Get the volume of the mesh
            #[must_use]
            pub fn vol(&self) -> f64 {
                self.mesh.elem_vols().sum()
            }

            /// Get the volume of all the elements
            pub fn vols<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {

                let res : Vec<_> = self.mesh.elem_vols().collect();
                to_numpy_1d(py, res)
            }

            /// Compute the vertex-to-element connectivity
            pub fn compute_vertex_to_elems(&mut self) {
                self.mesh.compute_vertex_to_elems();
            }

            /// Clear the vertex-to-element connectivity
            pub fn clear_vertex_to_elems(&mut self) {
                self.mesh.clear_vertex_to_elems();
            }

            /// Compute the face-to-element connectivity
            pub fn compute_face_to_elems(&mut self) {
                self.mesh.compute_face_to_elems();
            }

            /// Clear the face-to-element connectivity
            pub fn clear_face_to_elems(&mut self) {
                self.mesh.clear_face_to_elems();
            }

            /// Compute the element-to-element connectivity
            /// face-to-element connectivity is computed if not available
            pub fn compute_elem_to_elems(&mut self) {
                self.mesh.compute_elem_to_elems();
            }

            /// Clear the element-to-element connectivity
            pub fn clear_elem_to_elems(&mut self) {
                self.mesh.clear_elem_to_elems();
            }

            /// Compute the edges
            pub fn compute_edges(&mut self) {
                self.mesh.compute_edges()
            }

            /// Clear the edges
            pub fn clear_edges(&mut self) {
                self.mesh.clear_edges()
            }

            /// Compute the vertex-to-vertex connectivity
            /// Edges are computed if not available
            pub fn compute_vertex_to_vertices(&mut self) {
                self.mesh.compute_vertex_to_vertices();
            }

            /// Clear the vertex-to-vertex connectivity
            pub fn clear_vertex_to_vertices(&mut self) {
                self.mesh.clear_vertex_to_vertices();
            }

            /// Compute the volume and vertex volumes
            pub fn compute_volumes(&mut self) {
                self.mesh.compute_volumes();
            }

            /// Clear the volume and vertex volumes
            pub fn clear_volumes(&mut self) {
                self.mesh.clear_volumes();
            }

            /// Compute an octree
            pub fn compute_octree(&mut self) {
                self.mesh.compute_octree();
            }

            /// Clear the octree
            pub fn clear_octree(&mut self) {
                self.mesh.clear_octree();
            }

            /// Split all the elements and faces uniformly
            /// NB: vertex and element data is lost
            #[must_use]
            pub fn split(&self) -> Self {
                Self {
                    mesh: self.mesh.split(),
                }
            }

            /// Add the missing boundary faces and make sure that boundary faces are oriented outwards
            /// If internal faces are present, these are keps
            pub fn add_boundary_faces(&mut self) -> Idx {
                self.mesh.add_boundary_faces()
            }

            /// Write an ascii xdmf file containing the mesh and the vertex and element data
            /// time : a time stamp that will be used to display animations in Paraview
            pub fn write_xdmf(&self, file_name: &str, time: Option<f64>) -> PyResult<()> {
                let res = self.mesh.write_xdmf(file_name, time, None, None);
                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(())
            }

            #[doc = concat!("Get a copy of the mesh coordinates as a numpy array of shape (# of vertices, ", stringify!($dim), ")")]
            pub fn get_coords<'py>(&mut self, py: Python<'py>) -> &'py PyArray2<f64> {
                to_numpy_2d_copy(py, &self.mesh.coords, $dim)
            }

            /// Get a copy of the element connectivity as a numpy array of shape (# of elements, m)
            pub fn get_elems<'py>(&mut self, py: Python<'py>) -> &'py PyArray2<Idx> {
                to_numpy_2d_copy(py, &self.mesh.elems, <$etype as Elem>::N_VERTS as usize)
            }

            /// Get a copy of the element tags as a numpy array of shape (# of elements)
            #[must_use]
            pub fn get_etags<'py>(&self, py: Python<'py>) -> &'py PyArray1<Tag> {
                to_numpy_1d_copy(py, &self.mesh.etags)
            }

            /// Get a copy of the face connectivity as a numpy array of shape (# of faces, m)
            #[must_use]
            pub fn get_faces<'py>(&self, py: Python<'py>) -> &'py PyArray2<Idx> {
                to_numpy_2d_copy(
                    py,
                    &self.mesh.faces,
                    <$etype as Elem>::Face::N_VERTS as usize,
                )
            }

            /// Get a copy of the face tags as a numpy array of shape (# of faces)
            #[must_use]
            pub fn get_ftags<'py>(&self, py: Python<'py>) -> &'py PyArray1<Tag> {
                to_numpy_1d_copy(py, &self.mesh.ftags)
            }

            /// Reorder the vertices, element and faces using a Hilbert SFC
            pub fn reorder_hilbert<'py>(&mut self, py: Python<'py>) -> PyResult<(&'py PyArray1<Idx>, &'py PyArray1<Idx>, &'py PyArray1<Idx>)>{
                let (new_vertex_indices, new_elem_indices, new_face_indices) = self.mesh.reorder_hilbert();
                Ok(
                    (
                        to_numpy_1d(py, new_vertex_indices),
                        to_numpy_1d(py, new_elem_indices),
                        to_numpy_1d(py, new_face_indices)
                    )
                )

            }

            /// Convert a (scalar or vector) field defined at the element centers (P0) to a field defined at the vertices (P1)
            /// using a weighted average.
            pub fn elem_data_to_vertex_data<'py>(
                &mut self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
            ) -> PyResult<&'py PyArray2<f64>> {
                if arr.shape()[0] != self.mesh.n_elems() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }

                let res = self.mesh.elem_data_to_vertex_data(arr.as_slice().unwrap());

                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(to_numpy_2d(py, res.unwrap(), arr.shape()[1]))
            }

            /// Convert a field (scalar or vector) defined at the vertices (P1) to a field defined at the
            /// element centers (P0)
            pub fn vertex_data_to_elem_data<'py>(
                &mut self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
            ) -> PyResult<&'py PyArray2<f64>> {
                if arr.shape()[0] != self.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                let res = self.mesh.vertex_data_to_elem_data(arr.as_slice().unwrap());
                Ok(to_numpy_2d(py, res.unwrap(), arr.shape()[1]))
            }

            /// Interpolate a field (scalar or vector) defined at the vertices (P1) to a different mesh
            pub fn interpolate<'py>(
                &mut self,
                py: Python<'py>,
                other: &Self,
                arr: PyReadonlyArray2<f64>,
            ) -> PyResult<&'py PyArray2<f64>> {
                if arr.shape()[0] != self.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                let res = self.mesh.interpolate(&other.mesh, arr.as_slice().unwrap());
                Ok(to_numpy_2d(py, res.unwrap(), arr.shape()[1]))
            }

            /// Smooth a field defined at the mesh vertices using a 1st order least-square approximation
            pub fn smooth<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
                weight_exp: Option<i32>,
            ) -> PyResult<&'py PyArray2<f64>> {
                if arr.shape()[0] != self.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if arr.shape()[1] != 1 {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let res = self
                    .mesh
                    .smooth(arr.as_slice().unwrap(), weight_exp.unwrap_or(2));
                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(to_numpy_2d(py, res.unwrap(), arr.shape()[1]))
            }

            /// Compute the gradient of a field defined at the mesh vertices using a 1st order least-square approximation
            pub fn compute_gradient<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
                weight_exp: Option<i32>,
            ) -> PyResult<&'py PyArray2<f64>> {
                if arr.shape()[0] != self.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if arr.shape()[1] != 1 {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let res = self
                    .mesh
                    .gradient(arr.as_slice().unwrap(), weight_exp.unwrap_or(2));
                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(to_numpy_2d(
                    py,
                    res.unwrap(),
                    self.mesh.n_comps(FieldType::Vector) as usize,
                ))
            }

            /// Compute the hessian of a field defined at the mesh vertices using a 2nd order least-square approximation
            pub fn compute_hessian<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
                weight_exp: Option<i32>,
            ) -> PyResult<&'py PyArray2<f64>> {
                if arr.shape()[0] != self.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if arr.shape()[1] != 1 {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let res = self
                    .mesh
                    .hessian(arr.as_slice().unwrap(), weight_exp.unwrap_or(2));
                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(to_numpy_2d(
                    py,
                    res.unwrap(),
                    self.mesh.n_comps(FieldType::SymTensor) as usize,
                ))
            }
        }
    };
}

create_mesh!(Mesh33, 3, Tetrahedron);
create_mesh!(Mesh32, 3, Triangle);
create_mesh!(Mesh31, 3, Edge);
create_mesh!(Mesh22, 2, Triangle);
create_mesh!(Mesh21, 2, Edge);

#[pyfunction]
/// Extract the boundary mesh from a Mesh with tetrahedra in 3D
pub fn get_boundary_3d(m: &Mesh33) -> Mesh32 {
    Mesh32 {
        mesh: m.mesh.boundary(),
    }
}

/// Extract the boundary mesh from a Mesh with tetrahedra in 3D
#[pyfunction]
pub fn implied_metric_3d<'py>(py: Python<'py>, m: &Mesh33) -> PyResult<&'py PyArray2<f64>> {
    let m: Vec<_> = (0..m.mesh.n_elems())
        .flat_map(|i| m.mesh.gelem(i).implied_metric().into_iter())
        .collect();
    Ok(to_numpy_2d(py, m, 6))
}

/// Extract the boundary mesh from a Mesh with triangles in 2D
#[pyfunction]
pub fn get_boundary_2d(m: &Mesh22) -> Mesh21 {
    Mesh21 {
        mesh: m.mesh.boundary(),
    }
}

/// Extract the boundary mesh from a Mesh with triangles in 2D
#[pyfunction]
pub fn implied_metric_2d<'py>(py: Python<'py>, m: &Mesh22) -> PyResult<&'py PyArray2<f64>> {
    let m: Vec<_> = (0..m.mesh.n_elems())
        .flat_map(|i| m.mesh.gelem(i).implied_metric().into_iter())
        .collect();
    Ok(to_numpy_2d(py, m, 3))
}

/// Read a solution stored in a .sol(b) file
#[pyfunction]
#[allow(unused_variables)]
pub fn read_solb<'py>(py: Python<'py>, fname: &str) -> PyResult<&'py PyArray2<f64>> {
    #[cfg(feature = "libmeshb-sys")]
    {
        let reader = GmfReader::new(fname);
        if reader.is_invalid() {
            return Err(PyRuntimeError::new_err("Cannot open the file"));
        }

        let (sol, m) = reader.read_solution();
        return Ok(to_numpy_2d(py, sol, m));
    }

    Err(PyRuntimeError::new_err(
        "The meshb interface is not available",
    ))
}

/// Write a solution to a .sol(b) file
#[pyfunction]
#[allow(unused_variables)]
pub fn write_solb(fname: &str, dim: usize, arr: PyReadonlyArray2<f64>) -> PyResult<()> {
    #[cfg(feature = "libmeshb-sys")]
    {
        let mut writer = GmfWriter::new(fname, dim);

        if writer.is_invalid() {
            return Err(PyRuntimeError::new_err("Cannot write the solb file"));
        }

        writer.write_solution(&arr.to_vec().unwrap(), dim, arr.shape()[1]);

        return Ok(());
    }
    Err(PyRuntimeError::new_err(
        "The meshb interface is not available",
    ))
}

macro_rules! create_remesher {
    ($name: ident, $dim: expr, $etype: ident, $metric: ident, $mesh: ident, $geom: ident) => {
        #[doc = concat!("Remesher for a meshes consisting of ", stringify!($etype), " in ", stringify!($dim), "D")]
        #[doc = concat!("using ", stringify!($metric), " as metric and a piecewise linear representation of the geometry")]
        #[pyclass]
        pub struct $name {
            remesher: Remesher<$dim, $etype, $metric, LinearGeometry<$dim, <$etype as Elem>::Face>>,
        }

        #[doc = concat!("Create a remesher from a ", stringify!($mesh), " and a ",stringify!($metric) ," metric defined at the mesh vertices")]
        #[doc = concat!("A piecewise linear representation of the geometry is used, either from the ", stringify!($geom), " given or otherwise from the mesh boundary.")]
        #[pymethods]
        impl $name {
            #[new]
            pub fn new(
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
                geometry: Option<&$geom>,
            ) -> PyResult<Self> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != $metric::N as usize {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice()?;
                let m: Vec<_> = (0..mesh.n_verts())
                    .map(|i| $metric::from_slice(&m, i))
                    .collect();

                let mut gmesh = if let Some(geometry) = geometry {
                    geometry.mesh.clone()
                } else {
                    mesh.mesh.boundary()
                };
                orient_stl(&mesh.mesh, &mut gmesh);
                gmesh.compute_octree();
                let geom = LinearGeometry::new(gmesh).unwrap();

                let remesher = Remesher::new(&mesh.mesh, &m, geom);
                if let Err(res) = remesher {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(Self {
                    remesher: remesher.unwrap(),
                })
            }

            /// Convert a Hessian to the optimal metric using a Lp norm.
            #[classmethod]
            pub fn hessian_to_metric<'py>(
                _cls: &PyType,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
                p: Option<Idx>,
            ) -> PyResult<&'py PyArray2<f64>> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let mut res = Vec::with_capacity(m.shape()[0] * m.shape()[1]);
                let m = m.as_slice().unwrap();

                let exponent = if let Some(p) = p {
                    -2.0 / (2.0 * p as f64 + $dim as f64)
                } else {
                    0.0
                };

                for i_vert in 0..mesh.mesh.n_verts() {
                    let mut m_v = $metric::from_slice(m, i_vert);
                    let scale = f64::powf(m_v.vol(), exponent);
                    if !scale.is_nan() {
                        m_v.scale(scale);
                    }
                    res.extend(m_v.into_iter());
                }

                return Ok(to_numpy_2d(py, res, <$metric as Metric<$dim>>::N));
            }

            /// Scale a metric field to reach the desired (ideal) number of elements using min / max bounds on the cell size
            #[classmethod]
            #[allow(clippy::too_many_arguments)]
            pub fn scale_metric<'py>(
                _cls: &PyType,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
                h_min: f64,
                h_max: f64,
                n_elems: Idx,
                max_iter: Option<Idx>,
            ) -> PyResult<&'py PyArray2<f64>> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice().unwrap();

                let mut m: Vec<_> = (0..mesh.mesh.n_verts())
                    .map(|i| $metric::from_slice(m, i))
                    .collect();
                mesh.mesh
                    .scale_metric(&mut m, h_min, h_max, n_elems, max_iter.unwrap_or(10));
                let m: Vec<_> = m.iter().cloned().flatten().collect();

                return Ok(to_numpy_2d(py, m, <$metric as Metric<$dim>>::N));
            }

            /// Smooth a metric field
            #[classmethod]
            pub fn smooth_metric<'py>(
                _cls: &PyType,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
            ) -> PyResult<&'py PyArray2<f64>> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice().unwrap();
                let m: Vec<_> = (0..mesh.mesh.n_verts())
                    .map(|i| $metric::from_slice(m, i))
                    .collect();
                let m = mesh.mesh.smooth_metric(&m);
                if let Err(m) = m {
                    return Err(PyRuntimeError::new_err(m.to_string()));
                }

                let m: Vec<_> = m.unwrap().iter().cloned().flatten().collect();

                return Ok(to_numpy_2d(py, m, <$metric as Metric<$dim>>::N));
            }

            /// Apply a maximum gradation to a metric field
            #[classmethod]
            pub fn apply_metric_gradation<'py>(
                _cls: &PyType,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
                beta: f64,
                n_iter: Idx,
            ) -> PyResult<&'py PyArray2<f64>> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice().unwrap();
                let mut m: Vec<_> = (0..mesh.mesh.n_verts())
                    .map(|i| $metric::from_slice(m, i))
                    .collect();
                let res = mesh.mesh.apply_metric_gradation(&mut m, beta, n_iter);
                match res {
                    Ok(_) => {
                        let m: Vec<_> = m.iter().cloned().flatten().collect();

                        return Ok(to_numpy_2d(py, m, <$metric as Metric<$dim>>::N));
                    }
                    Err(res) => {
                        return Err(PyRuntimeError::new_err(res.to_string()));
                    }
                }
            }

            /// Convert a metic field defined at the element centers (P0) to a field defined at the vertices (P1)
            /// using a weighted average.
            #[classmethod]
            pub fn elem_data_to_vertex_data_metric<'py>(
                _cls: &PyType,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
            ) -> PyResult<&'py PyArray2<f64>> {
                if m.shape()[0] != mesh.mesh.n_elems() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice().unwrap();
                let res = mesh.mesh.elem_data_to_vertex_data_metric::<$metric>(&m);
                match res {
                    Ok(res) => {
                        return Ok(to_numpy_2d(py, res, <$metric as Metric<$dim>>::N));
                    }
                    Err(res) => {
                        return Err(PyRuntimeError::new_err(res.to_string()));
                    }
                }
            }

            /// Convert a metric field defined at the vertices (P1) to a field defined at the
            /// element centers (P0)
            #[classmethod]
            pub fn vertex_data_to_elem_data_metric<'py>(
                _cls: &PyType,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
            ) -> PyResult<&'py PyArray2<f64>> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice().unwrap();
                let res = mesh.mesh.vertex_data_to_elem_data_metric::<$metric>(&m);
                match res {
                    Ok(res) => {
                        return Ok(to_numpy_2d(py, res, <$metric as Metric<$dim>>::N));
                    }
                    Err(res) => {
                        return Err(PyRuntimeError::new_err(res.to_string()));
                    }
                }
            }

            /// Check that the mesh is valid
            pub fn check(&self) -> PyResult<()> {
                let res = self.remesher.check();
                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }

                Ok(())
            }

            /// Estimate the complexity (ideal number of elements)
            pub fn complexity(&self) -> f64 {
                self.remesher.complexity()
            }

            #[doc = concat!("Get the mesh as a ", stringify!($mesh))]
            #[must_use]
            pub fn to_mesh(&self, only_bdy_faces: Option<bool>) -> $mesh {
                $mesh {
                    mesh: self.remesher.to_mesh(only_bdy_faces.unwrap_or(false)),
                }
            }

            /// Get the number of vertices
            #[must_use]
            pub fn n_verts(&self) -> Idx {
                self.remesher.n_verts()
            }

            /// Get the number of elements
            #[must_use]
            pub fn n_elems(&self) -> Idx {
                self.remesher.n_elems()
            }

            /// Get the number of edges
            #[must_use]
            pub fn n_edges(&self) -> Idx {
                self.remesher.n_edges()
            }

            /// Perform a remeshing iteration
            #[allow(clippy::too_many_arguments)]
            pub fn remesh(
                &mut self,
                num_iter: Option<u32>,
                two_steps: Option<bool>,
                split_constrain_l: Option<f64>,
                split_constrain_q: Option<f64>,
                split_max_iter: Option<u32>,
                collapse_max_iter: Option<u32>,
                collapse_constrain_l: Option<f64>,
                collapse_constrain_q: Option<f64>,
                swap_max_iter: Option<u32>,
                swap_constrain_l: Option<f64>,
                smooth_type: Option<&str>,
                smooth_iter: Option<u32>,
            ) {
                let smooth_type = smooth_type.unwrap_or("laplacian");
                let smooth_type = if smooth_type == "laplacian" {
                    SmoothingType::Laplacian2
                } else if smooth_type == "nlopt" {
                    unreachable!()
                } else {
                    SmoothingType::Avro
                };

                let params = RemesherParams {
                    num_iter: num_iter.unwrap_or(2),
                    two_steps: two_steps.unwrap_or(false),
                    split_constrain_l: split_constrain_l.unwrap_or(1.0),
                    split_constrain_q: split_constrain_q.unwrap_or(0.75),
                    split_max_iter: split_max_iter.unwrap_or(2),
                    collapse_max_iter: collapse_max_iter.unwrap_or(2),
                    collapse_constrain_l: collapse_constrain_l.unwrap_or(1.0),
                    collapse_constrain_q: collapse_constrain_q.unwrap_or(0.75),
                    swap_max_iter: swap_max_iter.unwrap_or(2),
                    swap_constrain_l: swap_constrain_l.unwrap_or(0.5),
                    smooth_type,
                    smooth_iter: smooth_iter.unwrap_or(1),
                };
                self.remesher.remesh(params);
            }

            /// Get the element qualities as a numpy array of size (# or elements)
            #[must_use]
            pub fn qualities<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
                to_numpy_1d(py, self.remesher.qualities())
            }

            /// Get the element lengths (in metric space) as a numpy array of size (# or edges)
            #[must_use]
            pub fn lengths<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
                to_numpy_1d(py, self.remesher.lengths())
            }

            /// Get the infomation about the remeshing steps performed in remesh() as a json string
            pub fn stats_json(&self) -> String {
                self.remesher.stats_json()
            }
        }
    };
}

type IsoMetric2d = IsoMetric<2>;
type IsoMetric3d = IsoMetric<3>;
create_remesher!(Remesher2dIso, 2, Triangle, IsoMetric2d, Mesh22, Mesh21);
create_remesher!(Remesher2dAniso, 2, Triangle, AnisoMetric2d, Mesh22, Mesh21);
create_remesher!(Remesher3dIso, 3, Tetrahedron, IsoMetric3d, Mesh33, Mesh32);
create_remesher!(
    Remesher3dAniso,
    3,
    Tetrahedron,
    AnisoMetric3d,
    Mesh33,
    Mesh32
);

/// Python bindings for pytucanos
#[pymodule]
#[pyo3(name = "_pytucanos")]
fn pytucanos(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<Mesh33>()?;
    m.add_class::<Mesh32>()?;
    m.add_class::<Mesh31>()?;
    m.add_class::<Mesh22>()?;
    m.add_class::<Mesh21>()?;
    m.add_function(wrap_pyfunction!(get_boundary_3d, m)?)?;
    m.add_function(wrap_pyfunction!(get_boundary_2d, m)?)?;
    m.add_function(wrap_pyfunction!(read_solb, m)?)?;
    m.add_function(wrap_pyfunction!(write_solb, m)?)?;
    m.add_function(wrap_pyfunction!(implied_metric_3d, m)?)?;
    m.add_function(wrap_pyfunction!(implied_metric_2d, m)?)?;
    m.add_class::<Remesher2dIso>()?;
    m.add_class::<Remesher2dAniso>()?;
    m.add_class::<Remesher3dIso>()?;
    m.add_class::<Remesher3dAniso>()?;
    #[cfg(not(feature = "libmeshb-sys"))]
    m.add("HAVE_MESHB", false)?;
    #[cfg(feature = "libmeshb-sys")]
    m.add("HAVE_MESHB", true)?;
    Ok(())
}
