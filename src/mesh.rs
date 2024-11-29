use crate::{
    geometry::{LinearGeometry2d, LinearGeometry3d},
    to_numpy_1d, to_numpy_2d,
};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::PyDictMethods,
    pyclass, pymethods,
    types::{PyDict, PyType},
    Bound, PyResult, Python,
};
use std::collections::HashMap;
use tucanos::{
    mesh::{io::read_stl, Edge, Elem, GElem, Point, SimplexMesh, Tetrahedron, Triangle},
    metric::Metric,
    Idx, Tag,
};

macro_rules! create_mesh {
    ($name: ident, $dim: expr, $etype: ident) => {
        #[doc = concat!("Mesh consisting of ", stringify!($etype), " in ", stringify!($dim), "D")]
        #[pyclass]
        pub struct $name {
            pub mesh: SimplexMesh<$dim, $etype>,
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

                let coords = coords.as_slice()?;
                let coords = coords.chunks($dim).map(|p| {
                    let mut vx = Point::<$dim>::zeros();
                    vx.copy_from_slice(p);
                    vx
                }
                ).collect();

                let elems = elems.as_slice()?;
                let elems = elems.chunks($etype::N_VERTS as usize).map(|e| $etype::from_slice(e)).collect();

                let faces = faces.as_slice()?;
                let faces = faces.chunks(<$etype as Elem>::Face::N_VERTS as usize).map(|e| <$etype as Elem>::Face::from_slice(e)).collect();

                Ok(Self {
                    mesh: SimplexMesh::<$dim, $etype>::new(
                        coords,
                        elems,
                        etags.to_vec().unwrap(),
                        faces,
                        ftags.to_vec().unwrap(),
                    ),
                })
            }

            #[doc = concat!("Read a ", stringify!($name), " from a .mesh(b) file")]
            #[classmethod]
            pub fn from_meshb(_cls: &Bound<'_, PyType>, fname: &str) -> PyResult<Self> {
                let res = SimplexMesh::<$dim, $etype>::read_meshb(fname);
                match res {
                    Ok(mesh) => Ok(Self{mesh}),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
            }

            /// Write the mesh to a .mesh(b) file
            pub fn write_meshb(&self, fname: &str) -> PyResult<()> {
                self.mesh.write_meshb(fname).map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }

            /// Write a solution to a .sol(b) file
            pub fn write_solb(&self, fname: &str, arr: PyReadonlyArray2<f64>) -> PyResult<()> {
                self.mesh.write_solb(&arr.to_vec().unwrap(), fname).map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }


            /// Read a solution stored in a .sol(b) file
            #[classmethod]
            pub fn read_solb<'py>(
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                fname: &str
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                use pyo3::exceptions::PyRuntimeError;

                let res = SimplexMesh::<$dim, $etype>::read_solb(fname);
                match res {
                    Ok((sol, m)) => Ok(to_numpy_2d(py, sol, m)),
                    Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
                }
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
                self.mesh.gelems().map(|ge| ge.vol()).sum()
            }

            /// Get the volume of all the elements
            #[must_use]
            pub fn vols<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {

                let res : Vec<_> = self.mesh.gelems().map(|ge| ge.vol()).collect();
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
                self.mesh.compute_edges();
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
            pub fn add_boundary_faces<'py>(&mut self, py: Python<'py>) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
                let (bdy, ifc) = self.mesh.add_boundary_faces();
                let  dict_bdy = PyDict::new_bound(py);
                for (k, v) in bdy.iter() {
                    dict_bdy.set_item(k, v)?;
                }
                let  dict_ifc = PyDict::new_bound(py);
                for (k, v) in ifc.iter() {
                    dict_ifc.set_item(k, to_numpy_1d(py, v.to_vec()))?;
                }

                Ok((dict_bdy, dict_ifc))

            }

            /// Write a vtk file containing the mesh
            #[pyo3(signature = (file_name, vert_data=None, elem_data=None))]
            pub fn write_vtk(&self,
                file_name: &str,
                vert_data : Option<HashMap<String, PyReadonlyArray2<f64>>>,
                elem_data : Option<HashMap<String, PyReadonlyArray2<f64>>> ) -> PyResult<()> {

                let mut vdata = HashMap::new();
                if let Some(data) = vert_data.as_ref() {
                    for (name, arr) in data.iter() {
                        vdata.insert(name.to_string(), arr.as_slice().unwrap());
                    }
                }

                let mut edata = HashMap::new();
                if let Some(data) = elem_data.as_ref() {
                    for (name, arr) in data.iter() {
                        edata.insert(name.to_string(), arr.as_slice().unwrap());
                    }
                }

                let res = self.mesh.write_vtk(file_name, Some(vdata), Some(edata));

                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(())
            }

            /// Write a vtk file containing the boundary
            pub fn write_boundary_vtk(&self, file_name: &str) -> PyResult<()> {
                let res = self.mesh.boundary().0.write_vtk(file_name, None, None);
                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(())
            }

            #[doc = concat!("Get a copy of the mesh coordinates as a numpy array of shape (# of vertices, ", stringify!($dim), ")")]
            pub fn get_coords<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
                let mut coords = Vec::with_capacity(self.mesh.n_verts() as usize * $dim);
                for v in self.mesh.verts() {
                    coords.extend(v.iter().copied());
                }
                to_numpy_2d(py, coords, $dim)
            }

            /// Get a copy of the element connectivity as a numpy array of shape (# of elements, m)
            pub fn get_elems<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<Idx>> {
                let elems = self.mesh.elems().flatten().collect();
                to_numpy_2d(py, elems, <$etype as Elem>::N_VERTS as usize)
            }

            /// Get a copy of the element tags as a numpy array of shape (# of elements)
            #[must_use]
            pub fn get_etags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Tag>> {
                let etags = self.mesh.etags().collect();
                to_numpy_1d(py, etags)
            }

            /// Get a copy of the face connectivity as a numpy array of shape (# of faces, m)
            #[must_use]
            pub fn get_faces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<Idx>> {
                let faces = self.mesh.faces().flatten().collect();
                to_numpy_2d(
                    py,
                    faces,
                    <$etype as Elem>::Face::N_VERTS as usize,
                )
            }

            /// Get a copy of the face tags as a numpy array of shape (# of faces)
            #[must_use]
            pub fn get_ftags<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Tag>> {
                let ftags = self.mesh.ftags().collect();
                to_numpy_1d(py, ftags)
            }

            /// Reorder the vertices, element and faces using a Hilbert SFC
            pub fn reorder_hilbert<'py>(&mut self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray1<Idx>>, Bound<'py, PyArray1<Idx>>, Bound<'py, PyArray1<Idx>>)>{
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
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
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
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if arr.shape()[0] != self.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                let res = self.mesh.vertex_data_to_elem_data(arr.as_slice().unwrap());
                Ok(to_numpy_2d(py, res.unwrap(), arr.shape()[1]))
            }

            /// Interpolate a field (scalar or vector) defined at the vertices (P1) to a different mesh using linear interpolation
            #[pyo3(signature = (other, arr, tol=None))]
            pub fn interpolate_linear<'py>(
                &mut self,
                py: Python<'py>,
                other: &Self,
                arr: PyReadonlyArray2<f64>,
                tol: Option<f64>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if arr.shape()[0] != self.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                let tree = self.mesh.compute_elem_tree();
                let res = self.mesh.interpolate_linear(&tree, &other.mesh, arr.as_slice().unwrap(), tol);
                Ok(to_numpy_2d(py, res.unwrap(), arr.shape()[1]))
            }

            /// Interpolate a field (scalar or vector) defined at the vertices (P1) to a different mesh using nearest neighbor interpolation
            pub fn interpolate_nearest<'py>(
                &mut self,
                py: Python<'py>,
                other: &Self,
                arr: PyReadonlyArray2<f64>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if arr.shape()[0] != self.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                let tree = self.mesh.compute_vert_tree();
                let res = self.mesh.interpolate_nearest(&tree, &other.mesh, arr.as_slice().unwrap());
                Ok(to_numpy_2d(py, res.unwrap(), arr.shape()[1]))
            }

            /// Smooth a field defined at the mesh vertices using a 1st order least-square approximation
            #[pyo3(signature = (arr, weight_exp=None))]
            pub fn smooth<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
                weight_exp: Option<i32>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
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
            #[pyo3(signature = (arr, weight_exp=None))]
            pub fn compute_gradient<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
                weight_exp: Option<i32>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
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
                    $dim,
                ))
            }

            /// Compute the hessian of a field defined at the mesh vertices using a 2nd order least-square approximation
            /// if `weight_exp` is `None`, the vertex has a weight 10, its first order neighbors have
            /// a weight 1 and the 2nd order neighbors (if used) have a weight of 0.1
            #[pyo3(signature = (arr, weight_exp=None, use_second_order_neighbors=None))]
            pub fn compute_hessian<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
                weight_exp: Option<i32>,
                use_second_order_neighbors: Option<bool>,
            ) -> PyResult<Bound<'py,  PyArray2<f64>>> {
                if arr.shape()[0] != self.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if arr.shape()[1] != 1 {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let res = self
                    .mesh
                    .hessian(arr.as_slice().unwrap(), weight_exp, use_second_order_neighbors.unwrap_or(true));
                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(to_numpy_2d(
                    py,
                    res.unwrap(),
                    $dim * ($dim +1 ) / 2,
                ))
            }

            /// Compute the hessian of a field defined at the mesh vertices using L2 projection
            pub fn compute_hessian_l2proj<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
            ) -> PyResult<Bound<'py,  PyArray2<f64>>> {
                if arr.shape()[0] != self.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if arr.shape()[1] != 1 {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let grad = self
                    .mesh
                    .gradient_l2proj(arr.as_slice().unwrap());
                if let Err(res) = grad {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }

                let res = self.mesh.hessian_l2proj(&grad.unwrap());
                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }

                Ok(to_numpy_2d(
                    py,
                    res.unwrap(),
                    $dim * ($dim +1 ) / 2,
                ))
            }

            /// Check that the mesh is valid
            ///  - all elements have a >0 volume
            ///  - all boundary faces are tagged
            ///  - all the faces between different element tags are tagged
            ///  - no other face is tagged
            pub fn check(&self) -> PyResult<()> {
                self.mesh.check().map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }

            /// Compute the topology
            pub fn compute_topology(&mut self) {
                self.mesh.compute_topology();
            }

            /// Clear the topology
            pub fn clear_topology(&mut self) {
                self.mesh.clear_topology();
            }

            /// Automatically tag the elements based on a feature angle
            pub fn autotag<'py>(&mut self, py: Python<'py>, angle_deg: f64) -> PyResult<Bound<'py, PyDict>> {
                let res = self.mesh.autotag(angle_deg);
                if let Err(res) = res {
                     Err(PyRuntimeError::new_err(res.to_string()))
                } else {
                    let dict = PyDict::new_bound(py);
                    for (k, v) in res.unwrap().iter() {
                        dict.set_item(k, to_numpy_1d(py, v.to_vec()))?;
                    }
                    Ok(dict)
                }
            }

            /// Automatically tag the faces based on a feature angle
            pub fn autotag_bdy<'py>(&mut self, py: Python<'py>, angle_deg: f64) -> PyResult<Bound<'py, PyDict>> {
                let res = self.mesh.autotag_bdy(angle_deg);
                if let Err(res) = res {
                     Err(PyRuntimeError::new_err(res.to_string()))
                } else {
                    let dict = PyDict::new_bound(py);
                    for (k, v) in res.unwrap().iter() {
                        dict.set_item(k, to_numpy_1d(py, v.to_vec()))?;
                    }
                    Ok(dict)
                }
            }
        }
    };
}

create_mesh!(Mesh33, 3, Tetrahedron);
create_mesh!(Mesh32, 3, Triangle);
create_mesh!(Mesh31, 3, Edge);
create_mesh!(Mesh22, 2, Triangle);
create_mesh!(Mesh21, 2, Edge);

#[pymethods]
impl Mesh33 {
    /// Create a Mesh33 from basic elements
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    #[classmethod]
    #[pyo3(signature = (coords, hexs=None, hex_tags=None, pris=None, pri_tags=None, pyrs=None, pyr_tags=None, tets=None, tet_tags=None, quas=None, qua_tags=None, tris=None, tri_tags=None))]
    pub fn from_basic_elems(
        _cls: &Bound<'_, PyType>,
        coords: PyReadonlyArray2<f64>,
        hexs: Option<PyReadonlyArray2<Idx>>,
        hex_tags: Option<PyReadonlyArray1<Tag>>,
        pris: Option<PyReadonlyArray2<Idx>>,
        pri_tags: Option<PyReadonlyArray1<Tag>>,
        pyrs: Option<PyReadonlyArray2<Idx>>,
        pyr_tags: Option<PyReadonlyArray1<Tag>>,
        tets: Option<PyReadonlyArray2<Idx>>,
        tet_tags: Option<PyReadonlyArray1<Tag>>,
        quas: Option<PyReadonlyArray2<Idx>>,
        qua_tags: Option<PyReadonlyArray1<Tag>>,
        tris: Option<PyReadonlyArray2<Idx>>,
        tri_tags: Option<PyReadonlyArray1<Tag>>,
    ) -> PyResult<Self> {
        let mut res = SimplexMesh::<3, Tetrahedron>::empty();

        if coords.shape()[1] != 3 {
            return Err(PyValueError::new_err("Invalid dimension 1 for coords"));
        }
        let coords = coords.as_slice()?.chunks(3);
        res.add_verts(coords);

        if let Some(hexs) = hexs {
            if hexs.shape()[1] != 8 {
                return Err(PyValueError::new_err("Invalid dimension 1 for hexs"));
            }
            if let Some(hex_tags) = hex_tags {
                if hexs.shape()[0] != hex_tags.shape()[0] {
                    return Err(PyValueError::new_err(
                        "Invalid dimension 0 for hexs / hex_tags",
                    ));
                }
                res.add_hexs(
                    hexs.as_slice()?.chunks(8),
                    hex_tags.as_slice()?.iter().copied(),
                );
            } else {
                res.add_hexs(hexs.as_slice()?.chunks(8), (0..hexs.shape()[0]).map(|_| 1));
            };
        }

        if let Some(pris) = pris {
            if pris.shape()[1] != 6 {
                return Err(PyValueError::new_err("Invalid dimension 1 for pris"));
            }
            if let Some(pri_tags) = pri_tags {
                if pris.shape()[0] != pri_tags.shape()[0] {
                    return Err(PyValueError::new_err(
                        "Invalid dimension 0 for pris / pri_tags",
                    ));
                }
                res.add_pris(
                    pris.as_slice()?.chunks(6),
                    pri_tags.as_slice()?.iter().copied(),
                );
            } else {
                res.add_pris(pris.as_slice()?.chunks(6), (0..pris.shape()[0]).map(|_| 1));
            };
        }

        if let Some(pyrs) = pyrs {
            if pyrs.shape()[1] != 5 {
                return Err(PyValueError::new_err("Invalid dimension 1 for pyrs"));
            }
            if let Some(pyr_tags) = pyr_tags {
                if pyrs.shape()[0] != pyr_tags.shape()[0] {
                    return Err(PyValueError::new_err(
                        "Invalid dimension 0 for pyrs / pyr_tags",
                    ));
                }
                res.add_pyrs(
                    pyrs.as_slice()?.chunks(5),
                    pyr_tags.as_slice()?.iter().copied(),
                );
            } else {
                res.add_pyrs(pyrs.as_slice()?.chunks(5), (0..pyrs.shape()[0]).map(|_| 1));
            };
        }

        if let Some(tets) = tets {
            if tets.shape()[1] != 4 {
                return Err(PyValueError::new_err("Invalid dimension 1 for tets"));
            }
            if let Some(tet_tags) = tet_tags {
                if tets.shape()[0] != tet_tags.shape()[0] {
                    return Err(PyValueError::new_err(
                        "Invalid dimension 0 for tets / tet_tags",
                    ));
                }
                res.add_tets(
                    tets.as_slice()?.chunks(4),
                    tet_tags.as_slice()?.iter().copied(),
                );
            } else {
                res.add_tets(tets.as_slice()?.chunks(4), (0..tets.shape()[0]).map(|_| 1));
            };
        }

        if let Some(quas) = quas {
            if quas.shape()[1] != 4 {
                return Err(PyValueError::new_err("Invalid dimension 1 for quas"));
            }
            if let Some(qua_tags) = qua_tags {
                if quas.shape()[0] != qua_tags.shape()[0] {
                    return Err(PyValueError::new_err(
                        "Invalid dimension 0 for quas / qua_tags",
                    ));
                }
                res.add_quas(
                    quas.as_slice()?.chunks(4),
                    qua_tags.as_slice()?.iter().copied(),
                );
            } else {
                res.add_quas(quas.as_slice()?.chunks(4), (0..quas.shape()[0]).map(|_| 1));
            };
        }

        if let Some(tris) = tris {
            if tris.shape()[1] != 3 {
                return Err(PyValueError::new_err("Invalid dimension 1 for tris"));
            }
            if let Some(tri_tags) = tri_tags {
                if tris.shape()[0] != tri_tags.shape()[0] {
                    return Err(PyValueError::new_err(
                        "Invalid dimension 0 for tris / tri_tags",
                    ));
                }
                res.add_tris(
                    tris.as_slice()?.chunks(3),
                    tri_tags.as_slice()?.iter().copied(),
                );
            } else {
                res.add_tris(tris.as_slice()?.chunks(3), (0..tris.shape()[0]).map(|_| 1));
            };
        }

        Ok(Self { mesh: res })
    }

    /// Extract the boundary faces into a Mesh, and return the indices of the vertices in the
    /// parent mesh
    #[must_use]
    pub fn boundary<'py>(&self, py: Python<'py>) -> (Mesh32, Bound<'py, PyArray1<Idx>>) {
        let (bdy, ids) = self.mesh.boundary();
        (Mesh32 { mesh: bdy }, to_numpy_1d(py, ids))
    }

    pub fn implied_metric<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let res = self.mesh.implied_metric();

        if let Err(res) = res {
            return Err(PyRuntimeError::new_err(res.to_string()));
        }

        let m: Vec<f64> = res.unwrap().iter().flat_map(|m| m.into_iter()).collect();
        Ok(to_numpy_2d(py, m, 6))
    }

    /// Get a metric defined on all the mesh vertices such that
    ///  - for boundary vertices, the principal directions are aligned with the principal curvature directions
    ///    and the sizes to curvature radius ratio is r_h
    ///  - the metric is entended into the volume with gradation beta
    ///  - if an implied metric is provided, the result is limited to (1/step,step) times the implied metric
    ///  - if a normal size array is not provided, the minimum of the tangential sizes is used.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (geom, r_h, beta, h_min=None, h_n=None, h_n_tags=None))]
    pub fn curvature_metric<'py>(
        &self,
        py: Python<'py>,
        geom: &LinearGeometry3d,
        r_h: f64,
        beta: f64,
        h_min: Option<f64>,
        h_n: Option<PyReadonlyArray1<f64>>,
        h_n_tags: Option<PyReadonlyArray1<Tag>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let res = if let Some(h_n) = h_n {
            let h_n = h_n.as_slice()?;
            if h_n_tags.is_none() {
                return Err(PyRuntimeError::new_err("h_n_tags not given"));
            }
            let h_n_tags = h_n_tags.unwrap();
            let h_n_tags = h_n_tags.as_slice()?;
            self.mesh
                .curvature_metric(&geom.geom, r_h, beta, Some(h_n), Some(h_n_tags))
        } else {
            self.mesh
                .curvature_metric(&geom.geom, r_h, beta, None, None)
        };

        if let Err(res) = res {
            return Err(PyRuntimeError::new_err(res.to_string()));
        }
        let mut m = res.unwrap();

        if let Some(h_min) = h_min {
            m.iter_mut()
                .for_each(|x| x.scale_with_bounds(1.0, h_min, f64::MAX));
        }

        let m: Vec<f64> = m.iter().flat_map(|m| m.into_iter()).collect();

        Ok(to_numpy_2d(py, m, 6))
    }
}

#[pymethods]
impl Mesh32 {
    /// Create a Mesh32 from basic elements
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (coords, quas=None, qua_tags=None, tris=None, tri_tags=None, edgs=None, edg_tags=None))]
    pub fn from_basic_elems(
        _cls: &Bound<'_, PyType>,
        coords: PyReadonlyArray2<f64>,
        quas: Option<PyReadonlyArray2<Idx>>,
        qua_tags: Option<PyReadonlyArray1<Tag>>,
        tris: Option<PyReadonlyArray2<Idx>>,
        tri_tags: Option<PyReadonlyArray1<Tag>>,
        edgs: Option<PyReadonlyArray2<Idx>>,
        edg_tags: Option<PyReadonlyArray1<Tag>>,
    ) -> PyResult<Self> {
        let mut res = SimplexMesh::<3, Triangle>::empty();

        if coords.shape()[1] != 3 {
            return Err(PyValueError::new_err("Invalid dimension 1 for coords"));
        }
        let coords = coords.as_slice()?.chunks(3);
        res.add_verts(coords);

        if let Some(quas) = quas {
            if quas.shape()[1] != 4 {
                return Err(PyValueError::new_err("Invalid dimension 1 for quas"));
            }
            if let Some(qua_tags) = qua_tags {
                if quas.shape()[0] != qua_tags.shape()[0] {
                    return Err(PyValueError::new_err(
                        "Invalid dimension 0 for quas / qua_tags",
                    ));
                }
                res.add_quas(
                    quas.as_slice()?.chunks(4),
                    qua_tags.as_slice()?.iter().copied(),
                );
            } else {
                res.add_quas(quas.as_slice()?.chunks(4), (0..quas.shape()[0]).map(|_| 1));
            };
        }

        if let Some(tris) = tris {
            if tris.shape()[1] != 3 {
                return Err(PyValueError::new_err("Invalid dimension 1 for tris"));
            }
            if let Some(tri_tags) = tri_tags {
                if tris.shape()[0] != tri_tags.shape()[0] {
                    return Err(PyValueError::new_err(
                        "Invalid dimension 0 for tris / tri_tags",
                    ));
                }
                res.add_tris(
                    tris.as_slice()?.chunks(3),
                    tri_tags.as_slice()?.iter().copied(),
                );
            } else {
                res.add_tris(tris.as_slice()?.chunks(3), (0..tris.shape()[0]).map(|_| 1));
            };
        }

        if let Some(edgs) = edgs {
            if edgs.shape()[1] != 2 {
                return Err(PyValueError::new_err("Invalid dimension 1 for edgs"));
            }
            if let Some(edg_tags) = edg_tags {
                if edgs.shape()[0] != edg_tags.shape()[0] {
                    return Err(PyValueError::new_err(
                        "Invalid dimension 0 for edgs / edg_tags",
                    ));
                }
                res.add_edgs(
                    edgs.as_slice()?.chunks(2),
                    edg_tags.as_slice()?.iter().copied(),
                );
            } else {
                res.add_edgs(edgs.as_slice()?.chunks(2), (0..edgs.shape()[0]).map(|_| 1));
            };
        }

        Ok(Self { mesh: res })
    }

    #[doc = concat!("Read a ", stringify!($name), " from a .stl file")]
    #[classmethod]
    pub fn from_stl(_cls: &Bound<'_, PyType>, fname: &str) -> Self {
        Self {
            mesh: read_stl(fname),
        }
    }

    /// Reset the face tags of other to match those in self
    pub fn transfer_tags_face(&self, other: &mut Mesh33) -> PyResult<()> {
        let tree = self.mesh.compute_elem_tree();
        self.mesh
            .transfer_tags(&tree, &mut other.mesh)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Reset the element tags of other to match those in self
    pub fn transfer_tags_elem(&self, other: &mut Self) -> PyResult<()> {
        let tree = self.mesh.compute_elem_tree();
        self.mesh
            .transfer_tags(&tree, &mut other.mesh)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pymethods]
impl Mesh22 {
    /// Create a Mesh22 from basic elements
    #[allow(clippy::too_many_arguments)]
    #[classmethod]
    #[pyo3(signature = (coords, quas=None, qua_tags=None, tris=None, tri_tags=None, edgs=None, edg_tags=None))]
    pub fn from_basic_elems(
        _cls: &Bound<'_, PyType>,
        coords: PyReadonlyArray2<f64>,
        quas: Option<PyReadonlyArray2<Idx>>,
        qua_tags: Option<PyReadonlyArray1<Tag>>,
        tris: Option<PyReadonlyArray2<Idx>>,
        tri_tags: Option<PyReadonlyArray1<Tag>>,
        edgs: Option<PyReadonlyArray2<Idx>>,
        edg_tags: Option<PyReadonlyArray1<Tag>>,
    ) -> PyResult<Self> {
        let mut res = SimplexMesh::<2, Triangle>::empty();

        if coords.shape()[1] != 2 {
            return Err(PyValueError::new_err("Invalid dimension 1 for coords"));
        }
        let coords = coords.as_slice()?.chunks(2);
        res.add_verts(coords);

        if let Some(quas) = quas {
            if quas.shape()[1] != 4 {
                return Err(PyValueError::new_err("Invalid dimension 1 for quas"));
            }
            if let Some(qua_tags) = qua_tags {
                if quas.shape()[0] != qua_tags.shape()[0] {
                    return Err(PyValueError::new_err(
                        "Invalid dimension 0 for quas / qua_tags",
                    ));
                }
                res.add_quas(
                    quas.as_slice()?.chunks(4),
                    qua_tags.as_slice()?.iter().copied(),
                );
            } else {
                res.add_quas(quas.as_slice()?.chunks(4), (0..quas.shape()[0]).map(|_| 1));
            };
        }

        if let Some(tris) = tris {
            if tris.shape()[1] != 3 {
                return Err(PyValueError::new_err("Invalid dimension 1 for tris"));
            }
            if let Some(tri_tags) = tri_tags {
                if tris.shape()[0] != tri_tags.shape()[0] {
                    return Err(PyValueError::new_err(
                        "Invalid dimension 0 for tris / tri_tags",
                    ));
                }
                res.add_tris(
                    tris.as_slice()?.chunks(3),
                    tri_tags.as_slice()?.iter().copied(),
                );
            } else {
                res.add_tris(tris.as_slice()?.chunks(3), (0..tris.shape()[0]).map(|_| 1));
            };
        }

        if let Some(edgs) = edgs {
            if edgs.shape()[1] != 2 {
                return Err(PyValueError::new_err("Invalid dimension 1 for edgs"));
            }
            if let Some(edg_tags) = edg_tags {
                if edgs.shape()[0] != edg_tags.shape()[0] {
                    return Err(PyValueError::new_err(
                        "Invalid dimension 0 for edgs / edg_tags",
                    ));
                }
                res.add_edgs(
                    edgs.as_slice()?.chunks(2),
                    edg_tags.as_slice()?.iter().copied(),
                );
            } else {
                res.add_edgs(edgs.as_slice()?.chunks(2), (0..edgs.shape()[0]).map(|_| 1));
            };
        }

        Ok(Self { mesh: res })
    }
    /// Extract the boundary faces into a Mesh, and return the indices of the vertices in the
    /// parent mesh
    #[must_use]
    pub fn boundary<'py>(&self, py: Python<'py>) -> (Mesh21, Bound<'py, PyArray1<Idx>>) {
        let (bdy, ids) = self.mesh.boundary();
        (Mesh21 { mesh: bdy }, to_numpy_1d(py, ids))
    }

    pub fn implied_metric<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let res = self.mesh.implied_metric();

        if let Err(res) = res {
            return Err(PyRuntimeError::new_err(res.to_string()));
        }

        let m: Vec<f64> = res.unwrap().iter().flat_map(|m| m.into_iter()).collect();
        Ok(to_numpy_2d(py, m, 3))
    }

    /// Get a metric defined on all the mesh vertices such that
    ///  - for boundary vertices, the principal directions are aligned with the principal curvature directions
    ///    and the sizes to curvature radius ratio is r_h
    ///  - the metric is entended into the volume with gradation beta
    ///  - if a normal size array is not provided, the minimum of the tangential sizes is used.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (geom, r_h, beta, h_min=None, h_n=None, h_n_tags=None))]
    pub fn curvature_metric<'py>(
        &self,
        py: Python<'py>,
        geom: &LinearGeometry2d,
        r_h: f64,
        beta: f64,
        h_min: Option<f64>,
        h_n: Option<PyReadonlyArray1<f64>>,
        h_n_tags: Option<PyReadonlyArray1<Tag>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let res = if let Some(h_n) = h_n {
            let h_n = h_n.as_slice()?;
            if h_n_tags.is_none() {
                return Err(PyRuntimeError::new_err("h_n_tags not given"));
            }
            let h_n_tags = h_n_tags.unwrap();
            let h_n_tags = h_n_tags.as_slice()?;
            self.mesh
                .curvature_metric(&geom.geom, r_h, beta, Some(h_n), Some(h_n_tags))
        } else {
            self.mesh
                .curvature_metric(&geom.geom, r_h, beta, None, None)
        };

        if let Err(res) = res {
            return Err(PyRuntimeError::new_err(res.to_string()));
        }
        let mut m = res.unwrap();

        if let Some(h_min) = h_min {
            m.iter_mut()
                .for_each(|x| x.scale_with_bounds(1.0, h_min, f64::MAX));
        }

        let m: Vec<f64> = m.iter().flat_map(|m| m.into_iter()).collect();

        Ok(to_numpy_2d(py, m, 3))
    }
}

#[pymethods]
impl Mesh21 {
    /// Reset the face tags of other to match those in self
    pub fn transfer_tags_face(&self, other: &mut Mesh22) -> PyResult<()> {
        let tree = self.mesh.compute_elem_tree();
        self.mesh
            .transfer_tags(&tree, &mut other.mesh)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Reset the element tags of other to match those in self
    pub fn transfer_tags_elem(&self, other: &mut Self) -> PyResult<()> {
        let tree = self.mesh.compute_elem_tree();
        self.mesh
            .transfer_tags(&tree, &mut other.mesh)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}
