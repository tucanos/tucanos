use crate::{
    geometry::{LinearGeometry2d, LinearGeometry3d},
    to_numpy_1d, to_numpy_2d,
};
use numpy::{
    PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::{
    Bound, PyResult, Python,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::PyDictMethods,
    pyclass, pymethods,
    types::{PyDict, PyType},
};
use pytmesh::{PyPartitionerType, impl_mesh};
#[cfg(feature = "metis")]
use tmesh::mesh::partition::{MetisKWay, MetisPartitioner, MetisRecursive};
use tmesh::{
    Vertex,
    interpolate::{InterpolationMethod, Interpolator},
    mesh::{
        Mesh,
        partition::{HilbertPartitioner, KMeansPartitioner2d, KMeansPartitioner3d, RCMPartitioner},
    },
    spatialindex::ObjectIndex,
};
use tucanos::{
    Idx, Tag,
    mesh::{Edge, GElem, SimplexMesh, Tetrahedron, Triangle},
    metric::Metric,
};

macro_rules! create_mesh {
    ($name: ident, $dim: expr, $etype: ident) => {
        #[doc = concat!("Mesh consisting of ", stringify!($etype), " in ", stringify!($dim), "D")]
        #[pyclass]
        pub struct $name(pub(crate) SimplexMesh<$dim, $etype>);

        #[pymethods]
        impl $name {
            /// Get the volume of the mesh
            #[must_use]
            pub fn vol(&self) -> f64 {
                self.0.gelems().map(|ge| ge.vol()).sum()
            }

            /// Get the volume of all the elements
            #[must_use]
            pub fn vols<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                let res: Vec<_> = self.0.gelems().map(|ge| ge.vol()).collect();
                to_numpy_1d(py, res)
            }

            /// Add the missing boundary faces and make sure that boundary faces are oriented
            /// outwards.
            /// If internal faces are present, these are keps
            pub fn add_boundary_faces<'py>(
                &mut self,
                py: Python<'py>,
            ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
                let (bdy, ifc) = self.0.add_boundary_faces();
                let dict_bdy = PyDict::new(py);
                for (k, v) in bdy.iter() {
                    dict_bdy.set_item(k, v)?;
                }
                let dict_ifc = PyDict::new(py);
                for (k, v) in ifc.iter() {
                    dict_ifc.set_item(k, to_numpy_1d(py, v.to_vec()))?;
                }

                Ok((dict_bdy, dict_ifc))
            }

            /// Compute the vertex-to-element connectivity
            pub fn compute_vertex_to_elems(&mut self) {
                self.0.compute_vertex_to_elems();
            }

            /// Clear the vertex-to-element connectivity
            pub fn clear_vertex_to_elems(&mut self) {
                self.0.clear_vertex_to_elems();
            }

            /// Compute the face-to-element connectivity
            pub fn compute_face_to_elems(&mut self) {
                self.0.compute_face_to_elems();
            }

            /// Clear the face-to-element connectivity
            pub fn clear_face_to_elems(&mut self) {
                self.0.clear_face_to_elems();
            }

            /// Compute the element-to-element connectivity
            /// face-to-element connectivity is computed if not available
            pub fn compute_elem_to_elems(&mut self) {
                self.0.compute_elem_to_elems();
            }

            /// Clear the element-to-element connectivity
            pub fn clear_elem_to_elems(&mut self) {
                self.0.clear_elem_to_elems();
            }

            /// Compute the edges
            pub fn compute_edges(&mut self) {
                self.0.compute_edges();
            }

            /// Clear the edges
            pub fn clear_edges(&mut self) {
                self.0.clear_edges()
            }

            /// Compute the vertex-to-vertex connectivity
            /// Edges are computed if not available
            pub fn compute_vertex_to_vertices(&mut self) {
                self.0.compute_vertex_to_vertices();
            }

            /// Clear the vertex-to-vertex connectivity
            pub fn clear_vertex_to_vertices(&mut self) {
                self.0.clear_vertex_to_vertices();
            }

            /// Compute the volume and vertex volumes
            pub fn compute_volumes(&mut self) {
                self.0.compute_volumes();
            }

            /// Clear the volume and vertex volumes
            pub fn clear_volumes(&mut self) {
                self.0.clear_volumes();
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

                let res = self.0.elem_data_to_vertex_data(arr.as_slice().unwrap());

                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(to_numpy_2d(py, res.unwrap(), arr.shape()[1]))
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
                Ok(to_numpy_2d(py, res.unwrap(), arr.shape()[1]))
            }

            /// Smooth a field defined at the mesh vertices using a 1st order least-square
            /// approximation
            #[pyo3(signature = (arr, weight_exp=None))]
            pub fn smooth<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
                weight_exp: Option<i32>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if arr.shape()[0] != self.0.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if arr.shape()[1] != 1 {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let res = self
                    .0
                    .smooth(arr.as_slice().unwrap(), weight_exp.unwrap_or(2));
                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(to_numpy_2d(py, res.unwrap(), arr.shape()[1]))
            }

            /// Compute the gradient of a field defined at the mesh vertices using a 1st order
            /// least-square approximation
            #[pyo3(signature = (arr, weight_exp=None))]
            pub fn compute_gradient<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
                weight_exp: Option<i32>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if arr.shape()[0] != self.0.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if arr.shape()[1] != 1 {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let res = self
                    .0
                    .gradient(arr.as_slice().unwrap(), weight_exp.unwrap_or(2));
                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(to_numpy_2d(py, res.unwrap(), $dim))
            }

            /// Compute the hessian of a field defined at the mesh vertices using a 2nd order
            /// least-square approximation
            /// if `weight_exp` is `None`, the vertex has a weight 10, its first order neighbors
            /// have a weight 1 and the 2nd order neighbors (if used) have a weight of 0.1
            #[pyo3(signature = (arr, weight_exp=None, use_second_order_neighbors=None))]
            pub fn compute_hessian<'py>(
                &self,
                py: Python<'py>,
                arr: PyReadonlyArray2<f64>,
                weight_exp: Option<i32>,
                use_second_order_neighbors: Option<bool>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if arr.shape()[0] != self.0.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if arr.shape()[1] != 1 {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let res = self.0.hessian(
                    arr.as_slice().unwrap(),
                    weight_exp,
                    use_second_order_neighbors.unwrap_or(true),
                );
                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(to_numpy_2d(py, res.unwrap(), $dim * ($dim + 1) / 2))
            }

            /// Compute the hessian of a field defined at the mesh vertices using L2 projection
            pub fn compute_hessian_l2proj<'py>(
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

                let grad = self.0.gradient_l2proj(arr.as_slice().unwrap());
                if let Err(res) = grad {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }

                let res = self.0.hessian_l2proj(&grad.unwrap());
                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }

                Ok(to_numpy_2d(py, res.unwrap(), $dim * ($dim + 1) / 2))
            }

            /// Compute the topology
            pub fn compute_topology(&mut self) {
                self.0.compute_topology();
            }

            /// Clear the topology
            pub fn clear_topology(&mut self) {
                self.0.clear_topology();
            }

            /// Automatically tag the elements based on a feature angle
            pub fn autotag<'py>(
                &mut self,
                py: Python<'py>,
                angle_deg: f64,
            ) -> PyResult<Bound<'py, PyDict>> {
                let res = self.0.autotag(angle_deg);
                if let Err(res) = res {
                    Err(PyRuntimeError::new_err(res.to_string()))
                } else {
                    let dict = PyDict::new(py);
                    for (k, v) in res.unwrap().iter() {
                        dict.set_item(k, to_numpy_1d(py, v.to_vec()))?;
                    }
                    Ok(dict)
                }
            }

            /// Automatically tag the faces based on a feature angle
            pub fn autotag_bdy<'py>(
                &mut self,
                py: Python<'py>,
                angle_deg: f64,
            ) -> PyResult<Bound<'py, PyDict>> {
                let res = self.0.autotag_bdy(angle_deg);
                if let Err(res) = res {
                    Err(PyRuntimeError::new_err(res.to_string()))
                } else {
                    let dict = PyDict::new(py);
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
type Mesh1 = SimplexMesh<3, Tetrahedron>;
impl_mesh!(Mesh33, Mesh1, 3, 4, 3);
type Mesh2 = SimplexMesh<3, Triangle>;
impl_mesh!(Mesh32, Mesh2, 3, 3, 2);
type Mesh3 = SimplexMesh<3, Edge>;
impl_mesh!(Mesh31, Mesh3, 3, 2, 1);
type Mesh4 = SimplexMesh<2, Triangle>;
impl_mesh!(Mesh22, Mesh4, 2, 3, 2);
type Mesh5 = SimplexMesh<2, Edge>;
impl_mesh!(Mesh21, Mesh5, 2, 2, 1);

#[pymethods]
impl Mesh33 {
    /// Extract the boundary faces into a Mesh, and return the indices of the vertices in the
    /// parent mesh
    #[must_use]
    pub fn boundary<'py>(&self, py: Python<'py>) -> (Mesh32, Bound<'py, PyArray1<Idx>>) {
        let (bdy, ids) = self.0.boundary();
        (Mesh32(bdy), to_numpy_1d(py, ids))
    }

    pub fn implied_metric<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let res = self.0.implied_metric();

        if let Err(res) = res {
            return Err(PyRuntimeError::new_err(res.to_string()));
        }

        let m: Vec<f64> = res.unwrap().iter().flat_map(|m| m.into_iter()).collect();
        Ok(to_numpy_2d(py, m, 6))
    }

    /// Get a metric defined on all the mesh vertices such that
    ///  - for boundary vertices, the principal directions are aligned with the principal curvature
    ///    directions and the sizes to curvature radius ratio is r_h
    ///  - the metric is entended into the volume with gradation beta
    ///  - if an implied metric is provided, the result is limited to (1/step,step) times the
    ///    implied metric
    ///  - if a normal size array is not provided, the minimum of the tangential sizes is used.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (geom, r_h, beta, t=1.0/8.0, h_min=None, h_max=None, h_n=None, h_n_tags=None))]
    pub fn curvature_metric<'py>(
        &self,
        py: Python<'py>,
        geom: &LinearGeometry3d,
        r_h: f64,
        beta: f64,
        t: f64,
        h_min: Option<f64>,
        h_max: Option<f64>,
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
            self.0.curvature_metric(
                &geom.geom,
                r_h,
                beta,
                t,
                h_min,
                h_max,
                Some(h_n),
                Some(h_n_tags),
            )
        } else {
            self.0
                .curvature_metric(&geom.geom, r_h, beta, t, h_min, h_max, None, None)
        };

        if let Err(res) = res {
            return Err(PyRuntimeError::new_err(res.to_string()));
        }
        let m = res.unwrap();
        let m = m.iter().flat_map(|m| m.into_iter()).collect();

        Ok(to_numpy_2d(py, m, 6))
    }
}

#[pymethods]
impl Mesh32 {
    /// Reset the face tags of other to match those in self
    pub fn transfer_tags_face(&self, other: &mut Mesh33) -> PyResult<()> {
        let tree = ObjectIndex::new(&self.0);
        self.0
            .transfer_tags(&tree, &mut other.0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Reset the element tags of other to match those in self
    pub fn transfer_tags_elem(&self, other: &mut Self) -> PyResult<()> {
        let tree = ObjectIndex::new(&self.0);
        self.0
            .transfer_tags(&tree, &mut other.0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pymethods]
impl Mesh22 {
    /// Extract the boundary faces into a Mesh, and return the indices of the vertices in the
    /// parent mesh
    #[must_use]
    pub fn boundary<'py>(&self, py: Python<'py>) -> (Mesh21, Bound<'py, PyArray1<Idx>>) {
        let (bdy, ids) = self.0.boundary();
        (Mesh21(bdy), to_numpy_1d(py, ids))
    }

    pub fn implied_metric<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let res = self.0.implied_metric();

        if let Err(res) = res {
            return Err(PyRuntimeError::new_err(res.to_string()));
        }

        let m: Vec<f64> = res.unwrap().iter().flat_map(|m| m.into_iter()).collect();
        Ok(to_numpy_2d(py, m, 3))
    }

    /// Get a metric defined on all the mesh vertices such that
    ///  - for boundary vertices, the principal directions are aligned with the principal curvature
    ///    directions and the sizes to curvature radius ratio is r_h
    ///  - the metric is entended into the volume with gradation beta
    ///  - if a normal size array is not provided, the minimum of the tangential sizes is used.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (geom, r_h, beta, t=1.0/8.0, h_min=None, h_n=None, h_n_tags=None))]
    pub fn curvature_metric<'py>(
        &self,
        py: Python<'py>,
        geom: &LinearGeometry2d,
        r_h: f64,
        beta: f64,
        t: f64,
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
            self.0
                .curvature_metric(&geom.geom, r_h, beta, t, Some(h_n), Some(h_n_tags))
        } else {
            self.0
                .curvature_metric(&geom.geom, r_h, beta, t, None, None)
        };

        if let Err(res) = res {
            return Err(PyRuntimeError::new_err(res.to_string()));
        }
        let mut m = res.unwrap();

        if let Some(h_min) = h_min {
            for x in &mut m {
                x.scale_with_bounds(1.0, h_min, f64::MAX);
            }
        }

        let m: Vec<f64> = m.iter().flat_map(|m| m.into_iter()).collect();

        Ok(to_numpy_2d(py, m, 3))
    }
}

#[pymethods]
impl Mesh21 {
    /// Reset the face tags of other to match those in self
    pub fn transfer_tags_face(&self, other: &mut Mesh22) -> PyResult<()> {
        let tree = ObjectIndex::new(&self.0);
        self.0
            .transfer_tags(&tree, &mut other.0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Reset the element tags of other to match those in self
    pub fn transfer_tags_elem(&self, other: &mut Self) -> PyResult<()> {
        let tree = ObjectIndex::new(&self.0);
        self.0
            .transfer_tags(&tree, &mut other.0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}
