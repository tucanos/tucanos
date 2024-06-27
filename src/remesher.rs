use crate::{
    geometry::{LinearGeometry2d, LinearGeometry3d},
    mesh::{Mesh22, Mesh33},
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
use tucanos::{
    metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric, Metric},
    remesher::{Remesher, RemesherParams, SmoothingType},
    topo_elems::{Tetrahedron, Triangle},
    Idx,
};

macro_rules! create_remesher {
    ($name: ident, $dim: expr, $etype: ident, $metric: ident, $mesh: ident, $geom: ident) => {
        #[doc = concat!("Remesher for a meshes consisting of ", stringify!($etype), " in ", stringify!($dim), "D")]
        #[doc = concat!("using ", stringify!($metric), " as metric and a piecewise linear representation of the geometry")]
        #[pyclass]
        pub struct $name {
            remesher: Remesher<$dim, $etype, $metric>,
        }

        #[doc = concat!("Create a remesher from a ", stringify!($mesh), " and a ",stringify!($metric) ," metric defined at the mesh vertices")]
        #[doc = concat!("A piecewise linear representation of the geometry is used, either from the ", stringify!($geom), " given or otherwise from the mesh boundary.")]
        #[pymethods]
        impl $name {
            #[new]
            pub fn new(
                mesh: &$mesh,
                geometry: &$geom,
                m: PyReadonlyArray2<f64>,
            ) -> PyResult<Self> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != $metric::N as usize {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice()?;
                let m: Vec<_> = m.chunks($metric::N).map(|x| $metric::from_slice(x)).collect();

                let remesher = Remesher::new(&mesh.mesh, &m, &geometry.geom);
                if let Err(res) = remesher {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(Self {remesher: remesher.unwrap()})
            }

            /// Convert a Hessian $H$ to the optimal metric for a Lp norm, i.e.
            ///  $$ m = det(|H|)^{-1/(2p+dim)}|H| $$
            #[classmethod]
            pub fn hessian_to_metric<'py>(
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
                p: Option<Idx>,
            ) -> PyResult<Bound<'py,PyArray2<f64>>> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let mut res = Vec::with_capacity(m.shape()[0] * m.shape()[1]);
                let m = m.as_slice().unwrap();
                let mut m: Vec<_> = m.chunks($metric::N).map(|x| $metric::from_slice(x)).collect();

                let exponent = if let Some(p) = p {
                    2.0 / (2.0 * f64::from(p) + f64::from($dim))
                } else {
                    0.0
                };

                for m_v in m.iter_mut() {
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
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
                h_min: f64,
                h_max: f64,
                n_elems: Idx,
                fixed_m: Option<PyReadonlyArray2<f64>>,
                implied_m: Option<PyReadonlyArray2<f64>>,
                step: Option<f64>,
                max_iter: Option<Idx>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice().unwrap();
                let mut m: Vec<_> = m.chunks($metric::N).map(|x| $metric::from_slice(x)).collect();

                let res =  if let Some(fixed_m) = fixed_m {
                    let fixed_m = fixed_m.as_slice().unwrap();
                    let fixed_m: Vec<_> = fixed_m.chunks($metric::N).map(|x| $metric::from_slice(x)).collect();
                    if let Some(implied_m) = implied_m {
                        let implied_m = implied_m.as_slice().unwrap();
                        let implied_m: Vec<_> = implied_m.chunks($metric::N).map(|x| $metric::from_slice(x)).collect();
                        mesh.mesh
                            .scale_metric(&mut m, h_min, h_max, n_elems, Some(&fixed_m), Some(&implied_m), step, max_iter.unwrap_or(10))
                    } else {
                        mesh.mesh
                            .scale_metric(&mut m, h_min, h_max, n_elems, Some(&fixed_m), None, step, max_iter.unwrap_or(10))
                    }
                } else if let Some(implied_m) = implied_m {
                    let implied_m = implied_m.as_slice().unwrap();
                    let implied_m: Vec<_> = implied_m.chunks($metric::N).map(|x| $metric::from_slice(x)).collect();
                    mesh.mesh
                        .scale_metric(&mut m, h_min, h_max, n_elems, None, Some(&implied_m), step, max_iter.unwrap_or(10))
                } else {
                    mesh.mesh
                    .scale_metric(&mut m, h_min, h_max, n_elems, None, None, None, max_iter.unwrap_or(10))
                };

                if let Err(res) = res {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }

                let m: Vec<_> = m.iter().cloned().flatten().collect();
                return Ok(to_numpy_2d(py, m, <$metric as Metric<$dim>>::N));
            }

            /// Smooth a metric field
            #[classmethod]
            pub fn smooth_metric<'py>(
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice().unwrap();
                let m: Vec<_> = m.chunks($metric::N).map(|x| $metric::from_slice(x)).collect();
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
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
                beta: f64,
                n_iter: Idx,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice().unwrap();
                let mut m: Vec<_> = m.chunks($metric::N).map(|x| $metric::from_slice(x)).collect();
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
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if m.shape()[0] != mesh.mesh.n_elems() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice().unwrap();
                let m: Vec<_> = m.chunks($metric::N).map(|x| $metric::from_slice(x)).collect();
                let res = mesh.mesh.elem_data_to_vertex_data_metric::<$metric>(&m);
                match res {
                    Ok(res) => {
                        let res: Vec<_> = res.iter().cloned().flatten().collect();
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
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice().unwrap();
                let m: Vec<_> = m.chunks($metric::N).map(|x| $metric::from_slice(x)).collect();
                let res = mesh.mesh.vertex_data_to_elem_data_metric::<$metric>(&m);
                match res {
                    Ok(res) => {
                        let res: Vec<_> = res.iter().cloned().flatten().collect();
                        return Ok(to_numpy_2d(py, res, <$metric as Metric<$dim>>::N));
                    }
                    Err(res) => {
                        return Err(PyRuntimeError::new_err(res.to_string()));
                    }
                }
            }

            /// Limit a metric to be between 1/step and step times another metric
            #[classmethod]
            #[allow(clippy::too_many_arguments)]
            pub fn control_step_metric<'py>(
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
                m_other: PyReadonlyArray2<f64>,
                step: f64,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                if m_other.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m_other.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice().unwrap();
                let m = m.chunks($metric::N).map(|x| $metric::from_slice(x));

                let m_other = m_other.as_slice().unwrap();
                let m_other = m_other.chunks($metric::N).map(|x| $metric::from_slice(x));

                let mut res = Vec::with_capacity(mesh.mesh.n_verts() as usize * <$metric as Metric<$dim>>::N);

                for (mut m_i, m_other_i) in m.zip(m_other) {
                    m_i.control_step(&m_other_i, step);
                    res.extend(m_i.into_iter());
                }

                return Ok(to_numpy_2d(py, res, <$metric as Metric<$dim>>::N));
            }

            /// Compute the min/max sizes, max anisotropy and complexity of a metric
            #[classmethod]
            pub fn metric_info(
                _cls: &Bound<'_, PyType>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
            ) -> (f64, f64, f64, f64) {
                let m = m.as_slice().unwrap();
                let m = m.chunks($metric::N).map(|x| $metric::from_slice(x)).collect::<Vec<_>>();
                mesh.mesh.metric_info(&m)
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
            #[must_use]
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

            /// Get the default remesher parameters
            pub fn default_params<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyDict> {
                let default_params = RemesherParams::default();
                let dict = PyDict::new_bound(py);
                dict.set_item("num_iter", default_params.num_iter).unwrap();
                dict.set_item("two_steps", default_params.two_steps).unwrap();
                dict.set_item("split_max_iter", default_params.split_max_iter).unwrap();
                dict.set_item("split_min_l_rel", default_params.split_min_l_rel).unwrap();
                dict.set_item("split_min_l_abs", default_params.split_min_l_abs).unwrap();
                dict.set_item("split_min_q_rel", default_params.split_min_q_rel).unwrap();
                dict.set_item("split_min_q_abs", default_params.split_min_q_abs).unwrap();
                dict.set_item("collapse_max_iter", default_params.collapse_max_iter).unwrap();
                dict.set_item("collapse_max_l_rel", default_params.collapse_max_l_rel).unwrap();
                dict.set_item("collapse_max_l_abs", default_params.collapse_max_l_abs).unwrap();
                dict.set_item("collapse_min_q_rel", default_params.collapse_min_q_rel).unwrap();
                dict.set_item("collapse_min_q_abs", default_params.collapse_min_q_abs).unwrap();
                dict.set_item("swap_max_iter", default_params.swap_max_iter).unwrap();
                dict.set_item("swap_max_l_rel", default_params.swap_max_l_rel).unwrap();
                dict.set_item("swap_max_l_abs", default_params.swap_max_l_abs).unwrap();
                dict.set_item("swap_min_l_rel", default_params.swap_min_l_rel).unwrap();
                dict.set_item("swap_min_l_abs", default_params.swap_min_l_abs).unwrap();
                dict.set_item("smooth_iter", default_params.smooth_iter).unwrap();
                let smooth_type = match default_params.smooth_type {
                    tucanos::remesher::SmoothingType::Laplacian => "laplacian",
                    tucanos::remesher::SmoothingType::Avro => "avro",
                    #[cfg(feature = "nlopt")]
                    tucanos::remesher::SmoothingType::NLOpt => "nlopt",
                    tucanos::remesher::SmoothingType::Laplacian2 => "laplacian2",
                };
                dict.set_item("smooth_type", smooth_type).unwrap();
                dict.set_item("smooth_relax", to_numpy_1d(py, default_params.smooth_relax)).unwrap();
                dict.set_item("max_angle", default_params.max_angle).unwrap();

                dict
            }

            /// Perform a remeshing iteration
            #[allow(clippy::too_many_arguments)]
            pub fn remesh(
                &mut self,
                geometry: &$geom,
                num_iter:Option< u32>,
                two_steps: Option<bool>,
                split_max_iter:Option< u32>,
                split_min_l_rel:Option< f64>,
                split_min_l_abs:Option< f64>,
                split_min_q_rel:Option< f64>,
                split_min_q_abs:Option< f64>,
                collapse_max_iter:Option< u32>,
                collapse_max_l_rel:Option< f64>,
                collapse_max_l_abs:Option< f64>,
                collapse_min_q_rel:Option< f64>,
                collapse_min_q_abs:Option< f64>,
                swap_max_iter:Option< u32>,
                swap_max_l_rel:Option< f64>,
                swap_max_l_abs:Option< f64>,
                swap_min_l_rel:Option< f64>,
                swap_min_l_abs:Option< f64>,
                smooth_iter:Option< u32>,
                smooth_type: Option<&str>,
                smooth_relax: Option<PyReadonlyArray1<f64>>,
                smooth_keep_local_minima: Option<bool>,
                max_angle:Option< f64>,
                debug: Option<bool>,
            ) -> PyResult<()>{
                let smooth_type = smooth_type.unwrap_or("laplacian");

                let smooth_type = if smooth_type == "laplacian" {
                    SmoothingType::Laplacian
                } else if smooth_type == "laplacian2" {
                    SmoothingType::Laplacian2
                } else if smooth_type == "nlopt" {
                    unreachable!()
                } else {
                    SmoothingType::Avro
                };

                let default_params = RemesherParams::default();

                let params = RemesherParams {
                    num_iter: num_iter.unwrap_or(default_params.num_iter),
                    two_steps: two_steps.unwrap_or(default_params.two_steps),
                    split_max_iter: split_max_iter.unwrap_or(default_params.split_max_iter),
                    split_min_l_rel: split_min_l_rel.unwrap_or(default_params.split_min_l_rel),
                    split_min_l_abs: split_min_l_abs.unwrap_or(default_params.split_min_l_abs),
                    split_min_q_rel: split_min_q_rel.unwrap_or(default_params.split_min_q_rel),
                    split_min_q_abs: split_min_q_abs.unwrap_or(default_params.split_min_q_abs),
                    collapse_max_iter: collapse_max_iter.unwrap_or(default_params.collapse_max_iter),
                    collapse_max_l_rel: collapse_max_l_rel.unwrap_or(default_params.collapse_max_l_rel),
                    collapse_max_l_abs: collapse_max_l_abs.unwrap_or(default_params.collapse_max_l_abs),
                    collapse_min_q_rel: collapse_min_q_rel.unwrap_or(default_params.collapse_min_q_rel),
                    collapse_min_q_abs: collapse_min_q_abs.unwrap_or(default_params.collapse_min_q_abs),
                    swap_max_iter: swap_max_iter.unwrap_or(default_params.swap_max_iter),
                    swap_max_l_rel: swap_max_l_rel.unwrap_or(default_params.swap_max_l_rel),
                    swap_max_l_abs: swap_max_l_abs.unwrap_or(default_params.swap_max_l_abs),
                    swap_min_l_rel: swap_min_l_rel.unwrap_or(default_params.swap_min_l_rel),
                    swap_min_l_abs: swap_min_l_abs.unwrap_or(default_params.swap_min_l_abs),
                    smooth_iter: smooth_iter.unwrap_or(default_params.smooth_iter),
                    smooth_type,
                    smooth_relax: smooth_relax.map(|x| x.to_vec().unwrap()).unwrap_or(default_params.smooth_relax),
                    smooth_keep_local_minima: smooth_keep_local_minima.unwrap_or(default_params.smooth_keep_local_minima),
                    max_angle: max_angle.unwrap_or(default_params.max_angle),
                    debug: debug.unwrap_or(default_params.debug),
                };
                self.remesher.remesh(params, &geometry.geom).map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }

            /// Get the element qualities as a numpy array of size (# or elements)
            #[must_use]
            pub fn qualities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                to_numpy_1d(py, self.remesher.qualities())
            }

            /// Get the element lengths (in metric space) as a numpy array of size (# or edges)
            #[must_use]
            pub fn lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                to_numpy_1d(py, self.remesher.lengths())
            }

            /// Get the infomation about the remeshing steps performed in remesh() as a json string
            #[must_use]
            pub fn stats_json(&self) -> String {
                self.remesher.stats_json()
            }
        }
    };
}

type IsoMetric2d = IsoMetric<2>;
type IsoMetric3d = IsoMetric<3>;
create_remesher!(
    Remesher2dIso,
    2,
    Triangle,
    IsoMetric2d,
    Mesh22,
    LinearGeometry2d
);
create_remesher!(
    Remesher2dAniso,
    2,
    Triangle,
    AnisoMetric2d,
    Mesh22,
    LinearGeometry2d
);
create_remesher!(
    Remesher3dIso,
    3,
    Tetrahedron,
    IsoMetric3d,
    Mesh33,
    LinearGeometry3d
);
create_remesher!(
    Remesher3dAniso,
    3,
    Tetrahedron,
    AnisoMetric3d,
    Mesh33,
    LinearGeometry3d
);
