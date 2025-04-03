use crate::{
    geometry::{LinearGeometry2d, LinearGeometry3d},
    mesh::{Mesh22, Mesh33},
    to_numpy_1d, to_numpy_2d,
};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{
    Bound, PyResult, Python,
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods,
    types::PyType,
};
use tucanos::{
    Idx,
    mesh::{Tetrahedron, Triangle},
    metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric, Metric},
    remesher::{
        CollapseParams, Remesher, RemesherParams, RemeshingStep, SmoothParams, SmoothingMethod,
        SplitParams, SwapParams,
    },
};

#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub struct PyCollapseParams {
    l: f64,
    max_iter: Idx,
    max_l_rel: f64,
    max_l_abs: f64,
    min_q_rel: f64,
    min_q_abs: f64,
    max_angle: f64,
}

impl PyCollapseParams {
    fn from(other: &CollapseParams) -> Self {
        Self {
            l: other.l,
            max_iter: other.max_iter,
            max_l_rel: other.max_l_rel,
            max_l_abs: other.max_l_abs,
            min_q_rel: other.min_q_rel,
            min_q_abs: other.min_q_abs,
            max_angle: other.max_angle,
        }
    }
    fn to(&self) -> CollapseParams {
        CollapseParams {
            l: self.l,
            max_iter: self.max_iter,
            max_l_rel: self.max_l_rel,
            max_l_abs: self.max_l_abs,
            min_q_rel: self.min_q_rel,
            min_q_abs: self.min_q_abs,
            max_angle: self.max_angle,
        }
    }
}

#[pymethods]
impl PyCollapseParams {
    #[new]
    pub fn new(
        l: f64,
        max_iter: Idx,
        max_l_rel: f64,
        max_l_abs: f64,
        min_q_rel: f64,
        min_q_abs: f64,
        max_angle: f64,
    ) -> Self {
        Self {
            l,
            max_iter,
            max_l_rel,
            max_l_abs,
            min_q_rel,
            min_q_abs,
            max_angle,
        }
    }

    #[classmethod]
    pub fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self::from(&CollapseParams::default())
    }
}

#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub struct PySplitParams {
    l: f64,
    max_iter: Idx,
    min_l_rel: f64,
    min_l_abs: f64,
    min_q_rel: f64,
    min_q_rel_bdy: f64,
    min_q_abs: f64,
}

impl PySplitParams {
    fn from(other: &SplitParams) -> Self {
        Self {
            l: other.l,
            max_iter: other.max_iter,
            min_l_rel: other.min_l_rel,
            min_l_abs: other.min_l_abs,
            min_q_rel: other.min_q_rel,
            min_q_rel_bdy: other.min_q_rel_bdy,
            min_q_abs: other.min_q_abs,
        }
    }
    fn to(&self) -> SplitParams {
        SplitParams {
            l: self.l,
            max_iter: self.max_iter,
            min_l_rel: self.min_l_rel,
            min_l_abs: self.min_l_abs,
            min_q_rel: self.min_q_rel,
            min_q_rel_bdy: self.min_q_rel_bdy,
            min_q_abs: self.min_q_abs,
        }
    }
}

#[pymethods]
impl PySplitParams {
    #[new]
    #[allow(clippy::similar_names)]
    pub fn new(
        l: f64,
        max_iter: Idx,
        min_l_rel: f64,
        min_l_abs: f64,
        min_q_rel: f64,
        min_q_rel_bdy: f64,
        min_q_abs: f64,
    ) -> Self {
        Self {
            l,
            max_iter,
            min_l_rel,
            min_l_abs,
            min_q_rel,
            min_q_rel_bdy,
            min_q_abs,
        }
    }
    #[classmethod]
    pub fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self::from(&SplitParams::default())
    }
}

#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub struct PySwapParams {
    q: f64,
    max_iter: Idx,
    max_l_rel: f64,
    max_l_abs: f64,
    min_l_rel: f64,
    min_l_abs: f64,
    max_angle: f64,
}

impl PySwapParams {
    fn from(other: &SwapParams) -> Self {
        Self {
            q: other.q,
            max_iter: other.max_iter,
            max_l_rel: other.max_l_rel,
            max_l_abs: other.max_l_abs,
            min_l_rel: other.min_l_rel,
            min_l_abs: other.min_l_abs,
            max_angle: other.max_angle,
        }
    }
    fn to(&self) -> SwapParams {
        SwapParams {
            q: self.q,
            max_iter: self.max_iter,
            max_l_rel: self.max_l_rel,
            max_l_abs: self.max_l_abs,
            min_l_rel: self.min_l_rel,
            min_l_abs: self.min_l_abs,
            max_angle: self.max_angle,
        }
    }
}

#[pymethods]
impl PySwapParams {
    #[new]
    pub fn new(
        q: f64,
        max_iter: Idx,
        max_l_rel: f64,
        max_l_abs: f64,
        min_l_rel: f64,
        min_l_abs: f64,
        max_angle: f64,
    ) -> Self {
        Self {
            q,
            max_iter,
            max_l_rel,
            max_l_abs,
            min_l_rel,
            min_l_abs,
            max_angle,
        }
    }
    #[classmethod]
    pub fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self::from(&SwapParams::default())
    }
}

#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq, Eq)]
pub enum PySmoothingMethod {
    Laplacian,
    Avro,
    #[cfg(feature = "nlopt")]
    NLOpt,
    Laplacian2,
}

#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub struct PySmoothParams {
    n_iter: Idx,
    method: PySmoothingMethod,
    relax: Vec<f64>,
    keep_local_minima: bool,
    max_angle: f64,
}

impl PySmoothParams {
    fn from(other: &SmoothParams) -> Self {
        let method = match other.method {
            SmoothingMethod::Laplacian => PySmoothingMethod::Laplacian,
            SmoothingMethod::Avro => PySmoothingMethod::Avro,
            #[cfg(feature = "nlopt")]
            SmoothingMethod::NLOpt => PySmoothingMethod::NLOpt,
            SmoothingMethod::Laplacian2 => PySmoothingMethod::Laplacian2,
        };
        Self {
            n_iter: other.n_iter,
            method,
            relax: other.relax.clone(),
            keep_local_minima: other.keep_local_minima,
            max_angle: other.max_angle,
        }
    }
    fn to(&self) -> SmoothParams {
        let method = match self.method {
            PySmoothingMethod::Laplacian => SmoothingMethod::Laplacian,
            PySmoothingMethod::Avro => SmoothingMethod::Avro,
            #[cfg(feature = "nlopt")]
            PySmoothingMethod::NLOpt => SmoothingMethod::NLOpt,
            PySmoothingMethod::Laplacian2 => SmoothingMethod::Laplacian2,
        };
        SmoothParams {
            n_iter: self.n_iter,
            method,
            relax: self.relax.clone(),
            keep_local_minima: self.keep_local_minima,
            max_angle: self.max_angle,
        }
    }
}

#[pymethods]
impl PySmoothParams {
    #[new]
    pub fn new(
        n_iter: Idx,
        method: PySmoothingMethod,
        relax: Vec<f64>,
        keep_local_minima: bool,
        max_angle: f64,
    ) -> Self {
        Self {
            n_iter,
            method,
            relax,
            keep_local_minima,
            max_angle,
        }
    }
    #[classmethod]
    pub fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self::from(&SmoothParams::default())
    }
}

#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub enum PyRemeshingStep {
    Split(PySplitParams),
    Collapse(PyCollapseParams),
    Swap(PySwapParams),
    Smooth(PySmoothParams),
}

#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub struct PyRemesherParams {
    steps: Vec<PyRemeshingStep>,
    debug: bool,
}
impl PyRemesherParams {
    pub fn from(other: &RemesherParams) -> Self {
        let steps = other
            .steps
            .iter()
            .map(|s| match s {
                RemeshingStep::Split(p) => PyRemeshingStep::Split(PySplitParams::from(p)),
                RemeshingStep::Collapse(p) => PyRemeshingStep::Collapse(PyCollapseParams::from(p)),
                RemeshingStep::Swap(p) => PyRemeshingStep::Swap(PySwapParams::from(p)),
                RemeshingStep::Smooth(p) => PyRemeshingStep::Smooth(PySmoothParams::from(p)),
            })
            .collect::<Vec<_>>();
        Self {
            steps,
            debug: other.debug,
        }
    }

    pub fn to(&self) -> RemesherParams {
        let steps = self
            .steps
            .iter()
            .map(|s| match s {
                PyRemeshingStep::Split(p) => RemeshingStep::Split(p.to()),
                PyRemeshingStep::Collapse(p) => RemeshingStep::Collapse(p.to()),
                PyRemeshingStep::Swap(p) => RemeshingStep::Swap(p.to()),
                PyRemeshingStep::Smooth(p) => RemeshingStep::Smooth(p.to()),
            })
            .collect::<Vec<_>>();
        RemesherParams {
            steps,
            debug: self.debug,
        }
    }
}
#[pymethods]
impl PyRemesherParams {
    #[new]
    pub fn new(steps: Vec<PyRemeshingStep>, debug: bool) -> Self {
        Self { steps, debug }
    }

    #[classmethod]
    pub fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self::from(&RemesherParams::default())
    }
}

macro_rules! create_remesher {
    ($name: ident, $dim: expr, $etype: ident, $metric: ident, $mesh: ident, $geom: ident) => {
        #[doc = concat!("Remesher for a meshes consisting of ", stringify!($etype), " in ",
                stringify!($dim), "D")]
        #[doc = concat!("using ", stringify!($metric),
                " as metric and a piecewise linear representation of the geometry")]
        #[pyclass]
        pub struct $name {
            remesher: Remesher<$dim, $etype, $metric>,
        }

        #[doc = concat!("Create a remesher from a ", stringify!($mesh), " and a ",
                stringify!($metric) ," metric defined at the mesh vertices")]
        #[doc = concat!(
                    "A piecewise linear representation of the geometry is used, either from the ",
                    stringify!($geom), " given or otherwise from the mesh boundary.")]
        #[pymethods]
        impl $name {
            #[new]
            pub fn new(mesh: &$mesh, geometry: &$geom, m: PyReadonlyArray2<f64>) -> PyResult<Self> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != $metric::N as usize {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice()?;
                let m: Vec<_> = m
                    .chunks($metric::N)
                    .map(|x| $metric::from_slice(x))
                    .collect();

                let remesher = Remesher::new(&mesh.mesh, &m, &geometry.geom);
                if let Err(res) = remesher {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(Self {
                    remesher: remesher.unwrap(),
                })
            }

            /// Convert a Hessian $H$ to the optimal metric for a Lp norm, i.e.
            ///  $$ m = det(|H|)^{-1/(2p+dim)}|H| $$
            #[classmethod]
            #[pyo3(signature = (mesh, m, p=None))]
            pub fn hessian_to_metric<'py>(
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
                p: Option<Idx>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let mut res = Vec::with_capacity(m.shape()[0] * m.shape()[1]);
                let m = m.as_slice().unwrap();
                let mut m: Vec<_> = m
                    .chunks($metric::N)
                    .map(|x| $metric::from_slice(x))
                    .collect();

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

            /// Scale a metric field to reach the desired (ideal) number of elements using
            /// min / max bounds on the cell size
            #[classmethod]
            #[allow(clippy::too_many_arguments)]
            #[pyo3(signature = (mesh, m, h_min, h_max, n_elems, fixed_m=None, implied_m=None,
                        step=None, max_iter=None))]
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
                let mut m: Vec<_> = m
                    .chunks($metric::N)
                    .map(|x| $metric::from_slice(x))
                    .collect();

                let res = if let Some(fixed_m) = fixed_m {
                    let fixed_m = fixed_m.as_slice().unwrap();
                    let fixed_m: Vec<_> = fixed_m
                        .chunks($metric::N)
                        .map(|x| $metric::from_slice(x))
                        .collect();
                    if let Some(implied_m) = implied_m {
                        let implied_m = implied_m.as_slice().unwrap();
                        let implied_m: Vec<_> = implied_m
                            .chunks($metric::N)
                            .map(|x| $metric::from_slice(x))
                            .collect();
                        mesh.mesh.scale_metric(
                            &mut m,
                            h_min,
                            h_max,
                            n_elems,
                            Some(&fixed_m),
                            Some(&implied_m),
                            step,
                            max_iter.unwrap_or(10),
                        )
                    } else {
                        mesh.mesh.scale_metric(
                            &mut m,
                            h_min,
                            h_max,
                            n_elems,
                            Some(&fixed_m),
                            None,
                            step,
                            max_iter.unwrap_or(10),
                        )
                    }
                } else if let Some(implied_m) = implied_m {
                    let implied_m = implied_m.as_slice().unwrap();
                    let implied_m: Vec<_> = implied_m
                        .chunks($metric::N)
                        .map(|x| $metric::from_slice(x))
                        .collect();
                    mesh.mesh.scale_metric(
                        &mut m,
                        h_min,
                        h_max,
                        n_elems,
                        None,
                        Some(&implied_m),
                        step,
                        max_iter.unwrap_or(10),
                    )
                } else {
                    mesh.mesh.scale_metric(
                        &mut m,
                        h_min,
                        h_max,
                        n_elems,
                        None,
                        None,
                        None,
                        max_iter.unwrap_or(10),
                    )
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
                let m: Vec<_> = m
                    .chunks($metric::N)
                    .map(|x| $metric::from_slice(x))
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
            #[pyo3(signature = (mesh, m, beta, t=1.0/8.0, n_iter=10))]
            pub fn apply_metric_gradation<'py>(
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
                beta: f64,
                t: f64,
                n_iter: Idx,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                if m.shape()[0] != mesh.mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != <$metric as Metric<$dim>>::N {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice().unwrap();
                let mut m: Vec<_> = m
                    .chunks($metric::N)
                    .map(|x| $metric::from_slice(x))
                    .collect();
                let res = mesh.mesh.apply_metric_gradation(&mut m, beta, t, n_iter);
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

            /// Convert a metic field defined at the element centers (P0) to a field defined at the
            /// vertices (P1) using a weighted average.
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
                let m: Vec<_> = m
                    .chunks($metric::N)
                    .map(|x| $metric::from_slice(x))
                    .collect();
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
                let m: Vec<_> = m
                    .chunks($metric::N)
                    .map(|x| $metric::from_slice(x))
                    .collect();
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

                let mut res =
                    Vec::with_capacity(mesh.mesh.n_verts() as usize * <$metric as Metric<$dim>>::N);

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
                let m = m
                    .chunks($metric::N)
                    .map(|x| $metric::from_slice(x))
                    .collect::<Vec<_>>();
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
            #[pyo3(signature = (only_bdy_faces=None))]
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
            pub fn remesh(&mut self, geometry: &$geom, params: &PyRemesherParams) -> PyResult<()> {
                self.remesher
                    .remesh(&params.to(), &geometry.geom)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
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

            /// Get the metric
            #[must_use]
            pub fn metric<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
                return to_numpy_2d(py, self.remesher.metric(), <$metric as Metric<$dim>>::N);
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
