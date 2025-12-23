use crate::{
    geometry::{LinearGeometry2d, LinearGeometry3d, QuadraticGeometry2d, QuadraticGeometry3d},
    mesh::{PyMesh2d, PyMesh3d},
    to_numpy_2d,
};
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::{Bound, PyResult, Python, exceptions::PyRuntimeError, pyfunction};

use tmesh::mesh::Mesh;
use tucanos::{Tag, metric::MetricField};

/// Get the element-implied metric
#[pyfunction]
pub fn implied_metric_3d<'py>(py: Python<'py>, msh: &PyMesh3d) -> Bound<'py, PyArray2<f64>> {
    let res = MetricField::implied_metric(&msh.0);

    let m: Vec<f64> = res.metric().iter().flat_map(|m| m.into_iter()).collect();
    to_numpy_2d(py, m, 6)
}

/// Get a metric defined on all the mesh vertices such that
///  - for boundary vertices, the principal directions are aligned with the principal curvature
///    directions and the sizes to curvature radius ratio is r_h
///  - the metric is entended into the volume with gradation beta
///  - if an implied metric is provided, the result is limited to (1/step,step) times the
///    implied metric
///  - if a normal size array is not provided, the minimum of the tangential sizes is used.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (msh, geom, r_h, beta, t=1.0, h_min=None, h_max=None, h_n=None, h_n_tags=None))]
pub fn curvature_metric_3d<'py>(
    py: Python<'py>,
    msh: &PyMesh3d,
    geom: &LinearGeometry3d,
    r_h: f64,
    beta: f64,
    t: f64,
    h_min: Option<f64>,
    h_max: Option<f64>,
    h_n: Option<PyReadonlyArray1<f64>>,
    h_n_tags: Option<PyReadonlyArray1<Tag>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let v2v = msh.0.vertex_to_vertices();

    let res = if let Some(h_n) = h_n {
        let h_n = h_n.as_slice()?;
        if h_n_tags.is_none() {
            return Err(PyRuntimeError::new_err("h_n_tags not given"));
        }
        let h_n_tags = h_n_tags.unwrap();
        let h_n_tags = h_n_tags.as_slice()?;
        MetricField::curvature_metric_3d(
            &msh.0,
            &v2v,
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
        MetricField::curvature_metric_3d(
            &msh.0, &v2v, &geom.geom, r_h, beta, t, h_min, h_max, None, None,
        )
    };

    let m = res
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .metric()
        .iter()
        .flat_map(|m| m.into_iter())
        .collect();

    Ok(to_numpy_2d(py, m, 6))
}

/// Get a metric defined on all the mesh vertices such that
///  - for boundary vertices, the principal directions are aligned with the principal curvature
///    directions and the sizes to curvature radius ratio is r_h
///  - the metric is entended into the volume with gradation beta
///  - if an implied metric is provided, the result is limited to (1/step,step) times the
///    implied metric
///  - if a normal size array is not provided, the minimum of the tangential sizes is used.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (msh, geom, r_h, beta, t=1.0, h_min=None, h_max=None, h_n=None, h_n_tags=None))]
pub fn curvature_metric_3d_quadratic<'py>(
    py: Python<'py>,
    msh: &PyMesh3d,
    geom: &QuadraticGeometry3d,
    r_h: f64,
    beta: f64,
    t: f64,
    h_min: Option<f64>,
    h_max: Option<f64>,
    h_n: Option<PyReadonlyArray1<f64>>,
    h_n_tags: Option<PyReadonlyArray1<Tag>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let v2v = msh.0.vertex_to_vertices();

    let res = if let Some(h_n) = h_n {
        let h_n = h_n.as_slice()?;
        if h_n_tags.is_none() {
            return Err(PyRuntimeError::new_err("h_n_tags not given"));
        }
        let h_n_tags = h_n_tags.unwrap();
        let h_n_tags = h_n_tags.as_slice()?;
        MetricField::curvature_metric_3d(
            &msh.0,
            &v2v,
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
        MetricField::curvature_metric_3d(
            &msh.0, &v2v, &geom.geom, r_h, beta, t, h_min, h_max, None, None,
        )
    };

    let m = res
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .metric()
        .iter()
        .flat_map(|m| m.into_iter())
        .collect();

    Ok(to_numpy_2d(py, m, 6))
}

/// Get the element-implied metric
#[pyfunction]
pub fn implied_metric_2d<'py>(py: Python<'py>, msh: &PyMesh2d) -> Bound<'py, PyArray2<f64>> {
    let res = MetricField::implied_metric(&msh.0);

    let m: Vec<f64> = res.metric().iter().flat_map(|m| m.into_iter()).collect();
    to_numpy_2d(py, m, 3)
}

/// Get a metric defined on all the mesh vertices such that
///  - for boundary vertices, the principal directions are aligned with the principal curvature
///    directions and the sizes to curvature radius ratio is r_h
///  - the metric is entended into the volume with gradation beta
///  - if a normal size array is not provided, the minimum of the tangential sizes is used.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (msh, geom, r_h, beta, t=1.0, h_min=None, h_max=None, h_n=None, h_n_tags=None))]
pub fn curvature_metric_2d<'py>(
    py: Python<'py>,
    msh: &PyMesh2d,
    geom: &LinearGeometry2d,
    r_h: f64,
    beta: f64,
    t: f64,
    h_min: Option<f64>,
    h_max: Option<f64>,
    h_n: Option<PyReadonlyArray1<f64>>,
    h_n_tags: Option<PyReadonlyArray1<Tag>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let v2v = msh.0.vertex_to_vertices();

    let res = if let Some(h_n) = h_n {
        let h_n = h_n.as_slice()?;
        if h_n_tags.is_none() {
            return Err(PyRuntimeError::new_err("h_n_tags not given"));
        }
        let h_n_tags = h_n_tags.unwrap();
        let h_n_tags = h_n_tags.as_slice()?;
        MetricField::curvature_metric_2d(
            &msh.0,
            &v2v,
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
        MetricField::curvature_metric_2d(
            &msh.0, &v2v, &geom.geom, r_h, beta, t, h_min, h_max, None, None,
        )
    };

    let m = res
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .metric()
        .iter()
        .flat_map(|m| m.into_iter())
        .collect();

    Ok(to_numpy_2d(py, m, 3))
}

/// Get a metric defined on all the mesh vertices such that
///  - for boundary vertices, the principal directions are aligned with the principal curvature
///    directions and the sizes to curvature radius ratio is r_h
///  - the metric is entended into the volume with gradation beta
///  - if a normal size array is not provided, the minimum of the tangential sizes is used.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (msh, geom, r_h, beta, t=1.0, h_min=None, h_max=None, h_n=None, h_n_tags=None))]
pub fn curvature_metric_2d_quadratic<'py>(
    py: Python<'py>,
    msh: &PyMesh2d,
    geom: &QuadraticGeometry2d,
    r_h: f64,
    beta: f64,
    t: f64,
    h_min: Option<f64>,
    h_max: Option<f64>,
    h_n: Option<PyReadonlyArray1<f64>>,
    h_n_tags: Option<PyReadonlyArray1<Tag>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let v2v = msh.0.vertex_to_vertices();

    let res = if let Some(h_n) = h_n {
        let h_n = h_n.as_slice()?;
        if h_n_tags.is_none() {
            return Err(PyRuntimeError::new_err("h_n_tags not given"));
        }
        let h_n_tags = h_n_tags.unwrap();
        let h_n_tags = h_n_tags.as_slice()?;
        MetricField::curvature_metric_2d(
            &msh.0,
            &v2v,
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
        MetricField::curvature_metric_2d(
            &msh.0, &v2v, &geom.geom, r_h, beta, t, h_min, h_max, None, None,
        )
    };

    let m = res
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .metric()
        .iter()
        .flat_map(|m| m.into_iter())
        .collect();

    Ok(to_numpy_2d(py, m, 3))
}
