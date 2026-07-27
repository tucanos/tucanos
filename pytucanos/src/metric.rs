use crate::{
    geometry::{LinearGeometry2d, LinearGeometry3d, QuadraticGeometry2d, QuadraticGeometry3d},
    mesh::{PyMesh2d, PyMesh3d},
    to_numpy_2d,
};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{Bound, PyResult, Python, exceptions::PyRuntimeError, pyfunction};

use tmesh::mesh::Mesh;
use tucanos::{
    Tag,
    metric::{AnisoMetric2d, AnisoMetric3d, Metric, MetricField},
};

/// Intersect 3d anisotropic metrics
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn intersect_aniso_metric_3d<'py>(
    py: Python<'py>,
    m1: PyReadonlyArray2<f64>,
    m2: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if m1.shape()[1] != 6 || m2.shape()[1] != 6 {
        return Err(PyRuntimeError::new_err("Metrics must have 6 components"));
    }

    if m1.shape()[0] != m2.shape()[0] {
        return Err(PyRuntimeError::new_err(
            "Metrics must have the same number of rows",
        ));
    }

    let m1 = crate::as_c_slice(&m1)?;
    let m2 = crate::as_c_slice(&m2)?;

    let m = m1
        .chunks(6)
        .zip(m2.chunks(6))
        .flat_map(|(x, y)| {
            let m1 = AnisoMetric3d::from_slice(x);
            let m2 = AnisoMetric3d::from_slice(y);
            m1.intersect(&m2)
        })
        .collect::<Vec<_>>();
    Ok(to_numpy_2d(py, m, 6))
}

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
#[pyo3(signature = (msh, geom, r_h, beta, t=1.0, h_min=None, h_max=None, h_n=None, h_n_tags=None, max_surface_anisotropy=None, max_surface_h=None))]
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
    max_surface_anisotropy: Option<f64>,
    max_surface_h: Option<f64>,
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
            max_surface_anisotropy,
            max_surface_h,
        )
    } else {
        MetricField::curvature_metric_3d(
            &msh.0,
            &v2v,
            &geom.geom,
            r_h,
            beta,
            t,
            h_min,
            h_max,
            None,
            None,
            max_surface_anisotropy,
            max_surface_h,
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
#[pyo3(signature = (msh, geom, r_h, beta, t=1.0, h_min=None, h_max=None, h_n=None, h_n_tags=None, max_surface_anisotropy=None, max_surface_h=None))]
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
    max_surface_anisotropy: Option<f64>,
    max_surface_h: Option<f64>,
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
            max_surface_anisotropy,
            max_surface_h,
        )
    } else {
        MetricField::curvature_metric_3d(
            &msh.0,
            &v2v,
            &geom.geom,
            r_h,
            beta,
            t,
            h_min,
            h_max,
            None,
            None,
            max_surface_anisotropy,
            max_surface_h,
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

/// Intersect 3d anisotropic metrics
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn intersect_aniso_metric_2d<'py>(
    py: Python<'py>,
    m1: PyReadonlyArray2<f64>,
    m2: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if m1.shape()[1] != 3 || m2.shape()[1] != 3 {
        return Err(PyRuntimeError::new_err("Metrics must have 6 components"));
    }

    if m1.shape()[0] != m2.shape()[0] {
        return Err(PyRuntimeError::new_err(
            "Metrics must have the same number of rows",
        ));
    }

    let m1 = crate::as_c_slice(&m1)?;
    let m2 = crate::as_c_slice(&m2)?;

    let m = m1
        .chunks(3)
        .zip(m2.chunks(3))
        .flat_map(|(x, y)| {
            let m1 = AnisoMetric2d::from_slice(x);
            let m2 = AnisoMetric2d::from_slice(y);
            m1.intersect(&m2)
        })
        .collect::<Vec<_>>();
    Ok(to_numpy_2d(py, m, 3))
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
