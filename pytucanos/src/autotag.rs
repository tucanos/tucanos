use crate::{
    mesh::{PyBoundaryMesh2d, PyBoundaryMesh3d, PyMesh2d, PyMesh3d},
    to_numpy_1d,
};
use pyo3::{
    Bound, PyResult, Python,
    exceptions::PyRuntimeError,
    pyfunction,
    types::{PyDict, PyDictMethods},
};
use tmesh::spatialindex::ObjectIndex;
use tucanos::mesh::{autotag, transfer_tags};

/// Automatically tag the elements based on a feature angle
#[pyfunction]
pub fn autotag_3d<'py>(
    py: Python<'py>,
    msh: &mut PyBoundaryMesh3d,
    angle_deg: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let res = autotag(&mut msh.0, angle_deg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    for (k, v) in res {
        dict.set_item(k, to_numpy_1d(py, v.clone()))?;
    }
    Ok(dict)
}

/// Reset the face tags of other to match those in self
#[pyfunction]
pub fn transfer_tags_face_3d(msh: &PyBoundaryMesh3d, other: &mut PyMesh3d) {
    let tree = ObjectIndex::new(msh.0.clone());
    transfer_tags(&msh.0, &tree, &mut other.0);
}

/// Reset the element tags of other to match those in self
#[pyfunction]
pub fn transfer_tags_elem_3d(msh: &PyBoundaryMesh3d, other: &mut PyMesh3d) {
    let tree = ObjectIndex::new(msh.0.clone());
    transfer_tags(&msh.0, &tree, &mut other.0);
}

/// Automatically tag the elements based on a feature angle
#[pyfunction]
pub fn autotag_2d<'py>(
    py: Python<'py>,
    msh: &mut PyBoundaryMesh2d,
    angle_deg: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let res = autotag(&mut msh.0, angle_deg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    for (k, v) in res {
        dict.set_item(k, to_numpy_1d(py, v.clone()))?;
    }
    Ok(dict)
}

/// Reset the face tags of other to match those in self
#[pyfunction]
pub fn transfer_tags_face_2d(msh: &PyBoundaryMesh2d, other: &mut PyMesh2d) {
    let tree = ObjectIndex::new(msh.0.clone());
    transfer_tags(&msh.0, &tree, &mut other.0);
}

/// Reset the element tags of other to match those in self
#[pyfunction]
pub fn transfer_tags_elem_2d(msh: &PyBoundaryMesh2d, other: &mut PyBoundaryMesh2d) {
    let tree = ObjectIndex::new(msh.0.clone());
    transfer_tags(&msh.0, &tree, &mut other.0);
}
