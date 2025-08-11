//! Python bindings for tmesh

mod dual;
mod extruded;
mod mesh;
mod poly;

pub use mesh::PyPartitionerType;
use pyo3::{
    Bound, PyResult, Python, pymodule,
    types::{PyModule, PyModuleMethods},
};

/// Python bindings for tmesh
#[pymodule]
#[pyo3(name = "pytmesh")]
pub fn pymeshb(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<mesh::PyMesh2d>()?;
    m.add_class::<mesh::PyBoundaryMesh2d>()?;
    m.add_class::<mesh::PyMesh3d>()?;
    m.add_class::<mesh::PyBoundaryMesh3d>()?;
    m.add_class::<PyPartitionerType>()?;
    m.add_class::<dual::PyDualType>()?;
    m.add_class::<dual::PyDualMesh2d>()?;
    m.add_class::<dual::PyDualMesh3d>()?;
    m.add_class::<poly::PyPolyMeshType>()?;
    m.add_class::<poly::PyPolyMesh2d>()?;
    m.add_class::<poly::PyPolyMesh3d>()?;
    m.add_class::<extruded::PyExtrudedMesh2d>()?;
    #[cfg(not(feature = "metis"))]
    m.add("HAVE_METIS", false)?;
    #[cfg(feature = "metis")]
    m.add("HAVE_METIS", true)?;
    Ok(())
}
