//! Python bindings for tmesh

mod dual;
mod extruded;
mod mesh;
mod poly;

use pyo3::{
    Bound, PyResult, Python, pymodule,
    types::{PyModule, PyModuleMethods},
};

/// Python bindings for tmesh
#[pymodule]
#[pyo3(name = "pytmesh")]
pub fn pymeshb(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<crate::mesh::PyMesh2d>()?;
    m.add_class::<crate::mesh::PyBoundaryMesh2d>()?;
    m.add_class::<crate::mesh::PyMesh3d>()?;
    m.add_class::<crate::mesh::PyBoundaryMesh3d>()?;
    m.add_class::<crate::mesh::PyPartitionerType>()?;
    m.add_class::<crate::dual::PyDualType>()?;
    m.add_class::<crate::dual::PyDualMesh2d>()?;
    m.add_class::<crate::dual::PyDualMesh3d>()?;
    m.add_class::<crate::poly::PyPolyMeshType>()?;
    m.add_class::<crate::poly::PyPolyMesh2d>()?;
    m.add_class::<crate::poly::PyPolyMesh3d>()?;
    m.add_class::<crate::extruded::PyExtrudedMesh2d>()?;
    #[cfg(not(feature = "metis"))]
    m.add("HAVE_METIS", false)?;
    #[cfg(feature = "metis")]
    m.add("HAVE_METIS", true)?;
    Ok(())
}
