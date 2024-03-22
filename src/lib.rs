mod geometry;
mod mesh;
#[cfg(any(feature = "metis", feature = "scotch"))]
mod parallel;
mod remesher;

use numpy::{PyArray, PyArray1, PyArray2};
#[cfg(feature = "meshb")]
use pyo3::{pyfunction, wrap_pyfunction};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

fn to_numpy_1d<T: numpy::Element>(py: Python<'_>, vec: Vec<T>) -> &'_ PyArray1<T> {
    PyArray::from_vec(py, vec)
}

fn to_numpy_2d<T: numpy::Element>(py: Python<'_>, vec: Vec<T>, m: usize) -> &'_ PyArray2<T> {
    let n = vec.len();
    PyArray::from_vec(py, vec).reshape([n / m, m]).unwrap()
}

/// Read a solution stored in a .sol(b) file
#[pyfunction]
#[cfg(feature = "meshb")]
pub fn read_solb<'py>(py: Python<'py>, fname: &str) -> PyResult<&'py PyArray2<f64>> {
    use pyo3::exceptions::PyRuntimeError;

    let res = tucanos::meshb_io::read_solb(fname);
    match res {
        Ok((sol, m)) => Ok(to_numpy_2d(py, sol, m)),
        Err(err) => Err(PyRuntimeError::new_err(err.to_string())),
    }
}

/// Python bindings for pytucanos
#[pymodule]
#[pyo3(name = "_pytucanos")]
pub fn pytucanos(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<crate::mesh::Mesh33>()?;
    m.add_class::<crate::mesh::Mesh32>()?;
    m.add_class::<crate::mesh::Mesh31>()?;
    m.add_class::<crate::mesh::Mesh22>()?;
    m.add_class::<crate::mesh::Mesh21>()?;
    m.add_class::<crate::geometry::LinearGeometry2d>()?;
    m.add_class::<crate::geometry::LinearGeometry3d>()?;
    m.add_class::<crate::remesher::Remesher2dIso>()?;
    m.add_class::<crate::remesher::Remesher2dAniso>()?;
    m.add_class::<crate::remesher::Remesher3dIso>()?;
    m.add_class::<crate::remesher::Remesher3dAniso>()?;
    #[cfg(any(feature = "metis", feature = "scotch"))]
    m.add_class::<crate::parallel::ParallelRemesher2dIso>()?;
    #[cfg(any(feature = "metis", feature = "scotch"))]
    m.add_class::<crate::parallel::ParallelRemesher2dAniso>()?;
    #[cfg(any(feature = "metis", feature = "scotch"))]
    m.add_class::<crate::parallel::ParallelRemesher3dIso>()?;
    #[cfg(any(feature = "metis", feature = "scotch"))]
    m.add_class::<crate::parallel::ParallelRemesher3dAniso>()?;
    #[cfg(any(feature = "metis", feature = "scotch"))]
    m.add("HAVE_PARALLEL", true)?;
    #[cfg(not(any(feature = "metis", feature = "scotch")))]
    m.add("HAVE_PARALLEL", false)?;
    #[cfg(feature = "meshb")]
    m.add_function(wrap_pyfunction!(read_solb, m)?)?;
    #[cfg(not(feature = "meshb"))]
    m.add("HAVE_MESHB", false)?;
    #[cfg(feature = "meshb")]
    m.add("HAVE_MESHB", true)?;
    Ok(())
}
