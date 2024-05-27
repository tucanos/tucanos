mod geometry;
mod mesh;
#[cfg(any(feature = "metis", feature = "scotch"))]
mod parallel;
mod remesher;
use numpy::{PyArray, PyArray1, PyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

fn to_numpy_1d<T: numpy::Element>(py: Python<'_>, vec: Vec<T>) -> &'_ PyArray1<T> {
    PyArray::from_vec(py, vec)
}

fn to_numpy_2d<T: numpy::Element>(py: Python<'_>, vec: Vec<T>, m: usize) -> &'_ PyArray2<T> {
    let n = vec.len();
    PyArray::from_vec(py, vec).reshape([n / m, m]).unwrap()
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
    #[cfg(not(any(feature = "metis", feature = "scotch")))]
    m.add("HAVE_PARALLEL", false)?;
    #[cfg(any(feature = "metis", feature = "scotch"))]
    m.add("HAVE_PARALLEL", true)?;
    #[cfg(not(feature = "metis"))]
    m.add("HAVE_METIS", false)?;
    #[cfg(feature = "metis")]
    m.add("HAVE_METIS", true)?;
    #[cfg(not(feature = "scotch"))]
    m.add("HAVE_SCOTCH", false)?;
    #[cfg(feature = "scotch")]
    m.add("HAVE_SCOTCH", true)?;
    #[cfg(not(feature = "libmeshb"))]
    m.add("HAVE_LIBMESHB", false)?;
    #[cfg(feature = "libmeshb")]
    m.add("HAVE_LIBMESHB", true)?;
    Ok(())
}
