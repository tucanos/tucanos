mod geometry;
mod mesh;
mod parallel;
mod remesher;
use numpy::{PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::{
    Bound, PyResult, Python,
    exceptions::PyRuntimeError,
    pyfunction, pymodule,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction,
};

fn to_numpy_1d<T: numpy::Element>(py: Python<'_>, vec: Vec<T>) -> Bound<'_, PyArray1<T>> {
    PyArray::from_vec(py, vec)
}

fn to_numpy_2d<T: numpy::Element>(py: Python<'_>, vec: Vec<T>, m: usize) -> Bound<'_, PyArray2<T>> {
    let n = vec.len();
    PyArray::from_vec(py, vec).reshape([n / m, m]).unwrap()
}

/// Get the current thread affinity
#[pyfunction]
pub fn get_thread_affinity(py: Python<'_>) -> PyResult<Bound<'_, PyArray1<usize>>> {
    let bound_cores = affinity::get_thread_affinity();
    if let Err(err) = bound_cores {
        Err(PyRuntimeError::new_err(err.to_string()))
    } else {
        Ok(to_numpy_1d(py, bound_cores.unwrap()))
    }
}

/// Set the thread affinity and return the number of rayon threads
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn set_thread_affinity(cores: PyReadonlyArray1<usize>) -> PyResult<usize> {
    let tmp = cores.as_slice();
    if let Err(err) = tmp {
        Err(PyRuntimeError::new_err(err.to_string()))
    } else {
        let res = affinity::set_thread_affinity(tmp.unwrap());
        if let Err(err) = res {
            Err(PyRuntimeError::new_err(err.to_string()))
        } else {
            Ok(rayon::current_num_threads())
        }
    }
}

/// Python bindings for pytucanos
#[pymodule]
#[pyo3(name = "pytucanos")]
pub fn pytucanos(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(get_thread_affinity, m)?)?;
    m.add_function(wrap_pyfunction!(set_thread_affinity, m)?)?;
    m.add_class::<pytmesh::PyPartitionerType>()?;
    m.add_class::<mesh::Mesh33>()?;
    m.add_class::<mesh::Mesh32>()?;
    m.add_class::<mesh::Mesh31>()?;
    m.add_class::<mesh::Mesh22>()?;
    m.add_class::<mesh::Mesh21>()?;
    m.add_class::<geometry::LinearGeometry2d>()?;
    m.add_class::<geometry::LinearGeometry3d>()?;
    m.add_class::<remesher::PyCollapseParams>()?;
    m.add_class::<remesher::PySplitParams>()?;
    m.add_class::<remesher::PySwapParams>()?;
    m.add_class::<remesher::PySmoothParams>()?;
    m.add_class::<remesher::PySmoothingMethod>()?;
    m.add_class::<remesher::PyRemeshingStep>()?;
    m.add_class::<remesher::PyRemesherParams>()?;
    m.add_class::<remesher::Remesher2dIso>()?;
    m.add_class::<remesher::Remesher2dAniso>()?;
    m.add_class::<remesher::Remesher3dIso>()?;
    m.add_class::<remesher::Remesher3dAniso>()?;
    m.add_class::<parallel::PyParallelRemesherParams>()?;
    m.add_class::<parallel::ParallelRemesher2dIso>()?;
    m.add_class::<parallel::ParallelRemesher2dAniso>()?;
    m.add_class::<parallel::ParallelRemesher3dIso>()?;
    m.add_class::<parallel::ParallelRemesher3dAniso>()?;
    #[cfg(not(feature = "metis"))]
    m.add("HAVE_METIS", false)?;
    #[cfg(feature = "metis")]
    m.add("HAVE_METIS", true)?;
    Ok(())
}
