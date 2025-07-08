use crate::{
    geometry::{LinearGeometry2d, LinearGeometry3d},
    mesh::{Mesh22, Mesh33},
    remesher::PyRemesherParams,
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
    mesh::{PartitionType, Tetrahedron, Triangle},
    metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric, Metric},
    remesher::{ParallelRemesher, ParallelRemesherParams},
};

#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub struct PyParallelRemesherParams {
    n_layers: Idx,
    max_levels: Idx,
    min_verts: Idx,
}
impl PyParallelRemesherParams {
    pub fn from(other: &ParallelRemesherParams) -> Self {
        Self {
            n_layers: other.n_layers,
            max_levels: other.max_levels,
            min_verts: other.min_verts,
        }
    }
    pub fn to(&self) -> ParallelRemesherParams {
        ParallelRemesherParams::new(self.n_layers, self.max_levels, self.min_verts)
    }
}

#[pymethods]
impl PyParallelRemesherParams {
    #[new]
    pub fn new(n_layers: Idx, max_levels: Idx, min_verts: Idx) -> Self {
        Self {
            n_layers,
            max_levels,
            min_verts,
        }
    }

    #[classmethod]
    pub fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self::from(&ParallelRemesherParams::default())
    }
}

macro_rules! create_parallel_remesher {
    ($name: ident, $dim: expr, $etype: ident, $metric: ident, $mesh: ident, $geom: ident) => {
        #[doc = concat!("Parallel remesher for a meshes consisting of ", stringify!($etype),
                " in ", stringify!($dim), "D")]
        #[doc = concat!("using ", stringify!($metric),
                " as metric and a piecewise linear representation of the geometry")]
        #[pyclass]
        pub struct $name {
            dd: ParallelRemesher<$dim, $etype>,
        }

        #[doc = concat!("Create a parallel remesher from a ", stringify!($mesh), " and a ",
                stringify!($metric) ," metric defined at the mesh vertices")]
        #[doc = concat!(
                    "A piecewise linear representation of the geometry is used, either from the ",
                    stringify!($geom), " given or otherwise from the mesh boundary.")]
        #[pymethods]
        impl $name {
            #[new]
            pub fn new(mesh: &$mesh, partition_type: &str, n_partitions: Idx) -> PyResult<Self> {
                let partition_type = if partition_type == "scotch" {
                    PartitionType::Scotch(n_partitions)
                } else if partition_type == "metis_kway" {
                    PartitionType::MetisKWay(n_partitions)
                } else if partition_type == "metis_recursive" {
                    PartitionType::MetisRecursive(n_partitions)
                } else if partition_type == "hilbert" {
                    PartitionType::Hilbert(n_partitions)
                } else {
                    return Err(PyValueError::new_err(
        "Invalid partition type: allowed values are scotch, metis_kway, metis_recursive"));
                };

                let dd = ParallelRemesher::new(mesh.mesh.clone(), partition_type);
                if let Err(res) = dd {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(Self { dd: dd.unwrap() })
            }

            pub const fn set_debug(&mut self, debug: bool) {
                self.dd.set_debug(debug);
            }

            pub fn partitionned_mesh(&mut self) -> $mesh {
                $mesh {
                    mesh: self.dd.partitionned_mesh().clone(),
                }
            }

            #[allow(clippy::too_many_arguments)]
            pub fn remesh<'py>(
                &mut self,
                py: Python<'py>,
                geometry: &$geom,
                m: PyReadonlyArray2<f64>,
                params: &PyRemesherParams,
                parallel_params: &PyParallelRemesherParams,
            ) -> PyResult<($mesh, Bound<'py, PyArray2<f64>>, String)> {
                if m.shape()[0] != self.dd.n_verts() as usize {
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

                let (mesh, info, m) = py.allow_threads(|| {
                    self.dd
                        .remesh(&m, &geometry.geom, params.to(), &parallel_params.to())
                        .unwrap()
                });

                let mesh = $mesh { mesh };

                let m = m.iter().flat_map(|m| m.into_iter()).collect::<Vec<_>>();

                Ok((
                    mesh,
                    to_numpy_2d(py, m, <$metric as Metric<$dim>>::N),
                    info.to_json(),
                ))
            }

            /// Compute the element qualities & edge lengths
            #[classmethod]
            pub fn qualities_and_lengths<'py>(
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
            ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
                if m.shape()[0] != mesh.n_verts() as usize {
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

                let q = mesh.mesh.qualities(&m);
                let l = mesh
                    .mesh
                    .edge_lengths(&m)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

                Ok((to_numpy_1d(py, q), to_numpy_1d(py, l)))
            }
        }
    };
}

type IsoMetric2d = IsoMetric<2>;
type IsoMetric3d = IsoMetric<3>;
create_parallel_remesher!(
    ParallelRemesher2dIso,
    2,
    Triangle,
    IsoMetric2d,
    Mesh22,
    LinearGeometry2d
);
create_parallel_remesher!(
    ParallelRemesher2dAniso,
    2,
    Triangle,
    AnisoMetric2d,
    Mesh22,
    LinearGeometry2d
);
create_parallel_remesher!(
    ParallelRemesher3dIso,
    3,
    Tetrahedron,
    IsoMetric3d,
    Mesh33,
    LinearGeometry3d
);
create_parallel_remesher!(
    ParallelRemesher3dAniso,
    3,
    Tetrahedron,
    AnisoMetric3d,
    Mesh33,
    LinearGeometry3d
);
