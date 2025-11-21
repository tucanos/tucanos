#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::ref_as_ptr)]
use crate::{
    Idx,
    geometry::{LinearGeometry2d, LinearGeometry3d},
    mesh::{PyMesh2d, PyMesh3d},
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
#[cfg(not(feature = "metis"))]
use tmesh::mesh::partition::HilbertPartitioner;
#[cfg(feature = "metis")]
use tmesh::mesh::partition::{MetisPartitioner, MetisRecursive};
use tmesh::mesh::{GenericMesh, Tetrahedron, Triangle};
use tucanos::{
    mesh::MeshTopology,
    metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric, Metric, MetricField},
    remesher::{ParallelRemesher, ParallelRemesherParams},
};
#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub struct PyParallelRemesherParams {
    n_layers: u32,
    max_levels: u32,
    min_verts: usize,
}
impl PyParallelRemesherParams {
    const fn from(other: &ParallelRemesherParams) -> Self {
        Self {
            n_layers: other.n_layers,
            max_levels: other.max_levels,
            min_verts: other.min_verts,
        }
    }
    const fn to(&self) -> ParallelRemesherParams {
        ParallelRemesherParams::new(self.n_layers, self.max_levels, self.min_verts)
    }
}

#[pymethods]
impl PyParallelRemesherParams {
    #[new]
    const fn new(n_layers: u32, max_levels: u32, min_verts: usize) -> Self {
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
    ($name: ident, $dim: expr, $etype: ident, $metric: ident, $pymesh: ident, $geom: ident) => {
        #[doc = concat!("Parallel remesher for a meshes consisting of ", stringify!($etype),
                " in ", stringify!($dim), "D")]
        #[doc = concat!("using ", stringify!($metric),
                " as metric and a piecewise linear representation of the geometry")]
        #[pyclass]
        pub struct $name {
            #[cfg(not(feature = "metis"))]
            dd: ParallelRemesher<$dim, $etype<Idx>, GenericMesh<$dim, $etype<Idx>>, HilbertPartitioner>,
            #[cfg(feature = "metis")]
            dd: ParallelRemesher<$dim, $etype<Idx>, GenericMesh<$dim, $etype<Idx>>, MetisPartitioner<MetisRecursive>>,
        }

        #[doc = concat!("Create a parallel remesher from a ", stringify!($pymesh), " and a ",
                stringify!($metric) ," metric defined at the mesh vertices")]
        #[doc = concat!(
                    "A piecewise linear representation of the geometry is used, either from the ",
                    stringify!($geom), " given or otherwise from the mesh boundary.")]
        #[pymethods]
        impl $name {
            #[new]
            pub fn new(mesh: &$pymesh, n_partitions: usize) -> PyResult<Self> {
                let topo = MeshTopology::new(&mesh.0);
                let dd = ParallelRemesher::new(mesh.0.clone(), topo, n_partitions);
                if let Err(res) = dd {
                    return Err(PyRuntimeError::new_err(res.to_string()));
                }
                Ok(Self { dd: dd.unwrap() })
            }

            pub const fn set_debug(&mut self, debug: bool) {
                self.dd.set_debug(debug);
            }

            pub fn partitionned_mesh(&mut self) -> $pymesh {
                $pymesh(self.dd.partitionned_mesh().clone())
            }

            #[allow(clippy::too_many_arguments)]
            pub fn remesh<'py>(
                &mut self,
                py: Python<'py>,
                geometry: &$geom,
                m: PyReadonlyArray2<f64>,
                params: &PyRemesherParams,
                parallel_params: &PyParallelRemesherParams,
            ) -> PyResult<($pymesh, Bound<'py, PyArray2<f64>>, String)> {
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

                let (mesh, info, m) = py.detach(|| {
                    self.dd
                        .remesh(&m, &geometry.geom, params.to(), &parallel_params.to())
                        .unwrap()
                });

                let mesh = $pymesh(mesh);

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
                mesh: &$pymesh,
                m: PyReadonlyArray2<f64>,
            ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
                if m.shape()[0] != mesh.n_verts()  {
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
                let m = MetricField::new(&mesh.0, m);
                let q = m.qualities();
                let l = m.edge_lengths();

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
    PyMesh2d,
    LinearGeometry2d
);
create_parallel_remesher!(
    ParallelRemesher2dAniso,
    2,
    Triangle,
    AnisoMetric2d,
    PyMesh2d,
    LinearGeometry2d
);
create_parallel_remesher!(
    ParallelRemesher3dIso,
    3,
    Tetrahedron,
    IsoMetric3d,
    PyMesh3d,
    LinearGeometry3d
);
create_parallel_remesher!(
    ParallelRemesher3dAniso,
    3,
    Tetrahedron,
    AnisoMetric3d,
    PyMesh3d,
    LinearGeometry3d
);
