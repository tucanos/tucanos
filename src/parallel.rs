use crate::{
    geometry::{LinearGeometry2d, LinearGeometry3d},
    mesh::{Mesh22, Mesh33},
    to_numpy_1d, to_numpy_2d,
};
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods,
    types::PyType,
    Bound, PyResult, Python,
};
use tucanos::{
    mesh::{PartitionType, Tetrahedron, Triangle},
    metric::{AnisoMetric2d, AnisoMetric3d, IsoMetric, Metric},
    remesher::{ParallelRemesher, ParallelRemeshingParams, RemesherParams, SmoothingType},
    Idx,
};

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
            pub fn new(
                mesh: &$mesh,
                partition_type: &str,
                n_partitions: Idx,
            ) -> PyResult<Self> {

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
                Ok(Self {dd: dd.unwrap()})
            }

            pub fn set_debug(&mut self, debug: bool) {
                self.dd.set_debug(debug);
            }

            pub fn partitionned_mesh(&mut self) -> $mesh {
                $mesh{
                    mesh:self.dd.partitionned_mesh().clone()
                }
            }

            #[allow(clippy::too_many_arguments)]
            #[pyo3(signature = (geometry, m, num_iter=None, two_steps=None, split_max_iter=None,
                split_min_l_rel=None, split_min_l_abs=None, split_min_q_rel=None,
                split_min_q_abs=None, collapse_max_iter=None, collapse_max_l_rel=None,
                collapse_max_l_abs=None, collapse_min_q_rel=None, collapse_min_q_abs=None,
                swap_max_iter=None, swap_max_l_rel=None, swap_max_l_abs=None, swap_min_l_rel=None,
                swap_min_l_abs=None, smooth_iter=None, smooth_type=None, smooth_relax=None,
                smooth_keep_local_minima=None, max_angle=None, debug=None, n_layers=None,
                n_levels=None, min_verts=None))]
            pub fn remesh<'py>(&mut self,
                py: Python<'py>,
                geometry: &$geom,
                m: PyReadonlyArray2<f64>,
                num_iter:Option< u32>,
                two_steps: Option<bool>,
                split_max_iter:Option< u32>,
                split_min_l_rel:Option< f64>,
                split_min_l_abs:Option< f64>,
                split_min_q_rel:Option< f64>,
                split_min_q_abs:Option< f64>,
                collapse_max_iter:Option< u32>,
                collapse_max_l_rel:Option< f64>,
                collapse_max_l_abs:Option< f64>,
                collapse_min_q_rel:Option< f64>,
                collapse_min_q_abs:Option< f64>,
                swap_max_iter:Option< u32>,
                swap_max_l_rel:Option< f64>,
                swap_max_l_abs:Option< f64>,
                swap_min_l_rel:Option< f64>,
                swap_min_l_abs:Option< f64>,
                smooth_iter:Option< u32>,
                smooth_type: Option<&str>,
                smooth_relax: Option<PyReadonlyArray1<f64>>,
                smooth_keep_local_minima: Option<bool>,
                max_angle:Option< f64>,
                debug: Option<bool>,
                n_layers: Option<Idx>,
                n_levels: Option<Idx>,
                min_verts: Option<Idx>,
            ) -> PyResult<($mesh, Bound<'py, PyArray2<f64>>, String)> {

                if m.shape()[0] != self.dd.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != $metric::N as usize {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice()?;
                let m: Vec<_> = m.chunks($metric::N).map(|x| $metric::from_slice(x)).collect();

                let smooth_type = smooth_type.unwrap_or("laplacian");

                let smooth_type = if smooth_type == "laplacian" {
                    SmoothingType::Laplacian
                } else if smooth_type == "laplacian2" {
                    SmoothingType::Laplacian2
                } else if smooth_type == "nlopt" {
                    unreachable!()
                } else {
                    SmoothingType::Avro
                };

                let default_params = RemesherParams::default();

                let params = RemesherParams {
                    num_iter: num_iter.unwrap_or(default_params.num_iter),
                    two_steps: two_steps.unwrap_or(default_params.two_steps),
                    split_max_iter: split_max_iter.unwrap_or(default_params.split_max_iter),
                    split_min_l_rel: split_min_l_rel.unwrap_or(default_params.split_min_l_rel),
                    split_min_l_abs: split_min_l_abs.unwrap_or(default_params.split_min_l_abs),
                    split_min_q_rel: split_min_q_rel.unwrap_or(default_params.split_min_q_rel),
                    split_min_q_abs: split_min_q_abs.unwrap_or(default_params.split_min_q_abs),
                    collapse_max_iter: collapse_max_iter.unwrap_or(
                        default_params.collapse_max_iter),
                    collapse_max_l_rel: collapse_max_l_rel.unwrap_or(
                        default_params.collapse_max_l_rel),
                    collapse_max_l_abs: collapse_max_l_abs.unwrap_or(
                        default_params.collapse_max_l_abs),
                    collapse_min_q_rel: collapse_min_q_rel.unwrap_or(
                        default_params.collapse_min_q_rel),
                    collapse_min_q_abs: collapse_min_q_abs.unwrap_or(
                        default_params.collapse_min_q_abs),
                    swap_max_iter: swap_max_iter.unwrap_or(default_params.swap_max_iter),
                    swap_max_l_rel: swap_max_l_rel.unwrap_or(default_params.swap_max_l_rel),
                    swap_max_l_abs: swap_max_l_abs.unwrap_or(default_params.swap_max_l_abs),
                    swap_min_l_rel: swap_min_l_rel.unwrap_or(default_params.swap_min_l_rel),
                    swap_min_l_abs: swap_min_l_abs.unwrap_or(default_params.swap_min_l_abs),
                    smooth_iter: smooth_iter.unwrap_or(default_params.smooth_iter),
                    smooth_type,
                    smooth_relax: smooth_relax.map(|x| x.to_vec().unwrap()).unwrap_or(
                        default_params.smooth_relax),
                    smooth_keep_local_minima: smooth_keep_local_minima.unwrap_or(
                        default_params.smooth_keep_local_minima),
                    max_angle: max_angle.unwrap_or(default_params.max_angle),
                    debug: debug.unwrap_or(default_params.debug),
                };

                let dd_params = ParallelRemeshingParams::new(
                    n_layers.unwrap_or(2),
                    n_levels.unwrap_or(1),
                    min_verts.unwrap_or(0)
                );

                let (mesh, info, m) = py.allow_threads(||
                self.dd.remesh(&m, &geometry.geom, params, &dd_params).unwrap());

                let mesh = $mesh{mesh};

                let m = m.iter().flat_map(|m| m.into_iter()).collect::<Vec<_>>();

                Ok((mesh, to_numpy_2d(py, m, <$metric as Metric<$dim>>::N), info.to_json()))

            }

            /// Compute the element qualities & edge lengths
            #[classmethod]
            pub fn qualities_and_lengths<'py>(
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                mesh: &$mesh,
                m: PyReadonlyArray2<f64>,
            ) -> PyResult<(Bound<'py,PyArray1<f64>>, Bound<'py,PyArray1<f64>>)> {

                if m.shape()[0] != mesh.n_verts() as usize {
                    return Err(PyValueError::new_err("Invalid dimension 0"));
                }
                if m.shape()[1] != $metric::N as usize {
                    return Err(PyValueError::new_err("Invalid dimension 1"));
                }

                let m = m.as_slice()?;
                let m: Vec<_> = m.chunks($metric::N).map(|x| $metric::from_slice(x)).collect();

                let q = mesh.mesh.qualities(&m);
                let l = mesh.mesh.edge_lengths(&m).map_err(
                    |e| PyRuntimeError::new_err(e.to_string()))?;

                Ok((to_numpy_1d(py, q), to_numpy_1d(py, l)))
            }

        }
    }
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
