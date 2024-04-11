#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::doc_markdown)]

use tucanos::{
    geometry::LinearGeometry,
    mesh::SimplexMesh,
    metric::{AnisoMetric3d, IsoMetric, Metric},
    remesher::{self, Remesher},
    topo_elems::{Tetrahedron, Triangle},
};

pub struct tucanos_remesher3diso_t {
    implem: Remesher<3, Tetrahedron, IsoMetric<3>>,
}

pub struct tucanos_remesher3daniso_t {
    implem: Remesher<3, Tetrahedron, AnisoMetric3d>,
}

pub struct tucanos_mesh33_t {
    implem: SimplexMesh<3, Tetrahedron>,
}

pub struct tucanos_mesh32_t {
    implem: SimplexMesh<3, Triangle>,
}

pub struct tucanos_geom3d_t {
    implem: LinearGeometry<3, Triangle>,
}

#[repr(C)]
pub struct tucanos_params_t {
    /// Number of collapse - split - swap - smooth loops
    pub num_iter: u32,
    /// Perform a first loop targetting only the longest edges
    pub two_steps: bool,
    /// Max. number of loops through the mesh edges during the split step
    pub split_max_iter: u32,
    /// Constraint the length of the newly created edges to be > split_min_l_rel * min(l) during split
    pub split_min_l_rel: f64,
    /// Constraint the length of the newly created edges to be > split_min_l_abs during split
    pub split_min_l_abs: f64,
    /// Constraint the quality of the newly created elements to be > split_min_q_rel * min(q) during split
    pub split_min_q_rel: f64,
    /// Constraint the quality of the newly created elements to be > split_min_q_abs during split
    pub split_min_q_abs: f64,
    /// Max. number of loops through the mesh edges during the collapse step
    pub collapse_max_iter: u32,
    /// Constraint the length of the newly created edges to be < collapse_max_l_rel * max(l) during collapse
    pub collapse_max_l_rel: f64,
    /// Constraint the length of the newly created edges to be < collapse_max_l_abs during collapse
    pub collapse_max_l_abs: f64,
    /// Constraint the quality of the newly created elements to be > collapse_min_q_rel * min(q) during collapse
    pub collapse_min_q_rel: f64,
    /// Constraint the quality of the newly created elements to be > collapse_min_q_abs during collapse
    pub collapse_min_q_abs: f64,
    /// Max. number of loops through the mesh edges during the swap step
    pub swap_max_iter: u32,
    /// Constraint the length of the newly created edges to be < swap_max_l_rel * max(l) during swap
    pub swap_max_l_rel: f64,
    /// Constraint the length of the newly created edges to be < swap_max_l_abs during swap
    pub swap_max_l_abs: f64,
    /// Constraint the length of the newly created edges to be > swap_min_l_rel * min(l) during swap
    pub swap_min_l_rel: f64,
    /// Constraint the length of the newly created edges to be > swap_min_l_abs during swap
    pub swap_min_l_abs: f64,
    /// Number of smoothing steps
    pub smooth_iter: u32,
    /// Don't smooth vertices that are a local metric minimum
    pub smooth_keep_local_minima: bool,
    /// Max angle between the normals of the new faces and the geometry (in degrees)
    pub max_angle: f64,
    /// Debug mode
    pub debug: bool,
}

impl From<&tucanos_params_t> for remesher::RemesherParams {
    #[allow(clippy::field_reassign_with_default)]
    fn from(params: &tucanos_params_t) -> Self {
        let mut rparams = Self::default();
        rparams.num_iter = params.num_iter;
        rparams.two_steps = params.two_steps;
        rparams.split_max_iter = params.split_max_iter;
        rparams.split_min_l_rel = params.split_min_l_rel;
        rparams.split_min_l_abs = params.split_min_l_abs;
        rparams.split_min_q_rel = params.split_min_q_rel;
        rparams.split_min_q_abs = params.split_min_q_abs;
        rparams.collapse_max_iter = params.collapse_max_iter;
        rparams.collapse_max_l_rel = params.collapse_max_l_rel;
        rparams.collapse_max_l_abs = params.collapse_max_l_abs;
        rparams.collapse_min_q_rel = params.collapse_min_q_rel;
        rparams.collapse_min_q_abs = params.collapse_min_q_abs;
        rparams.swap_max_iter = params.swap_max_iter;
        rparams.swap_max_l_rel = params.swap_max_l_rel;
        rparams.swap_max_l_abs = params.swap_max_l_abs;
        rparams.swap_min_l_rel = params.swap_min_l_rel;
        rparams.swap_min_l_abs = params.swap_min_l_abs;
        rparams.smooth_iter = params.smooth_iter;
        rparams.smooth_keep_local_minima = params.smooth_keep_local_minima;
        rparams.max_angle = params.max_angle;
        rparams.debug = params.debug;
        rparams
    }
}

#[no_mangle]
pub unsafe extern "C" fn tucanos_params_init(params: *mut tucanos_params_t) {
    let params = &mut *params;
    let default = remesher::RemesherParams::default();
    params.num_iter = default.num_iter;
    params.two_steps = default.two_steps;
    params.split_max_iter = default.split_max_iter;
    params.split_min_l_rel = default.split_min_l_rel;
    params.split_min_l_abs = default.split_min_l_abs;
    params.split_min_q_rel = default.split_min_q_rel;
    params.split_min_q_abs = default.split_min_q_abs;
    params.collapse_max_iter = default.collapse_max_iter;
    params.collapse_max_l_rel = default.collapse_max_l_rel;
    params.collapse_max_l_abs = default.collapse_max_l_abs;
    params.collapse_min_q_rel = default.collapse_min_q_rel;
    params.collapse_min_q_abs = default.collapse_min_q_abs;
    params.swap_max_iter = default.swap_max_iter;
    params.swap_max_l_rel = default.swap_max_l_rel;
    params.swap_max_l_abs = default.swap_max_l_abs;
    params.swap_min_l_rel = default.swap_min_l_rel;
    params.swap_min_l_abs = default.swap_min_l_abs;
    params.smooth_iter = default.smooth_iter;
    params.smooth_keep_local_minima = default.smooth_keep_local_minima;
    params.max_angle = default.max_angle;
    params.debug = default.debug;
}

/// @brief Create a geometry for a tucanos_mesh33_t
///
/// @param mesh The mesh for which the geometry is to be created
/// @param boundary A mesh representing the boundary faces of `mesh`. This function consume and free the boundary.
#[no_mangle]
pub unsafe extern "C" fn tucanos_geom3d_new(
    mesh: *const tucanos_mesh33_t,
    boundary: *mut tucanos_mesh32_t,
) -> *mut tucanos_geom3d_t {
    let mesh = &(*mesh).implem;
    let boundary = Box::from_raw(boundary).implem;
    LinearGeometry::new(mesh, boundary).map_or(std::ptr::null_mut(), |implem| {
        Box::into_raw(Box::new(tucanos_geom3d_t { implem }))
    })
}

unsafe fn new_metric<const D: usize, MT: Metric<D>>(
    metric: *const f64,
    num_points: u32,
) -> impl ExactSizeIterator<Item = MT> {
    let metric = std::slice::from_raw_parts(metric, num_points as usize * MT::N);
    metric.chunks(MT::N).map(MT::from_slice)
}

/// @brief Create a remesher from a mesh, an isotropic metric and a geometry
#[no_mangle]
pub unsafe extern "C" fn tucanos_remesher3diso_new(
    mesh: *const tucanos_mesh33_t,
    metric: *const f64,
    geom: *const tucanos_geom3d_t,
) -> *mut tucanos_remesher3diso_t {
    let mesh = &(*mesh).implem;
    let metric = new_metric(metric, mesh.n_verts());
    let geom = &(*geom).implem;
    Remesher::new_with_iter(mesh, metric, geom).map_or(std::ptr::null_mut(), |implem| {
        Box::into_raw(Box::new(tucanos_remesher3diso_t { implem }))
    })
}

#[no_mangle]
pub unsafe extern "C" fn tucanos_remesher3diso_tomesh(
    remesher: *mut tucanos_remesher3diso_t,
    only_bdy_faces: bool,
) -> *mut tucanos_mesh33_t {
    let remesher = &mut (*remesher).implem;
    let implem = remesher.to_mesh(only_bdy_faces);
    Box::into_raw(Box::new(tucanos_mesh33_t { implem }))
}

/// @brief Create a remesher from a mesh, an anisotropic metric and a geometry
#[no_mangle]
pub unsafe extern "C" fn tucanos_remesher3daniso_new(
    mesh: *const tucanos_mesh33_t,
    metric: *const f64,
    geom: *const tucanos_geom3d_t,
) -> *mut tucanos_remesher3daniso_t {
    let mesh = &(*mesh).implem;
    let metric = new_metric(metric, mesh.n_verts());
    let geom = &(*geom).implem;
    Remesher::new_with_iter(mesh, metric, geom).map_or(std::ptr::null_mut(), |implem| {
        Box::into_raw(Box::new(tucanos_remesher3daniso_t { implem }))
    })
}

#[no_mangle]
pub unsafe extern "C" fn tucanos_remesher3diso_remesh(
    remesher: *mut tucanos_remesher3diso_t,
    params: *const tucanos_params_t,
    geom: *const tucanos_geom3d_t,
) -> bool {
    let remesher = &mut (*remesher).implem;
    let geom = &(*geom).implem;
    let params = &*params;
    remesher.remesh(params.into(), geom).is_ok()
}

#[no_mangle]
pub unsafe extern "C" fn tucanos_remesher3daniso_remesh(
    remesher: *mut tucanos_remesher3daniso_t,
    params: *const tucanos_params_t,
    geom: *const tucanos_geom3d_t,
) -> bool {
    let remesher = &mut (*remesher).implem;
    let geom = &(*geom).implem;
    let params = &*params;
    remesher.remesh(params.into(), geom).is_ok()
}

#[no_mangle]
pub unsafe extern "C" fn tucanos_remesher3daniso_tomesh(
    remesher: *mut tucanos_remesher3daniso_t,
    only_bdy_faces: bool,
) -> *mut tucanos_mesh33_t {
    let remesher = &mut (*remesher).implem;
    let implem = remesher.to_mesh(only_bdy_faces);
    Box::into_raw(Box::new(tucanos_mesh33_t { implem }))
}

#[no_mangle]
pub extern "C" fn tucanos_mesh33_new(
    num_verts: usize,
    verts: *const f64,
    num_elements: usize,
    elems: *const u32,
    tags: *const i16,
    num_faces: usize,
    faces: *const u32,
    ftags: *const i16,
) -> *mut tucanos_mesh33_t {
    let mut m = SimplexMesh::new_with_vector(
        (verts, num_verts).into(),
        (elems, num_elements).into(),
        (tags, num_elements).into(),
        (faces, num_faces).into(),
        (ftags, num_faces).into(),
    );
    m.compute_topology();
    Box::into_raw(Box::new(tucanos_mesh33_t { implem: m }))
}

#[no_mangle]
pub unsafe extern "C" fn tucanos_mesh33_num_verts(m: *const tucanos_mesh33_t) -> u32 {
    (*m).implem.n_verts()
}

#[no_mangle]
pub unsafe extern "C" fn tucanos_mesh33_num_elems(m: *const tucanos_mesh33_t) -> u32 {
    (*m).implem.n_elems()
}

#[no_mangle]
pub unsafe extern "C" fn tucanos_mesh33_verts(
    m: *const tucanos_mesh33_t,
    out: *mut f64,
    first: u32,
    last: u32,
) {
    let out = std::slice::from_raw_parts_mut(out, (last - first) as usize);
    let m = &((*m).implem);
    let mut k = 0;
    for i in first..last {
        out[k..k + 3].copy_from_slice(m.vert(i).as_slice());
        k += 3;
    }
}

#[no_mangle]
pub unsafe extern "C" fn tucanos_mesh33_elems(
    m: *const tucanos_mesh33_t,
    out: *mut u32,
    first: u32,
    last: u32,
) {
    let out = std::slice::from_raw_parts_mut(out, (last - first) as usize);
    let m = &((*m).implem);
    let mut k = 0;
    for i in first..last {
        for v in m.elem(i) {
            out[k] = v;
            k += 1;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn tucanos_mesh33_boundary(
    m: *const tucanos_mesh33_t,
) -> *mut tucanos_mesh32_t {
    let m = &*m;
    let (implem, _) = m.implem.boundary();
    Box::into_raw(Box::new(tucanos_mesh32_t { implem }))
}

#[no_mangle]
pub unsafe extern "C" fn tucanos_mesh33_delete(m: *mut tucanos_mesh33_t) {
    let _ = Box::from_raw(m);
}
