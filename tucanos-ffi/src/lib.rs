#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::doc_markdown)]
use log::warn;
use tmesh::mesh::{GenericMesh, Mesh, Simplex, Tetrahedron, Triangle};
use tucanos::{
    geometry::MeshedGeometry,
    mesh::MeshTopology,
    metric::{AnisoMetric3d, IsoMetric, Metric},
    remesher::{Remesher, RemesherParams},
};

mod ffi2d;

#[cfg(feature = "32bit-ints")]
pub type Idx = u32;
#[cfg(not(feature = "32bit-ints"))]
pub type Idx = usize;

pub struct tucanos_remesher3diso_t {
    implem: Remesher<3, Tetrahedron<Idx>, IsoMetric<3>>,
}

pub struct tucanos_remesher3daniso_t {
    implem: Remesher<3, Tetrahedron<Idx>, AnisoMetric3d>,
}

pub struct tucanos_mesh33_t {
    implem: GenericMesh<3, Tetrahedron<Idx>>,
}

pub struct tucanos_mesh32_t {
    implem: GenericMesh<3, Triangle<Idx>>,
}

pub struct tucanos_geom3d_t {
    implem: MeshedGeometry<3, GenericMesh<3, Triangle<Idx>>>,
}

#[cfg(feature = "64bit-tags")]
pub type tucanos_tag_t = i64;
#[cfg(feature = "32bit-tags")]
pub type tucanos_tag_t = i32;
#[cfg(not(any(feature = "32bit-tags", feature = "64bit-tags")))]
pub type tucanos_tag_t = i16;
#[cfg(feature = "32bit-ints")]
pub type tucanos_int_t = u32;
#[cfg(not(feature = "32bit-ints"))]
pub type tucanos_int_t = usize;

/// @brief Create a geometry for a tucanos_mesh33_t
///
/// @param mesh The mesh for which the geometry is to be created
/// @param boundary A mesh representing the boundary faces of `mesh`. This function consume and free the boundary.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_geom3d_new(
    mesh: *mut tucanos_mesh33_t,
    boundary: *mut tucanos_mesh32_t,
) -> *mut tucanos_geom3d_t {
    unsafe {
        let mesh = &mut (*mesh).implem;
        let boundary = Box::from_raw(boundary).implem;
        let topo = MeshTopology::new(mesh);
        MeshedGeometry::new(mesh, &topo, boundary).map_or(std::ptr::null_mut(), |implem| {
            Box::into_raw(Box::new(tucanos_geom3d_t { implem }))
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_geom3d_delete(geom: *mut tucanos_geom3d_t) {
    unsafe {
        let _ = Box::from_raw(geom);
    }
}

unsafe fn new_metric<const D: usize, MT: Metric<D>>(
    metric: *const f64,
    num_points: usize,
) -> impl ExactSizeIterator<Item = MT> {
    unsafe {
        let metric = std::slice::from_raw_parts(metric, num_points * MT::N);
        metric.chunks(MT::N).map(MT::from_slice)
    }
}

/// @brief Create a remesher from a mesh, an isotropic metric and a geometry
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_remesher3diso_new(
    mesh: *const tucanos_mesh33_t,
    metric: *const f64,
    geom: *const tucanos_geom3d_t,
) -> *mut tucanos_remesher3diso_t {
    unsafe {
        let mesh = &(*mesh).implem;
        let metric = new_metric(metric, mesh.n_verts());
        let geom = &(*geom).implem;
        let topo = MeshTopology::new(mesh);
        match Remesher::new_with_iter(mesh, &topo, metric, geom) {
            Ok(implem) => Box::into_raw(Box::new(tucanos_remesher3diso_t { implem })),
            Err(e) => {
                warn!("{e:?}");
                std::ptr::null_mut()
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_remesher3diso_delete(remesher: *mut tucanos_remesher3diso_t) {
    unsafe {
        let _ = Box::from_raw(remesher);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_remesher3diso_tomesh(
    remesher: *mut tucanos_remesher3diso_t,
    only_bdy_faces: bool,
) -> *mut tucanos_mesh33_t {
    unsafe {
        let remesher = &mut (*remesher).implem;
        let implem = remesher.to_mesh(only_bdy_faces);
        Box::into_raw(Box::new(tucanos_mesh33_t { implem }))
    }
}

/// @brief Create a remesher from a mesh, an anisotropic metric and a geometry
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_remesher3daniso_new(
    mesh: *const tucanos_mesh33_t,
    metric: *const f64,
    geom: *const tucanos_geom3d_t,
) -> *mut tucanos_remesher3daniso_t {
    unsafe {
        let mesh = &(*mesh).implem;
        let metric = new_metric(metric, mesh.n_verts());
        let geom = &(*geom).implem;
        let topo = MeshTopology::new(mesh);
        match Remesher::new_with_iter(mesh, &topo, metric, geom) {
            Ok(implem) => Box::into_raw(Box::new(tucanos_remesher3daniso_t { implem })),
            Err(e) => {
                warn!("{e:?}");
                std::ptr::null_mut()
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_remesher3daniso_delete(remesher: *mut tucanos_remesher3daniso_t) {
    unsafe {
        let _ = Box::from_raw(remesher);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_remesher3diso_remesh(
    remesher: *mut tucanos_remesher3diso_t,
    geom: *const tucanos_geom3d_t,
) -> bool {
    unsafe {
        assert!(!remesher.is_null());
        assert!(!geom.is_null());
        let remesher = &mut (*remesher).implem;
        let geom = &(*geom).implem;
        remesher.remesh(&RemesherParams::default(), geom).is_ok()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_remesher3daniso_remesh(
    remesher: *mut tucanos_remesher3daniso_t,
    geom: *const tucanos_geom3d_t,
) -> bool {
    unsafe {
        assert!(!remesher.is_null());
        assert!(!geom.is_null());
        let remesher = &mut (*remesher).implem;
        let geom = &(*geom).implem;
        remesher.remesh(&RemesherParams::default(), geom).is_ok()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_remesher3daniso_tomesh(
    remesher: *mut tucanos_remesher3daniso_t,
    only_bdy_faces: bool,
) -> *mut tucanos_mesh33_t {
    unsafe {
        let remesher = &mut (*remesher).implem;
        let implem = remesher.to_mesh(only_bdy_faces);
        Box::into_raw(Box::new(tucanos_mesh33_t { implem }))
    }
}

/// Create a new 3D mesh containing tetrahedrons
///
/// The numbering convention of the elements is the one Of VTK or CGNS:
/// - <https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html>
/// - <https://cgns.github.io/standard/SIDS/convention.html>
#[unsafe(no_mangle)]
pub extern "C" fn tucanos_mesh33_new(
    num_verts: usize,
    verts: *const f64,
    num_elements: usize,
    elems: *const tucanos_int_t,
    tags: *const tucanos_tag_t,
    num_faces: usize,
    faces: *const tucanos_int_t,
    ftags: *const tucanos_tag_t,
) -> *mut tucanos_mesh33_t {
    let implem = GenericMesh::new(
        (verts, num_verts).into(),
        (elems, num_elements).into(),
        (tags, num_elements).into(),
        (faces, num_faces).into(),
        (ftags, num_faces).into(),
    );
    Box::into_raw(Box::new(tucanos_mesh33_t { implem }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_mesh33_num_verts(m: *const tucanos_mesh33_t) -> usize {
    unsafe { (*m).implem.n_verts() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_mesh33_num_elems(m: *const tucanos_mesh33_t) -> usize {
    unsafe { (*m).implem.n_elems() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_mesh33_verts(
    m: *const tucanos_mesh33_t,
    out: *mut f64,
    first: usize,
    last: usize,
) {
    unsafe {
        let out = std::slice::from_raw_parts_mut(out, 3 * (last - first));
        let m = &((*m).implem);
        let mut k = 0;
        for i in first..last {
            out[k..k + 3].copy_from_slice(m.vert(i).as_slice());
            k += 3;
        }
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::useless_conversion)]
pub unsafe extern "C" fn tucanos_mesh33_elems(
    m: *const tucanos_mesh33_t,
    out: *mut tucanos_int_t,
    first: usize,
    last: usize,
) {
    unsafe {
        let m = &((*m).implem);
        let slice_size = (last - first) * Tetrahedron::<Idx>::N_VERTS;
        let out = std::slice::from_raw_parts_mut(out, slice_size);
        let mut k = 0;
        for i in first..last {
            for v in m.elem(i) {
                out[k] = v.try_into().unwrap();
                k += 1;
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_mesh33_boundary(
    m: *const tucanos_mesh33_t,
) -> *mut tucanos_mesh32_t {
    unsafe {
        let m = &*m;
        let (implem, _) = m.implem.boundary();
        Box::into_raw(Box::new(tucanos_mesh32_t { implem }))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_mesh33_delete(m: *mut tucanos_mesh33_t) {
    unsafe {
        let _ = Box::from_raw(m);
    }
}

/// @brief Enable Rust logger
///
/// See <https://docs.rs/env_logger/latest/env_logger/#enabling-logging> for details
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tucanos_init_log() {
    tucanos::init_log("warn");
}
