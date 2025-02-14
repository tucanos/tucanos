use tucanos_ffi_test::*;

#[test]
fn iso3d() {
    let vertices = [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.];
    let faces = [0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3];
    let metric = [0.1; 4];
    unsafe {
        // Create a mesh with a single tetrahedron
        let mesh = tucanos_mesh33_new(
            4,
            vertices.as_ptr(),
            1,
            [0, 1, 2, 3].as_ptr(),
            [1].as_ptr(),
            4,
            faces.as_ptr(),
            [1; 4].as_ptr(),
        );
        let boundary = tucanos_mesh33_boundary(mesh);
        let geom = tucanos_geom3d_new(mesh, boundary);
        assert!(!geom.is_null());
        let remesher = tucanos_remesher3diso_new(mesh, metric.as_ptr(), geom);
        let mut params: tucanos_params_t = std::mem::zeroed();
        tucanos_params_init(&mut params);
        tucanos_remesher3diso_remesh(remesher, &params, geom);
        tucanos_mesh33_delete(mesh);
        let mesh = tucanos_remesher3diso_tomesh(remesher, false);
        let num_verts = tucanos_mesh33_num_verts(mesh);
        assert_eq!(num_verts, 52);
        tucanos_mesh33_delete(mesh);
    }
}

#[test]
fn aniso3d() {
    let vertices = [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.];
    let faces = [0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3];
    let metric: Vec<_> = (0..4)
        .flat_map(|_| [10., 20., 15., 0., 0., 0.].into_iter())
        .collect();
    unsafe {
        // Create a mesh with a single tetrahedron
        let mesh = tucanos_mesh33_new(
            4,
            vertices.as_ptr(),
            1,
            [0, 1, 2, 3].as_ptr(),
            [1].as_ptr(),
            4,
            faces.as_ptr(),
            [1; 4].as_ptr(),
        );
        let boundary = tucanos_mesh33_boundary(mesh);
        let geom = tucanos_geom3d_new(mesh, boundary);
        assert!(!geom.is_null());
        let remesher = tucanos_remesher3daniso_new(mesh, metric.as_ptr(), geom);
        let mut params: tucanos_params_t = std::mem::zeroed();
        tucanos_params_init(&mut params);
        tucanos_remesher3daniso_remesh(remesher, &params, geom);
        tucanos_mesh33_delete(mesh);
        let mesh = tucanos_remesher3daniso_tomesh(remesher, false);
        let num_verts = tucanos_mesh33_num_verts(mesh);
        assert_eq!(num_verts, 30);
        tucanos_mesh33_delete(mesh);
    }
}
