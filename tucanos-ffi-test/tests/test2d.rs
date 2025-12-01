use tucanos_ffi_test::*;

#[test]
fn iso2d() {
    let vertices = [0., 0., 1., 0., 0., 1.];
    let faces = [0, 1, 1, 2, 2, 0];
    let metric = [0.1; 3];
    unsafe {
        // Create a mesh with a single triangle
        let mesh = tucanos_mesh22_new(
            3,
            vertices.as_ptr(),
            1,
            [0, 1, 2].as_ptr(),
            [1].as_ptr(),
            3,
            faces.as_ptr(),
            [1, 2, 3].as_ptr(),
        );
        let boundary = tucanos_mesh22_boundary(mesh);
        let geom = tucanos_geom2d_new(mesh, boundary);
        assert!(!geom.is_null());
        let remesher = tucanos_remesher2diso_new(mesh, metric.as_ptr(), geom);
        tucanos_remesher2diso_remesh(remesher, geom);
        tucanos_mesh22_delete(mesh);
        let mesh = tucanos_remesher2diso_tomesh(remesher, false);
        let num_verts = tucanos_mesh22_num_verts(mesh);
        assert_eq!(num_verts, 81);
        tucanos_mesh22_delete(mesh);
    }
}

#[test]
fn aniso2d() {
    let vertices = [0., 0., 1., 0., 0., 1.];
    let faces = [0, 1, 1, 2, 2, 0];
    let metric: Vec<_> = (0..3).flat_map(|_| [10., 20., 15.].into_iter()).collect();
    unsafe {
        // Create a mesh with a single triangle
        let mesh = tucanos_mesh22_new(
            3,
            vertices.as_ptr(),
            1,
            [0, 1, 2].as_ptr(),
            [1].as_ptr(),
            3,
            faces.as_ptr(),
            [1, 2, 3].as_ptr(),
        );
        let boundary = tucanos_mesh22_boundary(mesh);
        let geom = tucanos_geom2d_new(mesh, boundary);
        assert!(!geom.is_null());
        let remesher = tucanos_remesher2daniso_new(mesh, metric.as_ptr(), geom);
        tucanos_remesher2daniso_remesh(remesher, geom);
        tucanos_mesh22_delete(mesh);
        let mesh = tucanos_remesher2daniso_tomesh(remesher, false);
        let num_verts = tucanos_mesh22_num_verts(mesh);
        assert_eq!(num_verts, 10);
        tucanos_mesh22_delete(mesh);
    }
}
