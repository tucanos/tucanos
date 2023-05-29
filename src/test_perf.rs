use env_logger::Env;
use tucanos::{
    geometry::LinearGeometry,
    mesh::Point,
    metric::AnisoMetric3d,
    remesher::{Remesher, RemesherParams},
    test_meshes::test_mesh_3d,
    Result,
};

pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}

fn main() -> Result<()> {
    init_log("debug");

    // Initial mesh
    let mut mesh = test_mesh_3d().split();
    mesh.compute_topology();

    // Analytical metric
    let mfunc = |p: Point<3>| {
        let hx = 0.1;
        let hy = 0.1;
        let h0 = 0.001;
        let z = p[2];
        let hz = h0 + 2.0 * (0.1 - h0) * f64::abs(z - 0.5);
        let v0 = Point::<3>::new(hx, 0., 0.);
        let v1 = Point::<3>::new(0.0, hy, 0.);
        let v2 = Point::<3>::new(0., 0.0, hz);
        AnisoMetric3d::from_sizes(&v0, &v1, &v2)
    };
    let m: Vec<_> = mesh.verts().map(mfunc).collect();

    let (mut bdy, _) = mesh.boundary();
    bdy.compute_octree();
    let geom = LinearGeometry::new(&mesh, bdy)?;

    let mut remesher = Remesher::new(&mesh, &m, geom)?;

    remesher.remesh(RemesherParams::default());
    remesher.check()?;

    let mesh = remesher.to_mesh(true);
    mesh.write_vtk("test_perf.vtu", None, None)?;

    Ok(())
}
