use log::info;
use tmesh::{
    init_log,
    mesh::{BoundaryMesh2d, Mesh, Mesh2d},
};
use tucanos::{
    Result,
    geometry::{Geometry, MeshedGeometry},
    mesh::MeshTopology,
    metric::{AnisoMetric, AnisoMetric2d, Metric, MetricField},
    remesher::{Remesher, RemesherParams},
};

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    init_log("info");

    let prefix = "/midterm/FP/garnaud_x/CODA/tucanos/debug_anton/ringleb/adapt_in";

    // Load the mesh
    let mut mesh = Mesh2d::from_meshb(&format!("{prefix}.meshb"))?;
    let (bfaces, ifaces) = mesh.fix()?;
    assert!(bfaces.is_empty());
    assert!(ifaces.is_empty());

    // Load the metric and extract the relevant vertices
    let (metric, m) = Mesh2d::read_solb(&format!("{prefix}_m.solb"))?;
    assert_eq!(m, 3);
    // the file contains lengths, not a metric
    let metric = metric
        .chunks(3)
        .map(|s| {
            let m = AnisoMetric2d::from_slice(s);
            let mat = m.as_mat();
            let mut eig = mat.symmetric_eigen();
            eig.eigenvalues.iter_mut().for_each(|x| *x = 1. / (*x * *x));
            let mat = eig.recompose();
            AnisoMetric2d::from_mat(mat)
        })
        .collect::<Vec<_>>();

    let topo = MeshTopology::new(&mesh);

    // Check the mesh
    mesh.check(&mesh.all_faces())?;

    // Linear geometry
    let mut bdy = BoundaryMesh2d::from_meshb(&format!("{prefix}_bdy.meshb"))?;
    bdy.fix()?;

    // orient_geometry(&mesh, &mut bdy);
    let mut geom = MeshedGeometry::new(&bdy)?;
    geom.set_topo_map(topo.topo());

    let max_angle = geom.max_normal_angle(&mesh);
    info!("max_angle (mesh): {max_angle}");
    let max_angle = max_angle.max(10.0);
    info!("max_angle (for remeshing): {max_angle}");

    // remeshing params
    let params = RemesherParams::new(max_angle, 4);

    let (h_min, h_max, aniso_max, complexity) = MetricField::new(&mesh, metric.clone()).info();
    info!("h_min = {h_min:.2e}");
    info!("h_max = {h_max:.2e}");
    info!("max. aniso = {aniso_max:.2e}");
    info!("complexity = {complexity:.2e}");

    info!("Sequential remeshing");
    let mut remesher = Remesher::new(&mesh, &topo, &metric, &geom)?;
    remesher.remesh(&params, &geom)?;
    remesher.check()?;
    let new_mesh = remesher.to_mesh(true);

    new_mesh.write_meshb(&format!("{prefix}_adapted.meshb"))?;

    new_mesh.check(&new_mesh.all_faces())?;

    info!("Remeshing OK");
    let max_angle = geom.max_normal_angle(&new_mesh);
    info!("max_angle: {max_angle}");

    Ok(())
}
