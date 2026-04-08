use log::info;
use rustc_hash::FxHashSet;
#[cfg(not(feature = "metis"))]
use tmesh::mesh::partition::HilbertPartitioner;
#[cfg(feature = "metis")]
use tmesh::mesh::partition::{MetisPartitioner, MetisRecursive};
use tmesh::{
    Vert3d, init_log,
    mesh::{BoundaryMesh3d, GSimplex, Mesh, Mesh3d, SubMesh},
};
use tucanos::{
    Result, Tag,
    geometry::{Geometry, MeshedGeometry},
    mesh::MeshTopology,
    metric::{AnisoMetric, AnisoMetric3d, Metric, MetricField},
    remesher::{ParallelRemesher, ParallelRemesherParams, Remesher, RemesherParams},
};

/// Extract a ball of radius `radius` around `pt`
fn extract_ball(
    mut mesh: Mesh3d,
    metric: &[AnisoMetric3d],
    pt: &Vert3d,
    radius: f64,
    prefix: &str,
) -> Result<(Mesh3d, Vec<AnisoMetric3d>)> {
    let etags = mesh
        .gelems()
        .map(|ge| {
            let c = ge.center();
            if (c - pt).norm() < radius { 2 } else { 1 }
        })
        .collect::<Vec<_>>();
    mesh.etags_mut()
        .zip(etags.iter())
        .for_each(|(t, new_t)| *t = *new_t);

    let submesh = SubMesh::new(&mesh, |t| t == 2);

    let parent_ids = submesh.parent_vert_ids;

    mesh = submesh.mesh;
    let metric = parent_ids.iter().map(|&i| metric[i]).collect::<Vec<_>>();

    // Get the submesh & save it
    let (bfaces, _) = mesh.fix()?;
    assert_eq!(bfaces.len(), 1);
    let tag = bfaces.get(&2).unwrap();
    let new_tag = -mesh.ftags().map(Tag::abs).max().unwrap_or(1) - 1;
    mesh.ftags_mut()
        .filter(|t| *t == tag)
        .for_each(|t| *t = new_tag);

    mesh.write_meshb(&format!("{prefix}_extract_mesh.meshb"))
        .unwrap();

    Ok((mesh, metric))
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    init_log("info");

    let extract = false;
    let n_threads = 1;

    let prefix = "/midterm/FP/garnaud_x/CODA/tucanos/debug_romain/tagged_face/adapt_in";

    // Load the mesh
    let mut mesh = Mesh3d::from_meshb(&format!("{prefix}.meshb"))?;
    let (bfaces, ifaces) = mesh.fix()?;
    assert!(bfaces.is_empty());
    assert!(ifaces.is_empty());

    // Load the metric and extract the relevant vertices
    let (metric, m) = Mesh3d::read_solb(&format!("{prefix}_m.solb"))?;
    assert_eq!(m, 6);
    // the file contains lengths, not a metric
    let mut metric = metric
        .chunks(6)
        .map(|s| {
            let m = AnisoMetric3d::from_slice(s);
            let mat = m.as_mat();
            let mut eig = mat.symmetric_eigen();
            eig.eigenvalues.iter_mut().for_each(|x| *x = 1. / (*x * *x));
            let mat = eig.recompose();
            AnisoMetric3d::from_mat(mat)
        })
        .collect::<Vec<_>>();

    if extract {
        // Extract a submesh close to the issue
        let pt = Vert3d::new(31.1, 8.17, -0.1);
        let radius = 1.0;
        (mesh, metric) = extract_ball(mesh, &metric, &pt, radius, prefix)?;
    }

    let topo = MeshTopology::new(&mesh);

    // Check the mesh
    mesh.check(&mesh.all_faces())?;

    // Linear geometry
    let mut bdy = BoundaryMesh3d::from_meshb(&format!("{prefix}_bdy.meshb"))?;
    bdy.fix()?;

    // Extract only the relevant part
    if extract {
        let face_tags = mesh.ftags().collect::<FxHashSet<_>>();
        bdy = SubMesh::new(&bdy, |t| face_tags.contains(&t)).mesh;
    }

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

    let new_mesh = if n_threads == 1 {
        info!("Sequential remeshing");
        let mut remesher = Remesher::new(&mesh, &topo, &metric, &geom)?;
        remesher.remesh(&params, &geom)?;
        remesher.check()?;
        remesher.to_mesh(true)
    } else {
        #[cfg(feature = "metis")]
        info!("Parallel remeshing (metis)");
        #[cfg(not(feature = "metis"))]
        info!("Parallel remeshing (hilbert)");

        #[cfg(feature = "metis")]
        let mut dd =
            ParallelRemesher::<_, _, MetisPartitioner<MetisRecursive>>::new(mesh, topo, n_threads)?;
        #[cfg(not(feature = "metis"))]
        let mut dd = ParallelRemesher::<_, _, HilbertPartitioner>::new(mesh, topo, n_threads)?;
        dd.set_debug(false);
        let dd_params = ParallelRemesherParams::new(2, 2, 10000);
        let (mesh, stats, _) = dd.remesh(&metric, &geom, params, &dd_params)?;
        stats.print_summary();
        mesh
    };

    new_mesh.write_meshb(&format!("{prefix}_adapted.meshb"))?;

    new_mesh.check(&new_mesh.all_faces())?;

    info!("Remeshing OK");
    let max_angle = geom.max_normal_angle(&new_mesh);
    info!("max_angle: {max_angle}");

    Ok(())
}
