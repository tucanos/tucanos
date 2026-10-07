use log::info;
use std::time::Instant;
use tmesh::{
    Result, init_log,
    mesh::{BoundaryMesh3d, Mesh, Mesh3d},
};
use tucanos::{
    geometry::MeshedGeometry,
    mesh::MeshTopology,
    metric::{AnisoMetric, AnisoMetric3d, Metric},
    remesher::{Remesher, RemesherParams},
};

fn main() -> Result<()> {
    init_log("info");

    // Load the mesh
    let mut msh =
        Mesh3d::from_meshb("../solution/adapt_in.meshb")?;
    msh.check(&msh.all_faces())?;

    let (bdy, ifc) = msh.fix()?;
    assert!(bdy.is_empty());
    assert!(ifc.is_empty());

    // Load the solution
    let (metric, n_comp) =
        Mesh3d::read_solb("../solution/adapt_in_m.solb")?;
    assert_eq!(n_comp, 6);
    let metric = metric
        .chunks(6)
        .map(|s| {
            let m = AnisoMetric3d::from_slice(s);
            let mat = m.as_mat();
            let mut eig = mat.symmetric_eigen();
            eig.eigenvalues
                .iter_mut()
                .for_each(|x| *x = 1.0 / (*x * *x));
            let mat = eig.recompose();
            AnisoMetric3d::from_mat(mat)
        })
        .collect::<Vec<_>>();

    info!("# of verts: {}", msh.n_verts());
    info!("# of elems: {}", msh.n_elems());
    info!("# of faces: {}", msh.n_faces());

    let (mut bdy, _): (BoundaryMesh3d, _) = msh.boundary();
    bdy.fix()?;

    let topo = MeshTopology::new(&msh);
    let mut geom = MeshedGeometry::new(&bdy)?;
    geom.set_topo_map(topo.topo());

    let params = RemesherParams::default();
    let start = Instant::now();
    let mut remesher = Remesher::new(&msh, &topo, &metric, &geom)?;
    info!("Before remeshing");
    remesher.log_stats(log::Level::Info);
    remesher.remesh(&params, &geom)?;
    info!("After remeshing");
    remesher.log_stats(log::Level::Info);

    let elapsed = start.elapsed();
    info!("Done in {elapsed:.3?}s");

    Ok(())
}
