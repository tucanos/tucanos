use log::info;
use rustc_hash::FxHashSet;
use tmesh::mesh::Mesh;
use tucanos::{
    Result, Tag,
    geometry::{Geometry, LinearGeometry, NoGeometry, orient_geometry},
    init_log,
    mesh::{GElem, HasTmeshImpl, Point, SimplexMesh, Tetrahedron, Triangle},
    metric::{AnisoMetric3d, Metric},
    remesher::{Remesher, RemesherParams, SplitParams},
};

fn main() -> Result<()> {
    init_log("debug");

    let prefix = "forXGCRM/debug_xg";

    // Load the mesh
    let mut mesh = SimplexMesh::<3, Tetrahedron>::from_meshb(&format!("{prefix}_mesh.meshb"))?;

    // Extract a submesh close to the issue
    let pt = Point::<3>::new(38.5, 3.0, 4.8);
    let radius = 4.0;

    let etags = mesh
        .gelems()
        .map(|ge| {
            let c = ge.center();
            if (c - pt).norm() < radius { 2 } else { 1 }
        })
        .collect::<Vec<_>>();
    mesh.mut_etags()
        .zip(etags.iter())
        .for_each(|(t, new_t)| *t = *new_t);

    let submesh = mesh.extract_tag(2);

    // Load the metric and extract the relevant vertices
    let (metric, m) = SimplexMesh::<3, Tetrahedron>::read_solb(&format!("{prefix}_metric.solb"))?;
    assert_eq!(m, 6);
    let metric = metric
        .chunks(6)
        .map(AnisoMetric3d::from_slice)
        .collect::<Vec<_>>();

    let parent_ids = submesh.parent_vert_ids;
    let metric = parent_ids
        .iter()
        .map(|&i| metric[i as usize])
        .collect::<Vec<_>>();

    // Get the submesh & save it
    let mut mesh = submesh.mesh;
    let (bfaces, _) = mesh.add_boundary_faces();
    assert_eq!(bfaces.len(), 1);
    let tag = bfaces.get(&2).unwrap();
    let new_tag = -mesh.ftags().map(Tag::abs).max().unwrap_or(1) - 1;
    mesh.mut_ftags()
        .filter(|t| *t == tag)
        .for_each(|t| *t = new_tag);

    mesh.write_meshb(&format!("{prefix}_extract_mesh.meshb"))
        .unwrap();

    // Check the mesh
    mesh.check_simple()?;

    mesh.compute_topology();

    // remeshing params
    let params = RemesherParams::default();

    // Linear geometry
    let bdy = SimplexMesh::<3, Triangle>::from_meshb(&format!("{prefix}_bdy.meshb"))?;

    // Extract only the relevant part
    let face_tags = mesh.ftags().collect::<FxHashSet<_>>();
    let mut bdy = bdy.extract_tags(|t| face_tags.contains(&t)).mesh;

    orient_geometry(&mesh, &mut bdy);
    let geom = LinearGeometry::new(&mesh, bdy)?;

    let max_angle = geom.max_normal_angle(&mesh);
    info!("max_angle: {max_angle}");

    let mut remesher = Remesher::new(&mesh, &metric, &geom)?;
    remesher
        .split(
            &SplitParams {
                l: 1.4,
                min_q_rel_bdy: 0.0,
                ..Default::default() // max_iter: 2,
                                     // min_l_rel: 0.0,
                                     // min_l_abs: 0.0,
                                     // min_q_rel: 0.0,
                                     // min_q_abs: 0.0,
            },
            &geom,
            false,
        )
        .unwrap();
    // remesher.remesh(&params, &geom)?;
    // remesher.split(2.0_f64.sqrt(), &params, &geom)?;
    remesher.check()?;
    let new_mesh = remesher.to_mesh(true);

    new_mesh.check_simple()?;

    new_mesh.write_meshb(&format!("{prefix}_adapted.meshb"))?;

    if true {
        return Ok(());
    }

    // No geometry
    let geom = NoGeometry();

    let mut remesher = Remesher::new(&mesh, &metric, &geom)?;
    remesher.remesh(&params, &geom)?;
    remesher.check()?;
    let new_mesh = remesher.to_mesh(true);

    new_mesh.check_simple()?;

    new_mesh.write_meshb(&format!("{prefix}_adapted_nogeom.meshb"))?;

    Ok(())
}
