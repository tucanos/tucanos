use log::info;
use rustc_hash::FxHashSet;
use tucanos::{
    geometry::{Geometry, LinearGeometry, NoGeometry},
    init_log,
    mesh::{io::orient_stl, GElem, Point, SimplexMesh, Tetrahedron, Triangle},
    metric::{AnisoMetric3d, Metric},
    remesher::{Remesher, RemesherParams},
    Result,
};

fn main() -> Result<()> {
    init_log("debug");

    let prefix = "M6_debug_for_XG_first/debug_first_pb";

    // Load the mesh
    let mut mesh = SimplexMesh::<3, Tetrahedron>::read_meshb(&format!("{prefix}_mesh.meshb"))?;

    let new_tag = -mesh.ftags().map(|t| t.abs()).max().unwrap_or(1) - 1;
    // Extract a submesh close to the issue
    let pt = Point::<3>::new(0.49, 0.85, 0.0);
    let etags = mesh
        .gelems()
        .map(|ge| {
            let c = ge.center();
            if (c - pt).norm() < 0.1 {
                2
            } else {
                1
            }
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
    mesh.mut_ftags()
        .filter(|t| *t == tag)
        .for_each(|t| *t = new_tag);

    mesh.write_meshb(&format!("{prefix}_extract_mesh.meshb"))
        .unwrap();

    // Check the mesh
    mesh.check()?;

    mesh.compute_topology();

    // remeshing params
    let debug = false;
    let params = RemesherParams {
        two_steps: false,
        num_iter: 2,
        split_max_iter: 1,
        collapse_max_iter: 1,
        max_angle: 45.0,
        debug,
        ..RemesherParams::default()
    };

    // Linear geometry
    let bdy = SimplexMesh::<3, Triangle>::read_meshb(&format!("{prefix}_bdy.meshb"))?;

    // Extract only the relevant part
    let face_tags = mesh.ftags().collect::<FxHashSet<_>>();
    let mut bdy = bdy.extract(|t| face_tags.get(&t).is_some()).mesh;

    orient_stl(&mesh, &mut bdy);
    let geom = LinearGeometry::new(&mesh, bdy)?;

    let max_angle = geom.max_normal_angle(&mesh);
    info!("max_angle: {max_angle}");

    let mut remesher = Remesher::new(&mesh, &metric, &geom)?;
    remesher.remesh(params.clone(), &geom)?;
    // remesher.split(2.0_f64.sqrt(), &params, &geom)?;
    remesher.check()?;
    let mut new_mesh = remesher.to_mesh(true);

    new_mesh.compute_face_to_elems();
    new_mesh.check()?;

    new_mesh.write_meshb(&format!("{prefix}_adapted.meshb"))?;

    if true {
        return Ok(());
    }

    // No geometry
    let geom = NoGeometry();

    let mut remesher = Remesher::new(&mesh, &metric, &geom)?;
    remesher.remesh(params, &geom)?;
    remesher.check()?;
    let mut new_mesh = remesher.to_mesh(true);

    new_mesh.compute_face_to_elems();
    new_mesh.check()?;

    new_mesh.write_meshb("M6_debug_for_XG/extract_adapted_nogeom.meshb")?;

    Ok(())
}
