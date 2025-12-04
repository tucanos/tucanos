use tmesh::{
    init_log,
    io::{VTUEncoding, VTUFile},
    mesh::{GenericMesh, Mesh, Triangle, read_stl},
};
use tucanos::{
    geometry::MeshedGeometry,
    mesh::{MeshTopology, autotag},
    metric::IsoMetric,
    remesher::{Remesher, RemesherParams, RemeshingStep},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_log("debug");
    let mut mesh: GenericMesh<3, Triangle<usize>> =
        read_stl("/home/robert/devel/tucanos-project/cylinder64.stl")?;
    //let mut mesh = mesh.split();
    let mut mesh2: GenericMesh<3, Triangle<usize>> =
        read_stl("/home/robert/devel/tucanos-project/cylinder64_p.stl")?;
    autotag(&mut mesh, 30.0)?;
    autotag(&mut mesh2, 30.0)?;
    mesh.fix()?;
    mesh2.fix()?;
    mesh.write_vtk("/home/robert/devel/tucanos-project/cylinder64_input.vtu")?;
    let topo = MeshTopology::new(&mesh);
    let geom = MeshedGeometry::new(&mesh, &topo, mesh2)?;
    let metric: Vec<_> = mesh.verts().map(|_| IsoMetric::from(0.05)).collect();
    let mut remesher = Remesher::new(&mesh, &topo, &metric, &geom)?;
    let mut params = RemesherParams::default();
    for step in &mut params.steps {
        if let RemeshingStep::Split(p) = step {
            p.min_q_abs = 0.0;
            p.min_l_abs = 0.0;
        }
        if let RemeshingStep::Collapse(p) = step {
            p.min_q_abs = 0.0;
            p.max_l_abs = f64::MAX;
        }
    }
    remesher.remesh(&params, &geom)?;
    let out_mesh = remesher.to_mesh(false);
    let writer = VTUFile::from_mesh(&out_mesh, VTUEncoding::Binary);
    writer
        .export("/home/robert/devel/tucanos-project/cylinder64.vtu")
        .unwrap();
    Ok(())
}
