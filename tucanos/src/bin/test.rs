use tmesh::{
    io::{VTUEncoding, VTUFile},
    mesh::{Edge, GenericMesh, Mesh, Triangle, read_stl},
};
use tucanos::{
    geometry::LinearGeometry,
    mesh::{MeshTopology, autotag},
    metric::IsoMetric,
    remesher::{Remesher, RemesherParams},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mesh: GenericMesh<3, Triangle<usize>> =
        read_stl("/home/robert/devel/tucanos-project/cylinder64.stl")?;
    autotag(&mut mesh, 30.0)?;
    mesh.fix()?;
    mesh.write_vtk("/home/robert/devel/tucanos-project/cylinder64.vtu")?;
    mesh.boundary::<GenericMesh<3, Edge<usize>>>()
        .0
        .write_vtk("/home/robert/devel/tucanos-project/cylinder64_bdy.vtu")?;
    let topo = MeshTopology::new(&mesh);
    let boundary = mesh.clone();
    let geom = LinearGeometry::new(&mesh, &topo, boundary)?;
    let metric: Vec<_> = mesh.verts().map(|_| IsoMetric::from(0.05)).collect();
    let mut remesher = Remesher::new(&mesh, &topo, &metric, &geom)?;
    let params = RemesherParams::default();
    remesher.remesh(&params, &geom)?;
    let out_mesh = remesher.to_mesh(false);
    let writer = VTUFile::from_mesh(&out_mesh, VTUEncoding::Binary);
    writer
        .export("/home/robert/devel/tucanos-project/cylinder64.vtu")
        .unwrap();
    Ok(())
}
