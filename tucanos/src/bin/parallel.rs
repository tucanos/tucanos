use std::{fs::File, io::BufWriter, io::Write};

use clap::Parser;
use tmesh::init_log;
use tmesh::mesh::BoundaryMesh3d;
use tmesh::mesh::partition::{MetisKWay, MetisPartitioner};
use tmesh::{
    Result,
    mesh::{Mesh, Mesh3d},
};
use tucanos::geometry::MeshedGeometry;
use tucanos::remesher::{ParallelRemesher, ParallelRemesherParams};
use tucanos::{
    mesh::MeshTopology,
    metric::{AnisoMetric, AnisoMetric3d, Metric},
    remesher::RemesherParams,
};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Estimate mesh work/complexity from a metric field"
)]
struct Cli {
    /// Input 3D mesh (.mesh / .meshb)
    #[arg(long)]
    mesh: String,

    /// Input metric/solution (.sol / .solb) defined at mesh vertices
    #[arg(long)]
    metric: String,
}

#[allow(dead_code)]
fn write_bin(path: &str, data: impl IntoIterator<Item = f64>) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for x in data {
        writer.write_all(&x.to_le_bytes())?;
    }
    writer.flush()?;

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    init_log("info");

    let cli = Cli::parse();

    let msh = Mesh3d::from_meshb(&cli.mesh)?;
    let (metric, n_comp) = Mesh3d::read_solb(&cli.metric)?;

    let (mut bdy, _): (BoundaryMesh3d, _) = msh.boundary();
    bdy.fix()?;

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
    assert_eq!(metric.len(), msh.n_verts());

    let topo = MeshTopology::new(&msh);
    let mut geom = MeshedGeometry::new(&bdy)?;
    geom.set_topo_map(topo.topo());

    let params = RemesherParams::default();
    let parallel_params = ParallelRemesherParams {
        max_levels: 2,
        ..Default::default()
    };

    let n_parts = 16;
    let remesher = ParallelRemesher::<_, _, MetisPartitioner<MetisKWay>>::new(msh, topo, n_parts)?;
    let (_res, _info, _) = remesher.remesh(&metric, &geom, params, &parallel_params)?;

    Ok(())
}
