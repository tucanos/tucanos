use std::time::Instant;
use std::{fs::File, io::BufWriter, io::Write};

use clap::Parser;
use tmesh::mesh::BoundaryMesh3d;
use tmesh::mesh::partition::{MetisKWay, MetisPartitioner};
use tmesh::{
    Result, Tag,
    mesh::{
        GSimplex, Mesh, Mesh3d, SubMesh,
        partition::{KaHIPPartitioner, KahipEco, Partitioner},
    },
};
use tucanos::geometry::MeshedGeometry;
use tucanos::{
    mesh::MeshTopology,
    metric::{AnisoMetric, AnisoMetric3d, ImpliedMetric, Metric},
    remesher::{Remesher, RemesherParams},
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
    let cli = Cli::parse();

    let mut msh = Mesh3d::from_meshb(&cli.mesh)?;
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

    let _etags = msh.etags().collect::<Vec<_>>();

    let elem_implied_metric = msh
        .gelems()
        .map(|ge| ge.implied_metric())
        .collect::<Vec<_>>();
    let elem_metric = msh
        .elems()
        .map(|e| AnisoMetric3d::interpolate(e.into_iter().map(|j| (1.0 / 3.0, &metric[j]))))
        .collect::<Vec<_>>();
    let elem_vols = msh.gelems().map(|ge| ge.vol()).collect::<Vec<_>>();
    let qualities = msh
        .gelems()
        .zip(msh.elems())
        .map(|(ge, e)| AnisoMetric3d::quality(&ge, e.into_iter().map(|j| metric[j])))
        .collect::<Vec<_>>();

    let topo = MeshTopology::new(&msh);
    let mut geom = MeshedGeometry::new(&bdy)?;
    geom.set_topo_map(topo.topo());

    let mut i = 0;
    for n_parts in [2, 4, 8, 16, 32] {
        for seed in [1] {
            //124, 12345, 123456, 1234567, 12345678, 123456789] {
            let _partitioner = KaHIPPartitioner::<KahipEco>::new(&msh, n_parts, None)?;
            // let partitions = partitioner.compute_with(0.05, seed);
            let partitioner = MetisPartitioner::<MetisKWay>::new(&msh, n_parts, None)?;
            let partitions = partitioner.compute()?;
            msh.etags_mut()
                .zip(partitions.iter())
                .for_each(|(etag, &part)| {
                    *etag = part as Tag;
                });

            for i_part in 0..n_parts {
                let submsh = SubMesh::new(&msh, |etag| etag == i_part as Tag);
                let mut msh = submsh.mesh;
                let (bdy_tags, ifc_tags) = msh.fix()?;
                assert!(ifc_tags.is_empty());
                assert_eq!(bdy_tags.len(), 1);
                let etag = i_part as Tag;
                let ftag = *bdy_tags.get(&etag).unwrap();
                for t in msh.ftags_mut() {
                    if *t == ftag {
                        *t = -*t;
                    }
                }
                msh.etags_mut().for_each(|t| *t = 1);

                write_bin(
                    &format!("metric_vol_{i:04}.bin"),
                    submsh.parent_elem_ids.iter().map(|&i| elem_metric[i].vol()),
                )?;
                write_bin(
                    &format!("implied_metric_vol_{i:04}.bin"),
                    submsh
                        .parent_elem_ids
                        .iter()
                        .map(|&i| elem_implied_metric[i].vol()),
                )?;
                write_bin(
                    &format!("intersect_metric_vol_{i:04}.bin"),
                    submsh
                        .parent_elem_ids
                        .iter()
                        .map(|&i| elem_metric[i].intersect(&elem_implied_metric[i]).vol()),
                )?;
                write_bin(
                    &format!("qualities_{i:04}.bin"),
                    submsh.parent_elem_ids.iter().map(|&i| qualities[i]),
                )?;
                write_bin(
                    &format!("elem_vol_{i:04}.bin"),
                    submsh.parent_elem_ids.iter().map(|&i| elem_vols[i]),
                )?;
                i += 1;

                let metric = submsh
                    .parent_vert_ids
                    .iter()
                    .map(|&i| metric[i])
                    .collect::<Vec<_>>();

                let local_topo = MeshTopology::new_from(&msh, topo.topo().clone());

                let start = Instant::now();
                let mut remesh = Remesher::new(&msh, &local_topo, &metric, &geom)?;
                remesh.remesh(&RemesherParams::default(), &geom)?;
                let duration = start.elapsed();
                write_bin(&format!("elapsed_{i:04}.bin"), [duration.as_secs_f64()])?;
                println!("seed {seed} / part {i_part}: remeshing took {duration:?}");
            }
        }
    }

    Ok(())
}
