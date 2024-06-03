use env_logger::Env;
use rustc_hash::FxHashSet;
use std::time::Instant;
use tucanos::{
    Idx, Result, Tag, mesh::ConnectedComponents, mesh::PartitionType, mesh::SimplexMesh,
    mesh::Tetrahedron, mesh::test_meshes::test_mesh_3d,
};

pub fn init_log(level: &str) {
    env_logger::Builder::from_env(Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}

fn main() -> Result<()> {
    init_log("error");

    // Load the mesh
    let mut mesh = SimplexMesh::<3, Tetrahedron>::read_meshb("data/simple3d.meshb")?;
    let mut _mesh = test_mesh_3d().split().split().split();

    let n_parts = 8;

    let ptypes = [
        PartitionType::Hilbert(n_parts),
        #[cfg(feature = "scotch")]
        PartitionType::Scotch(n_parts),
        #[cfg(feature = "metis")]
        PartitionType::MetisRecursive(n_parts),
        #[cfg(feature = "metis")]
        PartitionType::MetisKWay(n_parts),
    ];

    for ptype in ptypes {
        println!("{ptype:?}");
        mesh.clear_all();
        let now = Instant::now();
        mesh.partition(ptype)?;
        let t = now.elapsed().as_secs_f64();
        mesh.compute_face_to_elems();
        let q = mesh.partition_quality()?;
        println!("elapsed time: {t:.2e}s");
        println!("quality: {q:.2e}");
        for i_part in 0..n_parts {
            let mut smsh = mesh.extract_tag(i_part as Tag + 1);
            let e2e = smsh.mesh.compute_elem_to_elems();
            let cc = ConnectedComponents::<Idx>::new(e2e);
            if cc.is_ok() {
                let n_cc = cc?.tags().iter().clone().collect::<FxHashSet<_>>().len();
                println!(
                    "partition {i_part} : {} elements,  {n_cc} connected components",
                    smsh.mesh.n_elems()
                );
            } else {
                println!(
                    "partition {i_part} : {} elements,  too many connected components",
                    smsh.mesh.n_elems()
                );
            }
        }
    }

    Ok(())
}
