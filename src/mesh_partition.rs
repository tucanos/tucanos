use lindel::Lineariseable;
use log::{debug, info};
use scotch::{Graph, Strategy};

use crate::{mesh::SimplexMesh, topo_elems::Elem, Error, Idx, Mesh, Result};

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    pub fn partition_scotch(&mut self, n_parts: Idx) -> Result<()> {
        info!("Partition the mesh into {} using scotch", n_parts);
        if self.elem_to_elems.is_none() {
            self.compute_elem_to_elems();
        }

        let mut partition = vec![0; self.n_elems() as usize];
        let e2e = self.elem_to_elems.as_ref().unwrap();

        let architecture = scotch::Architecture::complete(n_parts as i32);

        let xadj: Vec<scotch::Num> = e2e
            .ptr
            .iter()
            .copied()
            .map(|x| x.try_into().unwrap())
            .collect();
        let adjncy: Vec<scotch::Num> = e2e
            .indices
            .iter()
            .copied()
            .map(|x| x.try_into().unwrap())
            .collect();

        let mut graph = Graph::build(&scotch::graph::Data::new(
            0,
            &xadj,
            &[],
            &[],
            &[],
            &adjncy,
            &[],
        ))
        .unwrap();
        graph.check().unwrap();
        graph
            .mapping(&architecture, &mut partition)
            .compute(&mut Strategy::new())?;

        let partition = partition.iter().copied().map(|i| i as Idx).collect();
        self.partition = Some(partition);

        Ok(())
    }

    pub fn partition_metis(&mut self, n_parts: Idx) -> Result<()> {
        info!("Partition the mesh into {} using metis", n_parts);
        if self.elem_to_elems.is_none() {
            self.compute_elem_to_elems();
        }

        let mut partition = vec![0; self.n_elems() as usize];
        let e2e = self.elem_to_elems.as_ref().unwrap();

        let mut xadj: Vec<metis::Idx> = e2e
            .ptr
            .iter()
            .copied()
            .map(|x| x.try_into().unwrap())
            .collect();
        let mut adjncy: Vec<metis::Idx> = e2e
            .indices
            .iter()
            .copied()
            .map(|x| x.try_into().unwrap())
            .collect();

        metis::Graph::new(1, n_parts as metis::Idx, &mut xadj, &mut adjncy)
            .part_recursive(&mut partition)
            .unwrap();

        let partition = partition.iter().copied().map(|i| i as Idx).collect();
        self.partition = Some(partition);

        Ok(())
    }

    pub fn partition_quality(&self) -> Result<f64> {
        if self.faces_to_elems.is_none() {
            return Err(Error::from("face to element connectivity not computed"));
        }

        if self.partition.is_none() {
            return Err(Error::from("partition not computed"));
        }

        let f2e = self.faces_to_elems.as_ref().unwrap();
        let p = self.partition.as_ref().unwrap();

        let n = f2e
            .iter()
            .filter(|(_, v)| v.len() == 2 && p[v[0] as usize] != p[v[1] as usize])
            .count();
        Ok(n as f64 / f2e.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        test_meshes::{test_mesh_2d, test_mesh_3d},
        Result,
    };

    #[test]
    fn test_partition_scotch_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();

        mesh.partition_scotch(4)?;

        // mesh.write_xdmf("test.xdmf", 0.0)?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.03);

        Ok(())
    }

    #[test]
    fn test_partition_scotch_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();

        mesh.partition_scotch(4)?;

        // mesh.write_xdmf("test.xdmf", 0.0)?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.025);

        Ok(())
    }

    #[test]
    fn test_partition_metis_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();

        mesh.partition_metis(4)?;

        // mesh.write_xdmf("test.xdmf", 0.0)?;

        let q = mesh.partition_quality()?;
        // println!("{}", q);
        assert!(q < 0.025);

        Ok(())
    }

    #[test]
    fn test_partition_metis_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();

        mesh.partition_metis(4)?;

        mesh.write_xdmf("test.xdmf", 0.0)?;

        let q = mesh.partition_quality()?;
        // println!("{}", q);
        assert!(q < 0.02);

        Ok(())
    }
}
