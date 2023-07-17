use log::{info, warn};
use crate::{mesh::SimplexMesh, topo_elems::Elem, Idx, Result, Tag};
#[cfg(any(not(feature = "metis"),not(feature = "scotch")))]
use crate::Error;
    
impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    #[cfg(not(feature = "scotch"))]
    pub fn partition_scotch(&mut self, _n_parts: Idx) -> Result<()> {
        Err(Error::from("the scotch feature is not enabled"))
    }

    #[cfg(feature = "scotch")]
    pub fn partition_scotch(&mut self, n_parts: Idx) -> Result<()> {
        if self.etags().any(|t| t != 1) {
            warn!("Erase the element tags");
        }
        info!("Partition the mesh into {} using scotch", n_parts);

        let mut partition = vec![0; self.n_elems() as usize];
        let e2e = self.get_elem_to_elems()?;

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

        let mut graph = scotch::Graph::build(&scotch::graph::Data::new(
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
            .compute(&mut scotch::Strategy::new())?;

        self.mut_etags()
            .enumerate()
            .for_each(|(i, t)| *t = partition[i] as Tag + 1);

        Ok(())
    }

    #[cfg(not(feature = "metis"))]
    pub fn partition_metis(&mut self, _n_parts: Idx) -> Result<()> {
        Err(Error::from("the metis feature is not enabled"))
    }

    #[cfg(feature = "metis")]
    pub fn partition_metis(&mut self, n_parts: Idx) -> Result<()> {
        if self.etags().any(|t| t != 1) {
            warn!("Erase the element tags");
        }

        info!("Partition the mesh into {} using metis", n_parts);

        let mut partition = vec![0; self.n_elems() as usize];
        let e2e = self.get_elem_to_elems()?;

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

        self.mut_etags()
            .enumerate()
            .for_each(|(i, t)| *t = partition[i] as Tag + 1);

        Ok(())
    }

    pub fn partition_quality(&self) -> Result<f64> {
        let f2e = self.get_face_to_elems()?;

        let n = f2e
            .iter()
            .filter(|(_, v)| v.len() == 2 && self.etag(v[0]) != self.etag(v[1]))
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

    #[cfg(feature = "scotch")]
    #[test]
    fn test_partition_scotch_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_scotch(4)?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.03);

        Ok(())
    }

    #[cfg(feature = "scotch")]
    #[test]
    fn test_partition_scotch_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_scotch(4)?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.025);

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_metis(4)?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.03);

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_metis(4)?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.02);

        Ok(())
    }
}
