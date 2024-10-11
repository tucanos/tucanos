use crate::{
    mesh::{ordering::hilbert_indices, Elem, GElem, SimplexMesh},
    Idx, Result, Tag,
};
use log::{debug, warn};

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum PartitionType {
    Hilbert(Idx),
    Scotch(Idx),
    MetisRecursive(Idx),
    MetisKWay(Idx),
    None,
}

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    pub fn partition(&mut self, ptype: PartitionType) -> Result<()> {
        match ptype {
            PartitionType::Hilbert(n) => {
                self.partition_hilbert(n);
                Ok(())
            }
            PartitionType::Scotch(n) => self.partition_scotch(n),
            PartitionType::MetisRecursive(n) => self.partition_metis(n, "recursive"),
            PartitionType::MetisKWay(n) => self.partition_metis(n, "kway"),
            PartitionType::None => unreachable!(),
        }
    }

    pub fn partition_hilbert(&mut self, n_parts: Idx) {
        debug!("Partition the mesh into {} using a Hilbert curve", n_parts);

        if self.etags().any(|t| t != 1) {
            warn!("Erase the element tags");
        }

        if n_parts == 1 {
            self.mut_etags().for_each(|t| *t = 1);
        } else {
            let indices = hilbert_indices(self.bounding_box(), self.gelems().map(|ge| ge.center()));

            let m = self.n_elems() / n_parts + 1;
            let partition = indices.iter().map(|&i| (i / m) as Tag).collect::<Vec<_>>();

            self.mut_etags()
                .enumerate()
                .for_each(|(i, t)| *t = partition[i] as Tag + 1);
        }
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    #[cfg(not(feature = "scotch"))]
    pub fn partition_scotch(&mut self, _n_parts: Idx) -> Result<()> {
        use crate::Error;
        Err(Error::from("the scotch feature is not enabled"))
    }

    /// Partition the mesh using scotch into `n_parts`. The partition id, defined for all the elements
    /// is stored in self.etags
    #[cfg(feature = "scotch")]
    pub fn partition_scotch(&mut self, n_parts: Idx) -> Result<()> {
        debug!("Partition the mesh into {} using scotch", n_parts);

        if self.etags().any(|t| t != 1) {
            warn!("Erase the element tags");
        }

        if n_parts == 1 {
            self.mut_etags().for_each(|t| *t = 1);
            return Ok(());
        }

        let mut partition = vec![0; self.n_elems() as usize];
        let e2e = self.compute_elem_to_elems();

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

    #[allow(clippy::needless_pass_by_ref_mut)]
    #[cfg(not(feature = "metis"))]
    pub fn partition_metis(&mut self, _n_parts: Idx, _method: &str) -> Result<()> {
        use crate::Error;
        Err(Error::from("the metis feature is not enabled"))
    }

    /// Partition the mesh using metis into `n_parts`. The partition id, defined for all the elements
    /// is stored in self.etags
    #[cfg(feature = "metis")]
    pub fn partition_metis(&mut self, n_parts: Idx, method: &str) -> Result<()> {
        debug!("Partition the mesh into {} using metis", n_parts);

        if self.etags().any(|t| t != 1) {
            warn!("Erase the element tags");
        }

        if n_parts == 1 {
            self.mut_etags().for_each(|t| *t = 1);
            return Ok(());
        }

        let mut partition = vec![0; self.n_elems() as usize];
        let e2e = self.compute_elem_to_elems();

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

        let graph = metis::Graph::new(1, n_parts as metis::Idx, &mut xadj, &mut adjncy);
        match method {
            "recursive" => graph.part_recursive(&mut partition).unwrap(),
            "kway" => graph.part_kway(&mut partition).unwrap(),
            _ => unreachable!("Unknown method"),
        };

        self.mut_etags()
            .enumerate()
            .for_each(|(i, t)| *t = partition[i] as Tag + 1);

        Ok(())
    }

    /// Get the partition quality (ration of the number of interface faces to the total number of faces)
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
        mesh::test_meshes::{test_mesh_2d, test_mesh_3d},
        Result,
    };

    #[test]
    fn test_partition_hilbert_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_hilbert(4);

        let q = mesh.partition_quality()?;
        assert!(q < 0.025, "failed, q = {q}");

        Ok(())
    }

    #[test]
    fn test_partition_hilbert_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_hilbert(4);

        let q = mesh.partition_quality()?;
        assert!(q < 0.025, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "scotch")]
    #[test]
    fn test_partition_scotch_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_scotch(4)?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.03, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "scotch")]
    #[test]
    fn test_partition_scotch_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_scotch(4)?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.025, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_metis(4, "recursive")?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.03, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_metis(4, "recursive")?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.025, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_2d_kway() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_metis(4, "kway")?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.03, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_3d_kway() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();
        mesh.compute_elem_to_elems();
        mesh.partition_metis(4, "kway")?;

        let q = mesh.partition_quality()?;
        assert!(q < 0.022, "failed, q = {q}");

        Ok(())
    }
}
