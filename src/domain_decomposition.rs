use log::{debug, info, warn};
use rustc_hash::FxHashSet;

use crate::{
    geometry::Geometry,
    mesh::{SimplexMesh, SubSimplexMesh},
    metric::Metric,
    remesher::{Remesher, RemesherParams},
    topo_elems::Elem,
    Idx, Result, Tag,
};

#[allow(dead_code)]
pub enum PartitionType {
    Scotch(Idx),
    Metis(Idx),
    None,
}

/// Domain decomposition
pub struct DomainDecomposition<const D: usize, E: Elem> {
    pub mesh: SimplexMesh<D, E>,
    partition_tags: Vec<Tag>,
}

impl<const D: usize, E: Elem> DomainDecomposition<D, E> {
    /// Create a new domain decomposition.
    /// If part is `PartitionType::Scotch(n)` or `PartitionType::Metis(n)` the mesh is partitionned into n subdomains using
    /// scotch / metis. If None, the element tag in `mesh` is used as the partition Id
    ///
    /// NB: the mesh element tags will be modified
    pub fn new(mut mesh: SimplexMesh<D, E>, part: PartitionType) -> Result<Self> {
        // Partition if needed
        match part {
            PartitionType::Scotch(n) => {
                mesh.compute_elem_to_elems();
                mesh.partition_scotch(n)?;
            }
            PartitionType::Metis(n) => {
                mesh.compute_elem_to_elems();
                mesh.partition_metis(n)?;
            }
            PartitionType::None => {
                info!("Using the existing partition");
            }
        }

        // Get the partition interfaces
        let max_ftag = mesh.ftags().max().unwrap_or(0);
        let (bdy_tag, _) = mesh.add_boundary_faces();
        assert_eq!(mesh.n_tagged_faces(bdy_tag), 0);

        let partition_tags = mesh.etags().collect::<FxHashSet<_>>();
        let partition_tags = partition_tags.iter().copied().collect::<Vec<_>>();
        debug!("Partition tags: {:?}", partition_tags);

        // Use negative tags for interfaces
        mesh.mut_ftags().for_each(|t| {
            if *t > max_ftag {
                *t = -*t;
            }
        });

        Ok(Self {
            mesh,
            partition_tags,
        })
    }

    /// Get the partition quality (ration of the number of interface faces to the total number of faces)
    pub fn partition_quality(&self) -> Result<f64> {
        self.mesh.partition_quality()
    }

    /// Get an iterator over the partitiona as SubSimplexMeshes
    pub fn partitions(&self) -> impl Iterator<Item = SubSimplexMesh<D, E>> + '_ {
        self.partition_tags
            .iter()
            .map(|&t| self.mesh.extract_tag(t))
    }

    /// Get an element tag that is 2 for the cells that are neighbors of level `n_layers` of the partition interface
    /// (i.e. the faces with a <0 tag)
    pub fn flag_interface(mesh: &SimplexMesh<D, E>, n_layers: Idx) -> Vec<Tag> {
        let mut new_etag = vec![1; mesh.n_elems() as usize];

        let mut flag = vec![false; mesh.n_verts() as usize];
        mesh.faces()
            .zip(mesh.ftags())
            .filter(|(_, t)| *t < 0)
            .flat_map(|(f, _)| f)
            .for_each(|i| flag[i as usize] = true);

        for _ in 0..n_layers {
            mesh.elems().zip(new_etag.iter_mut()).for_each(|(e, t)| {
                if e.iter().any(|&i_vert| flag[i_vert as usize]) {
                    *t = 2;
                }
            });
            mesh.elems()
                .zip(new_etag.iter())
                .filter(|(_, t)| **t == 2)
                .flat_map(|(e, _)| e)
                .for_each(|i_vert| flag[i_vert as usize] = true);
        }

        new_etag
    }

    /// Get a SubSimplexMesh containing the elements that are neighbors of level `n_layers` of the partition interface
    /// (i.e. the faces with a <0 tag)
    pub fn interface(&mut self, n_layers: Idx) -> SubSimplexMesh<D, E> {
        let new_etags = Self::flag_interface(&self.mesh, n_layers);
        let tmp = self.mesh.etags().collect::<Vec<_>>();
        self.mesh
            .mut_etags()
            .enumerate()
            .for_each(|(i, t)| *t = new_etags[i]);
        let res = self.mesh.extract(|t| t == 2);
        self.mesh
            .mut_etags()
            .enumerate()
            .for_each(|(i, t)| *t = tmp[i]);
        res
    }

    /// Remesh using domain decomposition
    pub fn remesh<M: Metric<D>, G: Geometry<D>>(
        &mut self,
        m: &[M],
        geom: &G,
        params: RemesherParams,
        repart: PartitionType,
    ) -> Result<Self> {
        let mut res = SimplexMesh::new(Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut interface_mesh =
            SimplexMesh::new(Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut interface_m = Vec::new();

        for submesh in self.partitions() {
            let mut local_mesh = submesh.mesh;
            local_mesh.compute_topology();

            let local_m: Vec<_> = submesh
                .parent_vert_ids
                .iter()
                .map(|&i| m[i as usize])
                .collect();
            let mut local_remesher = Remesher::new(&local_mesh, &local_m, geom)?;
            local_remesher.remesh(params.clone(), geom);

            let mut local_mesh = local_remesher.to_mesh(true);
            let local_m = local_remesher.metrics();

            let new_etags = Self::flag_interface(&local_mesh, 2);

            local_mesh
                .mut_etags()
                .zip(new_etags.iter())
                .for_each(|(t0, t1)| *t0 = *t1);
            let (bdy_tag, interface_tags) = local_mesh.add_boundary_faces();
            assert_eq!(local_mesh.n_tagged_faces(bdy_tag), 0);
            if interface_tags.is_empty() {
                warn!("All the elements are in the interface");
            } else {
                assert_eq!(interface_tags.len(), 1);
                let tag = interface_tags.keys().next().unwrap();
                local_mesh.update_face_tags(|t| if t == *tag { Tag::MIN } else { t });
            }

            res.add(&local_mesh, |t| t == 1, |_| true, None::<fn(Tag) -> bool>);
            let (ids, _, _) = interface_mesh.add(
                &local_mesh,
                |t| t == 2,
                |_t| true,
                Some(|t| t != Tag::MIN && t < 0),
            );
            interface_m.extend(ids.iter().map(|&i| local_m[i as usize]));
        }

        interface_mesh.remove_faces(|t| t < 0 && t > Tag::MIN);
        interface_mesh.compute_topology();
        let mut interface_remesher = Remesher::new(&interface_mesh, &interface_m, geom)?;
        interface_remesher.remesh(params, geom);

        let interface_mesh = interface_remesher.to_mesh(true);

        res.add(&interface_mesh, |_| true, |_| true, Some(|t| t == Tag::MIN));
        res.remove_faces(|t| t < 0);

        match repart {
            PartitionType::None => {
                res.mut_etags().for_each(|t| *t = 1);
                Ok(Self::new(res, repart)?)
            }
            _ => Ok(Self::new(res, repart)?),
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        domain_decomposition::{DomainDecomposition, PartitionType},
        geometry::NoGeometry,
        mesh::Point,
        metric::IsoMetric,
        remesher::RemesherParams,
        test_meshes::test_mesh_2d,
        Result,
    };

    #[test]
    fn test_domain_decomposition_2d_metis() -> Result<()> {
        use std::collections::HashMap;

        use log::info;

        use crate::init_log;

        init_log("warning");
        let mesh = test_mesh_2d().split().split().split().split().split();

        let mut dd = DomainDecomposition::new(mesh, PartitionType::Metis(4))?;

        info!("Partition quality: {:?}", dd.partition_quality().unwrap());

        let mut f = vec![0.0; dd.mesh.n_verts() as usize];

        for sub_mesh in dd.partitions() {
            sub_mesh
                .parent_vert_ids
                .iter()
                .for_each(|&i| f[i as usize] += 1.0);
        }

        let mut data = HashMap::new();
        data.insert(String::from("u"), f.as_slice());

        dd.mesh.write_vtk("dd.vtu", Some(data), None)?;

        let h = |p: Point<2>| {
            let x = p[0];
            let y = p[1];
            let hmin = 0.001;
            let hmax = 0.1;
            let sigma: f64 = 0.25;
            hmin + (hmax - hmin)
                * (1.0 - f64::exp(-((x - 0.5).powi(2) + (y - 0.35).powi(2)) / sigma.powi(2)))
        };

        let m: Vec<_> = (0..dd.mesh.n_verts())
            .map(|i| IsoMetric::<2>::from(h(dd.mesh.vert(i))))
            .collect();

        let res = dd.remesh(
            &m,
            &NoGeometry(),
            RemesherParams::default(),
            PartitionType::None,
        )?;

        let mesh = res.mesh;
        mesh.write_vtk("res.vtu", None, None)?;
        mesh.boundary().0.write_vtk("res_bdy.vtu", None, None)?;

        let n = mesh.n_verts();
        for i in 0..n {
            let vi = mesh.vert(i);
            for j in i + 1..n {
                let vj = mesh.vert(j);
                let d = (vj - vi).norm();
                assert!(d > 1e-8, "{}, {}, {:?}, {:?}", i, j, vi, vj);
            }
        }

        mesh.check()?;

        Ok(())
    }
}
