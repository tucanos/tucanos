use crate::{
    geometry::Geometry,
    mesh::{SimplexMesh, SubSimplexMesh},
    metric::Metric,
    remesher::{Remesher, RemesherParams},
    topo_elems::Elem,
    Idx, Result, Tag,
};
use log::{debug, info, warn};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashSet;
use std::{sync::Mutex, time::Instant};

#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum PartitionType {
    Scotch(Idx),
    Metis(Idx),
    None,
}

pub struct DomainDecompositionRemeshingParams {
    n_layers: Idx,
    level: Idx,
    max_levels: Idx,
    min_verts: Idx,
}

impl DomainDecompositionRemeshingParams {
    #[must_use]
    pub const fn new(n_layers: Idx, max_levels: Idx, min_verts: Idx) -> Self {
        Self {
            n_layers,
            level: 0,
            max_levels,
            min_verts,
        }
    }

    #[must_use]
    pub const fn n_layers(&self) -> Idx {
        self.n_layers
    }

    #[must_use]
    pub const fn level(&self) -> Idx {
        self.level
    }

    #[must_use]
    const fn next(&self, n_verts: Idx) -> Option<Self> {
        if self.level + 1 < self.max_levels && n_verts > self.min_verts {
            Some(Self {
                n_layers: self.n_layers,
                level: self.level + 1,
                max_levels: self.max_levels,
                min_verts: self.min_verts,
            })
        } else {
            None
        }
    }
}

#[derive(Default, Clone)]
pub struct RemeshingStats {
    pub n_verts_init: Idx,
    pub n_verts_final: Idx,
    pub time: f64,
}

#[derive(Default)]
pub struct DomainDecompositionRemeshingStats {
    stats: RemeshingStats,
    partition_quality: f64,
    partitions: Vec<RemeshingStats>,
    interface: Option<Box<DomainDecompositionRemeshingStats>>,
}

impl DomainDecompositionRemeshingStats {
    fn print_short(&self, indent: String) {
        let s = &self.stats;
        if self.partition_quality > 0.0 {
            println!(
                "{} -> {} verts, partition quality = {}, {:.2e} secs",
                s.n_verts_init, s.n_verts_final, self.partition_quality, s.time,
            );
            for (i, s) in self.partitions.iter().enumerate() {
                println!(
                    "{indent} partition {i}: {} -> {} verts, {:.2e} secs",
                    s.n_verts_init, s.n_verts_final, s.time
                );
            }
            if let Some(ifc) = &self.interface {
                print!("{indent} interface: ");
                ifc.print_short(indent + "  ");
            }
        } else {
            println!(
                "{} -> {} verts, {:.2e} secs",
                s.n_verts_init, s.n_verts_final, s.time,
            );
        }
    }

    pub fn print_summary(&self) {
        self.print_short(String::from("  "));
    }
}

/// Domain decomposition
pub struct DomainDecomposition<const D: usize, E: Elem> {
    pub mesh: SimplexMesh<D, E>,
    partition_tags: Vec<Tag>,
    partition_bdy_tags: Vec<Tag>,
    partition_type: PartitionType,
    interface_bdy_tag: Tag,
    debug: bool,
}

impl<const D: usize, E: Elem> DomainDecomposition<D, E> {
    /// Create a new domain decomposition.
    /// If part is `PartitionType::Scotch(n)` or `PartitionType::Metis(n)` the mesh is partitionned into n subdomains using
    /// scotch / metis. If None, the element tag in `mesh` is used as the partition Id
    ///
    /// NB: the mesh element tags will be modified
    pub fn new(mut mesh: SimplexMesh<D, E>, partition_type: PartitionType) -> Result<Self> {
        // Partition if needed
        match partition_type {
            PartitionType::Scotch(n) => {
                assert!(n > 1, "Need at least 2 partitions");
                mesh.compute_elem_to_elems();
                mesh.partition_scotch(n)?;
            }
            PartitionType::Metis(n) => {
                assert!(n > 1, "Need at least 2 partitions");
                mesh.compute_elem_to_elems();
                mesh.partition_metis(n)?;
            }
            PartitionType::None => {
                info!("Using the existing partition");
            }
        }

        // Get the partition interfaces
        let (bdy_tags, ifc_tags) = mesh.add_boundary_faces();
        assert!(bdy_tags.is_empty());

        let partition_tags = mesh.etags().collect::<FxHashSet<_>>();
        let partition_tags = partition_tags.iter().copied().collect::<Vec<_>>();
        let partition_bdy_tags = ifc_tags.keys().copied().collect::<Vec<_>>();
        debug!("Partition tags: {:?}", partition_tags);

        // Use negative tags for interfaces
        mesh.mut_ftags().for_each(|t| {
            if partition_bdy_tags.iter().any(|&x| x == *t) {
                *t = -*t;
            }
        });

        Ok(Self {
            mesh,
            partition_tags,
            partition_type,
            partition_bdy_tags: ifc_tags.keys().copied().collect::<Vec<_>>(),
            interface_bdy_tag: Tag::MIN,
            debug: false,
        })
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Get the partition quality (ration of the number of interface faces to the total number of faces)
    pub fn partition_quality(&self) -> Result<f64> {
        self.mesh.partition_quality()
    }

    /// Get a parallel iterator over the partitiona as SubSimplexMeshes
    #[must_use]
    pub fn par_partitions(&self) -> impl IndexedParallelIterator<Item = SubSimplexMesh<D, E>> + '_ {
        self.partition_tags
            .par_iter()
            .map(|&t| self.mesh.extract_tag(t))
    }

    /// Get an iterator over the partitiona as SubSimplexMeshes
    pub fn seq_partitions(&self) -> impl Iterator<Item = SubSimplexMesh<D, E>> + '_ {
        self.partition_tags
            .iter()
            .map(|&t| self.mesh.extract_tag(t))
    }

    /// Get an element tag that is 2 for the cells that are neighbors of level `n_layers` of the partition interface
    /// (i.e. the faces with a <0 tag)
    #[must_use]
    pub fn flag_interface(&self, mesh: &SimplexMesh<D, E>, n_layers: Idx) -> Vec<Tag> {
        let mut new_etag = vec![1; mesh.n_elems() as usize];

        let mut flag = vec![false; mesh.n_verts() as usize];
        mesh.faces()
            .zip(mesh.ftags())
            .filter(|(_, t)| self.is_partition_bdy(*t))
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
        let new_etags = self.flag_interface(&self.mesh, n_layers);
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

    fn is_partition_bdy(&self, tag: Tag) -> bool {
        self.partition_bdy_tags.iter().any(|&x| -x == tag)
    }

    const fn is_interface_bdy(&self, tag: Tag) -> bool {
        tag == self.interface_bdy_tag
    }

    fn remesh_submesh<M: Metric<D>, G: Geometry<D>>(
        &self,
        m: &[M],
        geom: &G,
        params: RemesherParams,
        submesh: SubSimplexMesh<D, E>,
    ) -> (SimplexMesh<D, E>, Vec<M>) {
        let mut local_mesh = submesh.mesh;
        // to be consistent with the base topology
        local_mesh.mut_etags().for_each(|t| *t = 1);
        let mut topo = self.mesh.get_topology().unwrap().clone();
        topo.clear(|(_, t)| self.is_partition_bdy(t));
        local_mesh.compute_topology_from(topo);

        let local_m: Vec<_> = submesh
            .parent_vert_ids
            .iter()
            .map(|&i| m[i as usize])
            .collect();
        let mut local_remesher = Remesher::new(&local_mesh, &local_m, geom).unwrap();

        local_remesher.remesh(params, geom).unwrap();

        (local_remesher.to_mesh(true), local_remesher.metrics())
    }

    /// Remesh using domain decomposition
    #[allow(clippy::too_many_lines)]
    pub fn remesh<M: Metric<D>, G: Geometry<D>>(
        &mut self,
        m: &[M],
        geom: &G,
        params: RemesherParams,
        dd_params: DomainDecompositionRemeshingParams,
    ) -> Result<(SimplexMesh<D, E>, DomainDecompositionRemeshingStats)> {
        let res = Mutex::new(SimplexMesh::empty());
        let interface_mesh = Mutex::new(SimplexMesh::empty());
        let interface_m = Mutex::new(Vec::new());

        let level = dd_params.level();

        let stats = DomainDecompositionRemeshingStats {
            stats: RemeshingStats {
                n_verts_init: self.mesh.n_verts(),
                ..Default::default()
            },
            partition_quality: self.partition_quality()?,
            partitions: vec![RemeshingStats::default(); self.partition_tags.len()],
            ..Default::default()
        };
        let stats = Mutex::new(stats);

        let now = Instant::now();

        self.par_partitions()
            .enumerate()
            .for_each(|(i_part, submesh)| {
                if self.debug {
                    let fname = format!("level_{level}_part_{i_part}.vtu");
                    submesh.mesh.write_vtk(&fname, None, None).unwrap();
                }

                // Remesh the partition
                debug!("Remeshing level {level} / partition {i_part}");
                let n_verts_init = submesh.mesh.n_verts();
                let now = Instant::now();
                let (mut local_mesh, local_m) =
                    self.remesh_submesh(m, geom, params.clone(), submesh);

                // Get the stats
                let mut stats = stats.lock().unwrap();
                stats.partitions[i_part] = RemeshingStats {
                    n_verts_init,
                    n_verts_final: local_mesh.n_verts(),
                    time: now.elapsed().as_secs_f64(),
                };
                drop(stats);

                // Flag elements with n_layers of the interfaces with tag 2, other with tag 1
                let new_etags = self.flag_interface(&local_mesh, dd_params.n_layers);
                local_mesh
                    .mut_etags()
                    .zip(new_etags.iter())
                    .for_each(|(t0, t1)| *t0 = *t1);
                let (bdy_tags, interface_tags) = local_mesh.add_boundary_faces();
                assert!(bdy_tags.is_empty());

                // Flag the faces between elements tagged 1 and 2 as self.interface_bdy_tag
                if interface_tags.is_empty() {
                    warn!("All the elements are in the interface");
                } else {
                    assert_eq!(interface_tags.len(), 1);
                    let tag = interface_tags.keys().next().unwrap();
                    local_mesh.mut_ftags().for_each(|t| {
                        if *t == *tag {
                            *t = self.interface_bdy_tag;
                        }
                    });
                }

                if self.debug {
                    let fname = format!("level_{level}_part_{i_part}_remeshed.vtu");
                    local_mesh.write_vtk(&fname, None, None).unwrap();
                }

                // Update res
                let mut res = res.lock().unwrap();
                res.add(&local_mesh, |t| t == 1, |_| true, Some(1e-12));
                drop(res);

                // Update interface_mesh
                let part_tag = 2 + i_part as Tag;
                local_mesh.mut_etags().for_each(|t| {
                    if *t == 2 {
                        *t = part_tag;
                    }
                });
                let mut interface_mesh = interface_mesh.lock().unwrap();
                let (ids, _, _) =
                    interface_mesh.add(&local_mesh, |t| t == part_tag, |_t| true, Some(1e-12));
                drop(interface_mesh);
                let mut interface_m = interface_m.lock().unwrap();
                interface_m.extend(ids.iter().map(|&i| local_m[i as usize]));
            });

        let mut interface_mesh = interface_mesh.into_inner().unwrap();
        if self.debug {
            let fname = format!("level_{level}_interface.vtu");
            interface_mesh.write_vtk(&fname, None, None).unwrap();
            let fname = format!("level_{level}_interface_bdy.vtu");
            interface_mesh
                .boundary()
                .0
                .write_vtk(&fname, None, None)
                .unwrap();
        }

        // to be consistent with the base topology
        interface_mesh.mut_etags().for_each(|t| *t = 1);
        interface_mesh.remove_faces(|t| self.is_partition_bdy(t));
        if self.debug {
            interface_mesh.compute_face_to_elems();
            interface_mesh.check().unwrap();
        }

        let mut stats = stats.into_inner().unwrap();

        let topo = self.mesh.get_topology().unwrap().clone();
        interface_mesh.compute_topology_from(topo);
        let interface_m = interface_m.into_inner().unwrap();

        let mut interface_mesh = if let Some(dd_params) = dd_params.next(interface_mesh.n_verts()) {
            let mut dd = Self::new(interface_mesh, self.partition_type)?;
            dd.set_debug(self.debug);
            dd.interface_bdy_tag = self.interface_bdy_tag + 1;
            let (interface_mesh, interface_stats) =
                dd.remesh(&interface_m, geom, params, dd_params)?;
            stats.interface = Some(Box::new(interface_stats));
            interface_mesh
        } else {
            debug!("Remeshing level {level} / interface");
            let mut interface_remesher = Remesher::new(&interface_mesh, &interface_m, geom)?;
            if self.debug {
                interface_remesher.check().unwrap();
            }
            let n_verts_init = interface_mesh.n_verts();
            let now = Instant::now();
            interface_remesher.remesh(params, geom)?;
            stats.interface = Some(Box::new(DomainDecompositionRemeshingStats {
                stats: RemeshingStats {
                    n_verts_init,
                    n_verts_final: interface_remesher.n_verts(),
                    time: now.elapsed().as_secs_f64(),
                },
                partition_quality: -1.0,
                partitions: Vec::new(),
                interface: None,
            }));
            interface_remesher.to_mesh(true)
        };

        if self.debug {
            interface_mesh
                .write_vtk("level_{level}_interface_remeshed.vtu", None, None)
                .unwrap();
        }

        // Merge res and interface_mesh
        let mut res = res.into_inner().unwrap();
        interface_mesh.mut_etags().for_each(|t| *t = 2);
        res.add(&interface_mesh, |_| true, |_| true, Some(1e-12));
        if self.debug {
            res.compute_face_to_elems();
            res.check().unwrap();
        }
        res.remove_faces(|t| self.is_interface_bdy(t));
        res.mut_etags().for_each(|t| *t = 1);
        if self.debug {
            res.compute_face_to_elems();
            res.check().unwrap();
        }

        stats.stats.n_verts_final = res.n_verts();
        stats.stats.time = now.elapsed().as_secs_f64();

        Ok((res, stats))
    }
}

#[cfg(test)]
#[cfg(any(feature = "metis", feature = "scotch"))]
mod tests {

    use crate::{
        domain_decomposition::{
            DomainDecomposition, DomainDecompositionRemeshingParams, PartitionType,
        },
        geometry::NoGeometry,
        mesh::Point,
        metric::IsoMetric,
        remesher::RemesherParams,
        test_meshes::{test_mesh_2d, test_mesh_3d},
        Result,
    };

    fn test_domain_decomposition_2d(debug: bool, ptype: PartitionType) -> Result<()> {
        use log::info;
        // use crate::init_log;
        // init_log("warning");
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.mut_etags().for_each(|t| *t = 1);
        mesh.compute_topology();

        let mut dd = DomainDecomposition::new(mesh, ptype)?;

        info!("Partition quality: {:?}", dd.partition_quality().unwrap());

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

        let dd_params = DomainDecompositionRemeshingParams::new(2, 1, 0);
        let (mut mesh, _) = dd.remesh(&m, &NoGeometry(), RemesherParams::default(), dd_params)?;

        if debug {
            mesh.write_vtk("res.vtu", None, None)?;
            mesh.boundary().0.write_vtk("res_bdy.vtu", None, None)?;
        }

        let n = mesh.n_verts();
        for i in 0..n {
            let vi = mesh.vert(i);
            for j in i + 1..n {
                let vj = mesh.vert(j);
                let d = (vj - vi).norm();
                assert!(d > 1e-8, "{i}, {j}, {vi:?}, {vj:?}");
            }
        }

        mesh.compute_face_to_elems();
        mesh.check()?;

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    #[should_panic]
    fn test_dd_2d_metis_1() {
        test_domain_decomposition_2d(false, PartitionType::Metis(1)).unwrap();
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_2d_metis_2() -> Result<()> {
        test_domain_decomposition_2d(false, PartitionType::Metis(2))
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_2d_metis_3() -> Result<()> {
        test_domain_decomposition_2d(false, PartitionType::Metis(3))
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_2d_metis_4() -> Result<()> {
        test_domain_decomposition_2d(false, PartitionType::Metis(4))
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_2d_metis_5() -> Result<()> {
        test_domain_decomposition_2d(false, PartitionType::Metis(5))
    }

    #[cfg(feature = "scotch")]
    #[test]
    #[should_panic]
    fn test_dd_2d_scotch_1() {
        test_domain_decomposition_2d(false, PartitionType::Scotch(1)).unwrap();
    }

    #[cfg(feature = "scotch")]
    #[test]
    fn test_dd_2d_scotch_2() -> Result<()> {
        test_domain_decomposition_2d(false, PartitionType::Scotch(2))
    }

    #[cfg(feature = "scotch")]
    #[test]
    fn test_dd_2d_scotch_3() -> Result<()> {
        test_domain_decomposition_2d(false, PartitionType::Scotch(3))
    }

    #[cfg(feature = "scotch")]
    #[test]
    fn test_dd_2d_scotch_4() -> Result<()> {
        test_domain_decomposition_2d(false, PartitionType::Scotch(4))
    }

    #[cfg(feature = "scotch")]
    #[test]
    fn test_dd_2d_scotch_5() -> Result<()> {
        test_domain_decomposition_2d(false, PartitionType::Scotch(5))
    }

    fn test_domain_decomposition_3d(debug: bool, ptype: PartitionType) -> Result<()> {
        use log::info;
        // use crate::init_log;
        // init_log("warning");
        let mut mesh = test_mesh_3d().split().split().split();
        mesh.compute_topology();
        let mut dd = DomainDecomposition::new(mesh, ptype)?;
        // dd.set_debug(true);
        info!("Partition quality: {:?}", dd.partition_quality().unwrap());

        let h = |p: Point<3>| {
            let x = p[0];
            let y = p[1];
            let z = p[2];
            let hmin = 0.025;
            let hmax = 0.25;
            let sigma: f64 = 0.25;
            hmin + (hmax - hmin)
                * (1.0
                    - f64::exp(
                        -((x - 0.5).powi(2) + (y - 0.35).powi(2) + (z - 0.65).powi(2))
                            / sigma.powi(2),
                    ))
        };

        let m: Vec<_> = (0..dd.mesh.n_verts())
            .map(|i| IsoMetric::<3>::from(h(dd.mesh.vert(i))))
            .collect();

        let dd_params = DomainDecompositionRemeshingParams::new(2, 2, 0);
        let (mut mesh, _) = dd.remesh(&m, &NoGeometry(), RemesherParams::default(), dd_params)?;

        if debug {
            mesh.write_vtk("res.vtu", None, None)?;
            mesh.boundary().0.write_vtk("res_bdy.vtu", None, None)?;
        }

        let n = mesh.n_verts();
        for i in 0..n {
            let vi = mesh.vert(i);
            for j in i + 1..n {
                let vj = mesh.vert(j);
                let d = (vj - vi).norm();
                assert!(d > 1e-8, "{i}, {j}, {vi:?}, {vj:?}");
            }
        }
        mesh.compute_face_to_elems();
        mesh.check()?;

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    #[should_panic]
    fn test_dd_3d_metis_1() {
        test_domain_decomposition_3d(false, PartitionType::Metis(1)).unwrap();
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_3d_metis_2() -> Result<()> {
        test_domain_decomposition_3d(false, PartitionType::Metis(2))
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_3d_metis_3() -> Result<()> {
        test_domain_decomposition_3d(false, PartitionType::Metis(3))
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_3d_metis_4() -> Result<()> {
        test_domain_decomposition_3d(false, PartitionType::Metis(4))
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_3d_metis_5() -> Result<()> {
        test_domain_decomposition_3d(false, PartitionType::Metis(5))
    }

    #[cfg(feature = "scotch")]
    #[test]
    #[should_panic]
    fn test_dd_3d_scotch_1() {
        test_domain_decomposition_3d(false, PartitionType::Scotch(1)).unwrap();
    }

    #[cfg(feature = "scotch")]
    #[test]
    fn test_dd_3d_scotch_2() -> Result<()> {
        test_domain_decomposition_3d(false, PartitionType::Scotch(2))
    }

    #[cfg(feature = "scotch")]
    #[test]
    fn test_dd_3d_scotch_3() -> Result<()> {
        test_domain_decomposition_3d(false, PartitionType::Scotch(3))
    }

    #[cfg(feature = "scotch")]
    #[test]
    fn test_dd_3d_scotch_4() -> Result<()> {
        test_domain_decomposition_3d(false, PartitionType::Scotch(4))
    }

    #[cfg(feature = "scotch")]
    #[test]
    fn test_dd_3d_scotch_5() -> Result<()> {
        test_domain_decomposition_3d(false, PartitionType::Scotch(5))
    }
}
