use crate::{
    Result, Tag,
    geometry::Geometry,
    mesh::MeshTopology,
    metric::Metric,
    remesher::{Remesher, RemesherParams},
};
use log::{debug, warn};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashSet;
use serde::Serialize;
use std::{marker::PhantomData, sync::Mutex, time::Instant};
use tmesh::mesh::{GenericMesh, Mesh, Simplex, SubMesh, partition::Partitioner};

#[derive(Clone, Debug)]
pub struct ParallelRemesherParams {
    pub n_layers: u32,
    pub level: u32,
    pub max_levels: u32,
    pub min_verts: usize,
}

impl ParallelRemesherParams {
    #[must_use]
    pub const fn new(n_layers: u32, max_levels: u32, min_verts: usize) -> Self {
        Self {
            n_layers,
            level: 0,
            max_levels,
            min_verts,
        }
    }

    #[must_use]
    pub const fn n_layers(&self) -> u32 {
        self.n_layers
    }

    #[must_use]
    pub const fn level(&self) -> u32 {
        self.level
    }

    #[must_use]
    const fn next(&self, n_verts: usize) -> Option<Self> {
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

impl Default for ParallelRemesherParams {
    fn default() -> Self {
        Self::new(2, 1, 10000)
    }
}

#[derive(Default, Clone, Serialize)]
pub struct RemeshingInfo {
    pub n_verts_init: usize,
    pub n_verts_final: usize,
    pub time: f64,
}

#[derive(Default, Serialize)]
pub struct ParallelRemeshingInfo {
    info: RemeshingInfo,
    partition_time: f64,
    partition_quality: f64,
    partition_imbalance: f64,
    partitions: Vec<RemeshingInfo>,
    interface: Option<Box<Self>>,
}

impl ParallelRemeshingInfo {
    fn print_short(&self, indent: String) {
        let s = &self.info;
        if self.partition_quality > 0.0 {
            println!(
                "{} -> {} verts, partition quality = {}, partition imbalance = {}, {:.2e} secs",
                s.n_verts_init,
                s.n_verts_final,
                self.partition_quality,
                self.partition_imbalance,
                s.time,
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

    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self).unwrap()
    }
}

/// Domain decomposition
pub struct ParallelRemesher<const D: usize, M: Mesh<D>, P: Partitioner> {
    mesh: M,
    topo: MeshTopology,
    n_parts: usize,
    partition_tags: Vec<Tag>,
    partition_bdy_tags: Vec<Tag>,
    interface_bdy_tag: Tag,
    partition_time: f64,
    partition_quality: f64,
    partition_imbalance: f64,
    debug: bool,
    _p: PhantomData<P>,
}

impl<const D: usize, M: Mesh<D>, P: Partitioner> ParallelRemesher<D, M, P> {
    /// Create a new parallel remesher based on domain decomposition.
    /// If part is `PartitionType::Scotch(n)` or `PartitionType::Metis(n)` the mesh is partitionned into n subdomains using
    /// scotch / metis. If None, the element tag in `mesh` is used as the partition Id
    ///
    /// NB: the mesh element tags will be modified
    pub fn new(mut mesh: M, topo: MeshTopology, n_parts: usize) -> Result<Self> {
        // Partition if needed
        let now = Instant::now();
        let (partition_quality, partition_imbalance) = mesh.partition::<P>(n_parts, None)?;

        let partition_time = now.elapsed().as_secs_f64();

        // Get the partition interfaces
        let (bdy_tags, ifc_tags) = mesh.fix().unwrap();
        assert!(bdy_tags.is_empty());

        let partition_tags = mesh.etags().collect::<FxHashSet<_>>();
        let partition_tags = partition_tags.iter().copied().collect::<Vec<_>>();
        let partition_bdy_tags = ifc_tags.values().copied().collect::<Vec<_>>();
        debug!("Partition tags: {partition_tags:?}");

        // Use negative tags for interfaces
        mesh.ftags_mut().for_each(|t| {
            if partition_bdy_tags.contains(t) {
                *t = -*t;
            }
        });

        Ok(Self {
            mesh,
            topo,
            partition_tags,
            n_parts,
            partition_bdy_tags: ifc_tags.values().copied().collect::<Vec<_>>(),
            interface_bdy_tag: Tag::MIN,
            partition_time,
            partition_quality,
            partition_imbalance,
            debug: false,
            _p: PhantomData,
        })
    }

    pub const fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    #[must_use]
    pub const fn partitionned_mesh(&self) -> &M {
        &self.mesh
    }

    #[must_use]
    pub fn n_verts(&self) -> usize {
        self.mesh.n_verts()
    }

    /// Get a parallel iterator over the partitiona as SubSimplexMeshes
    #[must_use]
    pub fn par_partitions(&self) -> impl IndexedParallelIterator<Item = SubMesh<D, M>> + '_ {
        self.partition_tags
            .par_iter()
            .map(|&tag| SubMesh::new(&self.mesh, |t| t == tag))
    }

    /// Get an iterator over the partitiona as SubSimplexMeshes
    pub fn seq_partitions(&self) -> impl Iterator<Item = SubMesh<D, M>> + '_ {
        self.partition_tags
            .iter()
            .map(|&tag| SubMesh::new(&self.mesh, |t| t == tag))
    }

    /// Get an element tag that is 2 for the cells that are neighbors of level `n_layers` of the partition interface
    /// (i.e. the faces with a <0 tag)
    #[must_use]
    pub fn flag_interface(&self, mesh: &impl Mesh<D>, n_layers: u32) -> Vec<Tag> {
        let mut new_etag = vec![1; mesh.n_elems()];

        let mut flag = vec![false; mesh.n_verts()];
        mesh.faces()
            .zip(mesh.ftags())
            .filter(|(_, t)| self.is_partition_bdy(*t))
            .flat_map(|(f, _)| f)
            .for_each(|i| flag[i] = true);

        for _ in 0..n_layers {
            mesh.elems().zip(new_etag.iter_mut()).for_each(|(e, t)| {
                if e.into_iter().any(|i_vert| flag[i_vert]) {
                    *t = 2;
                }
            });
            mesh.elems()
                .zip(new_etag.iter())
                .filter(|(_, t)| **t == 2)
                .flat_map(|(e, _)| e)
                .for_each(|i_vert| flag[i_vert] = true);
        }

        new_etag
    }

    fn is_partition_bdy(&self, tag: Tag) -> bool {
        self.partition_bdy_tags.iter().any(|&x| -x == tag)
    }

    const fn is_interface_bdy(&self, tag: Tag) -> bool {
        tag == self.interface_bdy_tag
    }

    fn remesh_submesh<T: Metric<D>, G: Geometry<D>>(
        &self,
        m: &[T],
        geom: &G,
        params: &RemesherParams,
        submesh: SubMesh<D, M>,
    ) -> (GenericMesh<D, M::C>, Vec<T>) {
        let mut local_mesh = submesh.mesh;

        // to be consistent with the base topology
        local_mesh.etags_mut().for_each(|t| *t = 1);
        let mut topo = self.topo.topo().clone();
        topo.clear(|(_, t)| self.is_partition_bdy(t));

        let local_topo = MeshTopology::new_from(&local_mesh, topo);

        let local_m: Vec<_> = submesh.parent_vert_ids.iter().map(|&i| m[i]).collect();
        let mut local_remesher = Remesher::new(&local_mesh, &local_topo, &local_m, geom).unwrap();

        local_remesher.remesh(params, geom).unwrap();

        (local_remesher.to_mesh(true), local_remesher.metrics())
    }

    /// Remesh using domain decomposition
    #[allow(clippy::too_many_lines)]
    pub fn remesh<T: Metric<D>, G: Geometry<D>>(
        &self,
        m: &[T],
        geom: &G,
        params: RemesherParams,
        dd_params: &ParallelRemesherParams,
    ) -> Result<(GenericMesh<D, M::C>, ParallelRemeshingInfo, Vec<T>)> {
        let res = Mutex::new(GenericMesh::empty());
        let res_m = Mutex::new(Vec::new());
        let ifc = Mutex::new(GenericMesh::empty());
        let ifc_m = Mutex::new(Vec::new());

        let level = dd_params.level();

        let info = ParallelRemeshingInfo {
            info: RemeshingInfo {
                n_verts_init: self.mesh.n_verts(),
                ..Default::default()
            },
            partition_time: self.partition_time,
            partition_quality: self.partition_quality,
            partition_imbalance: self.partition_imbalance,
            partitions: vec![RemeshingInfo::default(); self.partition_tags.len()],
            ..Default::default()
        };
        let info = Mutex::new(info);

        if self.debug {
            let fname = format!("level_{level}_init.vtu");
            self.mesh.write_vtk(&fname)?;
        }

        let now = Instant::now();

        self.par_partitions()
            .enumerate()
            .for_each(|(i_part, submesh)| {
                if self.debug {
                    let fname = format!("level_{level}_part_{i_part}.vtu");
                    submesh.mesh.write_vtk(&fname).unwrap();
                }

                // Remesh the partition
                debug!("Remeshing level {level} / partition {i_part}");
                let n_verts_init = submesh.mesh.n_verts();
                let now = Instant::now();
                let (mut local_mesh, local_m) =
                    self.remesh_submesh(m, geom, &params.clone(), submesh);

                // Get the info
                let mut info = info.lock().unwrap();
                info.partitions[i_part] = RemeshingInfo {
                    n_verts_init,
                    n_verts_final: local_mesh.n_verts(),
                    time: now.elapsed().as_secs_f64(),
                };
                drop(info);

                // Flag elements with n_layers of the interfaces with tag 2, other with tag 1
                let new_etags = self.flag_interface(&local_mesh, dd_params.n_layers);
                local_mesh
                    .etags_mut()
                    .zip(new_etags.iter())
                    .for_each(|(t0, t1)| *t0 = *t1);
                let (bdy_tags, interface_tags) = local_mesh.fix().unwrap();
                assert!(bdy_tags.is_empty());

                // Flag the faces between elements tagged 1 and 2 as self.interface_bdy_tag
                if interface_tags.is_empty() {
                    warn!("All the elements are in the interface");
                } else {
                    assert_eq!(interface_tags.len(), 1);
                    let tag = interface_tags.values().next().unwrap();
                    local_mesh.ftags_mut().for_each(|t| {
                        if *t == *tag {
                            *t = self.interface_bdy_tag;
                        }
                    });
                }

                if self.debug {
                    let fname = format!("level_{level}_part_{i_part}_remeshed.vtu");
                    local_mesh.write_vtk(&fname).unwrap();
                }

                // Update res
                let mut res = res.lock().unwrap();
                let (ids, _, _) = res.add(&local_mesh, |t| t == 1, |_| true, Some(1e-12));
                if self.debug {
                    let fname = format!("level_{level}_part_{i_part}_res.vtu");
                    res.write_vtk(&fname).unwrap();
                }
                drop(res);
                let mut res_m = res_m.lock().unwrap();
                res_m.extend(ids.iter().map(|&i| local_m[i]));
                drop(res_m);

                // Update ifc
                let part_tag = 2 + i_part as Tag;
                local_mesh.etags_mut().for_each(|t| {
                    if *t == 2 {
                        *t = part_tag;
                    }
                });
                let mut ifc = ifc.lock().unwrap();
                let (ids, _, _) = ifc.add(&local_mesh, |t| t == part_tag, |_t| true, Some(1e-12));
                if self.debug {
                    let fname = format!("level_{level}_part_{i_part}_ifc.vtu");
                    ifc.write_vtk(&fname).unwrap();
                }
                drop(ifc);
                let mut ifc_m = ifc_m.lock().unwrap();
                ifc_m.extend(ids.iter().map(|&i| local_m[i]));
            });

        let mut ifc = ifc.into_inner().unwrap();
        if self.debug {
            let fname = format!("level_{level}_ifc.vtu");
            ifc.write_vtk(&fname).unwrap();
            let fname = format!("level_{level}_ifc_bdy.vtu");
            ifc.boundary::<GenericMesh<D, <M::C as Simplex>::FACE>>()
                .0
                .write_vtk(&fname)
                .unwrap();
        }

        // to be consistent with the base topology
        ifc.etags_mut().for_each(|t| *t = 1);
        ifc.remove_faces(|t| self.is_partition_bdy(t));
        for i in 0..ifc.n_faces() {
            if ifc.ftag(i) == Tag::MIN {
                ifc.invert_face(i);
            }
        }

        if self.debug {
            ifc.check(&ifc.all_faces()).unwrap();
        }

        let mut info = info.into_inner().unwrap();

        let ifc_topo = MeshTopology::new_from(&ifc, self.topo.topo().clone());
        let ifc_m = ifc_m.into_inner().unwrap();

        let (mut ifc, ifc_m) = if let Some(dd_params) = dd_params.next(ifc.n_verts()) {
            debug!("Remeshing level {level} / interface (parallel)");
            let mesh = ifc;
            let mut dd = ParallelRemesher::<_, _, P>::new(mesh, ifc_topo, self.n_parts)?;
            dd.set_debug(self.debug);
            dd.interface_bdy_tag = self.interface_bdy_tag + 1;
            let (ifc, interface_info, ifc_m) = dd.remesh(&ifc_m, geom, params, &dd_params)?;
            info.interface = Some(Box::new(interface_info));
            (ifc, ifc_m)
        } else {
            debug!("Remeshing level {level} / interface (seq)");
            let mut ifc_remesher = Remesher::new(&ifc, &ifc_topo, &ifc_m, geom)?;
            if self.debug {
                ifc_remesher.check().unwrap();
            }
            let n_verts_init = ifc.n_verts();
            let now = Instant::now();
            ifc_remesher.remesh(&params, geom)?;
            info.interface = Some(Box::new(ParallelRemeshingInfo {
                info: RemeshingInfo {
                    n_verts_init,
                    n_verts_final: ifc_remesher.n_verts(),
                    time: now.elapsed().as_secs_f64(),
                },
                partition_time: 0.0,
                partition_quality: 0.0,
                partition_imbalance: 0.0,
                partitions: Vec::new(),
                interface: None,
            }));
            (ifc_remesher.to_mesh(true), ifc_remesher.metrics())
        };

        if self.debug {
            let fname = format!("level_{level}_ifc_remeshed.vtu");
            ifc.write_vtk(&fname).unwrap();
        }

        // Merge res and ifc
        let mut res = res.into_inner().unwrap();
        if self.debug {
            res.check(&res.all_faces()).unwrap();
        }
        ifc.etags_mut().for_each(|t| *t = 2);
        let (ids, _, _) = res.add(&ifc, |_| true, |_| true, Some(1e-12));
        if self.debug {
            res.check(&res.all_faces()).unwrap();
        }
        let mut res_m = res_m.into_inner().unwrap();
        res_m.extend(ids.iter().map(|&i| ifc_m[i]));

        res.remove_faces(|t| self.is_interface_bdy(t));
        res.etags_mut().for_each(|t| *t = 1);
        if self.debug {
            res.check(&res.all_faces()).unwrap();
        }

        if self.debug {
            let fname = format!("level_{level}_final.vtu");
            res.write_vtk(&fname).unwrap();
        }

        info.info.n_verts_final = res.n_verts();
        info.info.time = now.elapsed().as_secs_f64() + self.partition_time;

        Ok((res, info, res_m))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Result,
        geometry::NoGeometry,
        mesh::{
            MeshTopology,
            test_meshes::{test_mesh_2d, test_mesh_3d},
        },
        metric::IsoMetric,
        remesher::{ParallelRemesher, ParallelRemesherParams, RemesherParams},
    };
    #[cfg(feature = "metis")]
    use tmesh::mesh::partition::{MetisPartitioner, MetisRecursive};
    use tmesh::{
        Vert2d, Vert3d,
        mesh::{
            Mesh,
            partition::{HilbertPartitioner, Partitioner},
        },
    };

    fn test_domain_decomposition_2d<P: Partitioner>(debug: bool, n_parts: usize) -> Result<()> {
        // use crate::init_log;
        // init_log("debug");
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.etags_mut().for_each(|t| *t = 1);
        let topo = MeshTopology::new(&mesh);

        let mut dd = ParallelRemesher::<_, _, P>::new(mesh, topo, n_parts)?;
        dd.set_debug(debug);

        let h = |p: Vert2d| {
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

        let dd_params = ParallelRemesherParams::new(2, 1, 0);
        let (mesh, _, _) = dd.remesh(&m, &NoGeometry(), RemesherParams::default(), &dd_params)?;

        if debug {
            mesh.write_vtk("res.vtu")?;
            mesh.write_vtk("res_bdy.vtu")?;
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

        mesh.check(&mesh.all_faces())?;

        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_dd_2d_hilbert_1() {
        test_domain_decomposition_2d::<HilbertPartitioner>(false, 1).unwrap();
    }

    #[test]
    fn test_dd_2d_hilbert_2() -> Result<()> {
        test_domain_decomposition_2d::<HilbertPartitioner>(false, 2)
    }

    #[test]
    fn test_dd_2d_hilbert_3() -> Result<()> {
        test_domain_decomposition_2d::<HilbertPartitioner>(false, 3)
    }

    #[test]
    fn test_dd_2d_hilbert_4() -> Result<()> {
        test_domain_decomposition_2d::<HilbertPartitioner>(false, 4)
    }

    #[test]
    fn test_dd_2d_hilbert_5() -> Result<()> {
        test_domain_decomposition_2d::<HilbertPartitioner>(false, 5)
    }

    #[cfg(feature = "metis")]
    #[test]
    #[should_panic]
    fn test_dd_2d_metis_1() {
        use tmesh::mesh::partition::{MetisPartitioner, MetisRecursive};

        test_domain_decomposition_2d::<MetisPartitioner<MetisRecursive>>(false, 1).unwrap();
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_2d_metis_2() -> Result<()> {
        test_domain_decomposition_2d::<MetisPartitioner<MetisRecursive>>(false, 2)
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_2d_metis_3() -> Result<()> {
        test_domain_decomposition_2d::<MetisPartitioner<MetisRecursive>>(false, 3)
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_2d_metis_4() -> Result<()> {
        test_domain_decomposition_2d::<MetisPartitioner<MetisRecursive>>(false, 4)
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_2d_metis_5() -> Result<()> {
        test_domain_decomposition_2d::<MetisPartitioner<MetisRecursive>>(false, 5)
    }

    fn test_domain_decomposition_3d<P: Partitioner>(debug: bool, n_parts: usize) -> Result<()> {
        // use crate::init_log;
        // init_log("warning");
        let mesh = test_mesh_3d().split().split().split();
        let topo = MeshTopology::new(&mesh);
        let dd = ParallelRemesher::<_, _, P>::new(mesh, topo, n_parts)?;
        // dd.set_debug(true);

        let h = |p: Vert3d| {
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

        let dd_params = ParallelRemesherParams::new(2, 2, 0);
        let (mesh, _, _) = dd.remesh(&m, &NoGeometry(), RemesherParams::default(), &dd_params)?;

        if debug {
            mesh.write_vtk("res.vtu")?;
            mesh.write_vtk("res_bdy.vtu")?;
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
        mesh.check(&mesh.all_faces())?;

        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_dd_3d_hilbert_1() {
        test_domain_decomposition_3d::<HilbertPartitioner>(false, 1).unwrap();
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "Too long for debug build")]
    fn test_dd_3d_hilbert_2() -> Result<()> {
        test_domain_decomposition_3d::<HilbertPartitioner>(false, 2)
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "Too long for debug build")]
    fn test_dd_3d_hilbert_3() -> Result<()> {
        test_domain_decomposition_3d::<HilbertPartitioner>(false, 3)
    }

    #[test]
    fn test_dd_3d_hilbert_4() -> Result<()> {
        test_domain_decomposition_3d::<HilbertPartitioner>(false, 4)
    }

    #[test]
    fn test_dd_3d_hilbert_5() -> Result<()> {
        test_domain_decomposition_3d::<HilbertPartitioner>(false, 5)
    }

    #[cfg(feature = "metis")]
    #[test]
    #[should_panic]
    fn test_dd_3d_metis_1() {
        test_domain_decomposition_3d::<MetisPartitioner<MetisRecursive>>(false, 1).unwrap();
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_3d_metis_2() -> Result<()> {
        test_domain_decomposition_3d::<MetisPartitioner<MetisRecursive>>(false, 2)
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_3d_metis_3() -> Result<()> {
        test_domain_decomposition_3d::<MetisPartitioner<MetisRecursive>>(false, 3)
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_3d_metis_4() -> Result<()> {
        test_domain_decomposition_3d::<MetisPartitioner<MetisRecursive>>(false, 4)
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_dd_3d_metis_5() -> Result<()> {
        test_domain_decomposition_3d::<MetisPartitioner<MetisRecursive>>(false, 5)
    }
}
