use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "metis")]
use tmesh::mesh::partition::{MetisKWay, MetisPartitioner, MetisRecursive};
use tmesh::{
    io::{VTUEncoding, VTUFile},
    mesh::{
        Cell, Face, Mesh,
        partition::{HilbertPartitioner, PartitionType},
    },
    spatialindex::ObjectIndex,
};

use super::{
    Edge, Elem, HasTmeshImpl, Point, SimplexMesh, SubSimplexMesh, Tetrahedron, Triangle, Vertex,
};
use crate::{Idx, Result, Tag};

fn to_cell<const C: usize, E: Elem>(e: E) -> Cell<C>
where
    Cell<C>: Default,
{
    let mut res = Cell::<C>::default();
    res.iter_mut().zip(e).for_each(|(x, y)| *x = y as usize);
    res
}

fn to_elem<const C: usize, E: Elem>(e: Cell<C>) -> E
where
    Cell<C>: Default,
{
    E::from_iter(e.iter().map(|&i| i as Idx))
}

/// Implement Mesh for simple meshes made of Vec of Vertex, Cell and Faces
macro_rules! impl_mesh {
    ($name: ident, $dim: expr, $cell_dim: expr, $face_dim: expr) => {
        impl Mesh<$dim, $cell_dim, $face_dim> for SimplexMesh<$dim, $name> {
            fn empty() -> Self {
                Self::empty()
            }

            fn n_verts(&self) -> usize {
                self.n_verts() as usize
            }

            fn vert(&self, i: usize) -> Point<$dim> {
                self.vert(i as Idx)
            }

            fn par_verts(&self) -> impl IndexedParallelIterator<Item = Point<$dim>> + Clone + '_ {
                self.par_verts()
            }

            fn verts(&self) -> impl ExactSizeIterator<Item = Point<$dim>> + Clone + '_ {
                self.verts()
            }

            fn add_verts<I: ExactSizeIterator<Item = Point<$dim>>>(&mut self, v: I) {
                self.verts.reserve(v.len());
                for x in v {
                    self.verts.push(x);
                }
            }

            fn n_elems(&self) -> usize {
                self.n_elems() as usize
            }

            fn elem(&self, i: usize) -> Cell<$cell_dim> {
                to_cell(self.elem(i as Idx))
            }

            fn par_elems(
                &self,
            ) -> impl IndexedParallelIterator<Item = Cell<$cell_dim>> + Clone + '_ {
                self.par_elems().map(to_cell)
            }

            fn elems(&self) -> impl ExactSizeIterator<Item = Cell<$cell_dim>> + Clone + '_ {
                self.elems().map(to_cell)
            }

            fn add_elems<
                I1: ExactSizeIterator<Item = Cell<$cell_dim>>,
                I2: ExactSizeIterator<Item = tmesh::Tag>,
            >(
                &mut self,
                elems: I1,
                etags: I2,
            ) {
                self.elems.reserve(elems.len());
                self.etags.reserve(etags.len());
                for (e, t) in elems.zip(etags) {
                    self.elems.push(to_elem(e));
                    self.etags.push(t.try_into().unwrap());
                }
            }

            fn clear_elems(&mut self) {
                self.elems.clear();
                self.etags.clear();
            }

            fn add_elems_and_tags<I: ExactSizeIterator<Item = (Cell<$cell_dim>, tmesh::Tag)>>(
                &mut self,
                elems_and_tags: I,
            ) {
                self.elems.reserve(elems_and_tags.len());
                self.etags.reserve(elems_and_tags.len());
                for (e, t) in elems_and_tags {
                    self.elems.push(to_elem(e));
                    self.etags.push(t.try_into().unwrap());
                }
            }

            fn etag(&self, i: usize) -> tmesh::Tag {
                self.etag(i as Idx).try_into().unwrap()
            }

            fn par_etags(&self) -> impl IndexedParallelIterator<Item = tmesh::Tag> + Clone + '_ {
                self.par_etags().map(|x| x.try_into().unwrap())
            }

            fn etags(&self) -> impl ExactSizeIterator<Item = tmesh::Tag> + Clone + '_ {
                self.etags().map(|x| x.try_into().unwrap())
            }

            fn n_faces(&self) -> usize {
                self.n_faces() as usize
            }

            fn face(&self, i: usize) -> Face<$face_dim> {
                to_cell(self.face(i as Idx))
            }

            fn par_faces(
                &self,
            ) -> impl IndexedParallelIterator<Item = Face<$face_dim>> + Clone + '_ {
                self.par_faces().map(to_cell)
            }

            fn faces(&self) -> impl ExactSizeIterator<Item = Face<$face_dim>> + Clone + '_ {
                self.faces().map(to_cell)
            }

            fn add_faces<
                I1: ExactSizeIterator<Item = Face<$face_dim>>,
                I2: ExactSizeIterator<Item = tmesh::Tag>,
            >(
                &mut self,
                faces: I1,
                ftags: I2,
            ) {
                self.faces.reserve(faces.len());
                self.ftags.reserve(ftags.len());
                for (e, t) in faces.zip(ftags) {
                    self.faces.push(to_elem(e));
                    self.ftags.push(t.try_into().unwrap());
                }
            }

            fn clear_faces(&mut self) {
                self.faces.clear();
                self.ftags.clear();
            }

            fn add_faces_and_tags<I: ExactSizeIterator<Item = (Face<$face_dim>, tmesh::Tag)>>(
                &mut self,
                faces_and_tags: I,
            ) {
                self.faces.reserve(faces_and_tags.len());
                self.ftags.reserve(faces_and_tags.len());
                for (e, t) in faces_and_tags {
                    self.faces.push(to_elem(e));
                    self.ftags.push(t.try_into().unwrap());
                }
            }

            fn ftag(&self, i: usize) -> tmesh::Tag {
                self.ftag(i as Idx).try_into().unwrap()
            }

            fn par_ftags(&self) -> impl IndexedParallelIterator<Item = tmesh::Tag> + Clone + '_ {
                self.par_ftags().map(|x| x.try_into().unwrap())
            }

            fn ftags(&self) -> impl ExactSizeIterator<Item = tmesh::Tag> + Clone + '_ {
                self.ftags().map(|x| x.try_into().unwrap())
            }

            fn invert_elem(&mut self, i: usize) {
                self.elems.index_mut(i as Idx).invert()
            }

            fn invert_face(&mut self, i: usize) {
                self.faces.index_mut(i as Idx).invert()
            }

            fn set_etags<I: ExactSizeIterator<Item = tmesh::Tag>>(&mut self, tags: I) {
                self.mut_etags()
                    .zip(tags)
                    .for_each(|(x, y)| *x = y.try_into().unwrap())
            }
        }

        impl HasTmeshImpl<$dim, $name> for SimplexMesh<$dim, $name> {
            fn elem_tree(&self) -> ObjectIndex<$dim> {
                ObjectIndex::new(self)
            }

            fn vtu_writer(&self) -> VTUFile {
                VTUFile::from_mesh(self, VTUEncoding::Binary)
            }

            fn partition_simple(&mut self, ptype: PartitionType) -> Result<(f64, f64)> {
                match ptype {
                    PartitionType::Hilbert(n) => self.partition::<HilbertPartitioner>(n, None),
                    PartitionType::RCM(_) => unimplemented!(),
                    PartitionType::KMeans(_) => unimplemented!(),
                    #[cfg(feature = "metis")]
                    PartitionType::MetisRecursive(n) => {
                        self.partition::<MetisPartitioner<MetisRecursive>>(n, None)
                    }
                    #[cfg(feature = "metis")]
                    PartitionType::MetisKWay(n) => {
                        self.partition::<MetisPartitioner<MetisKWay>>(n, None)
                    }
                    PartitionType::None => unreachable!(),
                }
            }

            fn boundary_flag(&self) -> Vec<bool> {
                <Self as Mesh<$dim, $cell_dim, $face_dim>>::boundary_flag(self)
            }

            fn extract_tag(&self, tag: Tag) -> SubSimplexMesh<$dim, $name> {
                let mut res = SimplexMesh::<$dim, $name>::empty();
                let (parent_vert_ids, parent_elem_ids, parent_face_ids) =
                    Mesh::add(&mut res, self, |t| t == tag, |_| true, None);
                // res.fix_orientation(&res.all_faces());
                // res.check_simple().unwrap();
                SubSimplexMesh::<$dim, $name> {
                    mesh: res,
                    parent_vert_ids: parent_vert_ids.iter().cloned().map(|i| i as Idx).collect(),
                    parent_elem_ids: parent_elem_ids.iter().cloned().map(|i| i as Idx).collect(),
                    parent_face_ids: parent_face_ids.iter().cloned().map(|i| i as Idx).collect(),
                }
            }

            fn check_simple(&self) -> Result<()> {
                let all_faces = self.all_faces();
                <Self as Mesh<$dim, $cell_dim, $face_dim>>::check(self, &all_faces)
            }

            fn remove_faces<F1: FnMut(Tag) -> bool>(&mut self, face_filter: F1) {
                <Self as Mesh<$dim, $cell_dim, $face_dim>>::remove_faces(self, face_filter);
            }

            fn add<F1, F2>(
                &mut self,
                other: &Self,
                elem_filter: F1,
                face_filter: F2,
                merge_tol: Option<f64>,
            ) -> (Vec<usize>, Vec<usize>, Vec<usize>)
            where
                F1: FnMut(Tag) -> bool,
                F2: FnMut(Tag) -> bool,
            {
                <Self as Mesh<$dim, $cell_dim, $face_dim>>::add(
                    self,
                    other,
                    elem_filter,
                    face_filter,
                    merge_tol,
                )
            }

            fn fix_face_orientation(&mut self) {
                let all_faces = self.all_faces();
                self.fix_faces_orientation(&all_faces);
            }
        }
    };
}

impl_mesh!(Tetrahedron, 3, 4, 3);
impl_mesh!(Triangle, 3, 3, 2);
impl_mesh!(Triangle, 2, 3, 2);
impl_mesh!(Edge, 3, 2, 1);
impl_mesh!(Edge, 2, 2, 1);
impl_mesh!(Vertex, 3, 1, 0);
impl_mesh!(Vertex, 2, 1, 0);

#[cfg(test)]
mod tests {

    use crate::{
        Result,
        mesh::{
            Elem, HasTmeshImpl, Point, SimplexMesh, Tetrahedron, Triangle,
            test_meshes::{test_mesh_2d, test_mesh_3d},
        },
        metric::{AnisoMetric2d, AnisoMetric3d, Metric},
    };
    use tempfile::NamedTempFile;
    use tmesh::mesh::{Mesh, partition::PartitionType};

    #[test]
    fn test_2d_ascii() -> Result<()> {
        let mesh = test_mesh_2d().split();

        assert_eq!(mesh.n_verts(), 9);
        assert_eq!(mesh.n_faces(), 8);
        assert_eq!(mesh.n_elems(), 8);

        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".mesh";

        mesh.write_meshb(&fname)?;

        let mesh2 = SimplexMesh::<2, Triangle>::from_meshb(&fname)?;

        assert_eq!(mesh2.n_verts(), 9);
        assert_eq!(mesh2.n_faces(), 8);
        assert_eq!(mesh2.n_elems(), 8);

        let v0 = Point::<2>::new(0.5, 0.);
        let v1 = Point::<2>::new(0., 0.01);
        let m = AnisoMetric2d::from_sizes(&v0, &v1);
        let m = vec![m; mesh.n_verts() as usize];
        let m_vec = m.iter().copied().flatten().collect::<Vec<_>>();

        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".sol";

        mesh.write_solb(&m_vec, &fname)?;

        let (m2, n) = SimplexMesh::<2, Triangle>::read_solb(&fname)?;
        assert_eq!(n, 3);

        let m2 = m2
            .chunks(n)
            .map(AnisoMetric2d::from_slice)
            .collect::<Vec<_>>();

        for (x, y) in m
            .iter()
            .copied()
            .flatten()
            .zip(m2.iter().copied().flatten())
        {
            assert!((x - y).abs() < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_2d_binary() -> Result<()> {
        let mesh = test_mesh_2d().split();

        assert_eq!(mesh.n_verts(), 9);
        assert_eq!(mesh.n_faces(), 8);
        assert_eq!(mesh.n_elems(), 8);

        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".meshb";

        mesh.write_meshb(&fname)?;

        let mesh2 = SimplexMesh::<2, Triangle>::from_meshb(&fname)?;

        assert_eq!(mesh2.n_verts(), 9);
        assert_eq!(mesh2.n_faces(), 8);
        assert_eq!(mesh2.n_elems(), 8);

        let v0 = Point::<2>::new(0.5, 0.);
        let v1 = Point::<2>::new(0., 0.01);
        let m = AnisoMetric2d::from_sizes(&v0, &v1);
        let m = vec![m; mesh.n_verts() as usize];
        let m_vec = m.iter().copied().flatten().collect::<Vec<_>>();

        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".solb";

        mesh.write_solb(&m_vec, &fname)?;

        let (m2, n) = SimplexMesh::<2, Triangle>::read_solb(&fname)?;
        assert_eq!(n, 3);

        let m2 = m2
            .chunks(n)
            .map(AnisoMetric2d::from_slice)
            .collect::<Vec<_>>();

        for (x, y) in m
            .iter()
            .copied()
            .flatten()
            .zip(m2.iter().copied().flatten())
        {
            assert!((x - y).abs() < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_3d_ascii() -> Result<()> {
        let mesh = test_mesh_3d().split();

        assert_eq!(mesh.n_verts(), 26);
        assert_eq!(mesh.n_faces(), 48);
        assert_eq!(mesh.n_elems(), 40);

        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".mesh";

        mesh.write_meshb(&fname)?;

        let mesh2 = SimplexMesh::<3, Tetrahedron>::from_meshb(&fname)?;

        assert_eq!(mesh2.n_verts(), 26);
        assert_eq!(mesh2.n_faces(), 48);
        assert_eq!(mesh2.n_elems(), 40);

        let v0 = Point::<3>::new(0.5, 0., 0.0);
        let v1 = Point::<3>::new(0., 0.01, 0.0);
        let v2 = Point::<3>::new(0., 0., 0.1);
        let m = AnisoMetric3d::from_sizes(&v0, &v1, &v2);
        let m = vec![m; mesh.n_verts() as usize];
        let m_vec = m.iter().copied().flatten().collect::<Vec<_>>();

        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".sol";

        mesh.write_solb(&m_vec, &fname)?;

        let (m2, n) = SimplexMesh::<3, Tetrahedron>::read_solb(&fname)?;
        assert_eq!(n, 6);

        let m2 = m2
            .chunks(n)
            .map(AnisoMetric3d::from_slice)
            .collect::<Vec<_>>();

        for (x, y) in m
            .iter()
            .copied()
            .flatten()
            .zip(m2.iter().copied().flatten())
        {
            assert!((x - y).abs() < 1e-10);
        }

        Ok(())
    }

    #[test]
    fn test_3d_binary() -> Result<()> {
        let mesh = test_mesh_3d().split();

        assert_eq!(mesh.n_verts(), 26);
        assert_eq!(mesh.n_faces(), 48);
        assert_eq!(mesh.n_elems(), 40);

        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".meshb";

        mesh.write_meshb(&fname)?;

        let mesh2 = SimplexMesh::<3, Tetrahedron>::from_meshb(&fname)?;

        assert_eq!(mesh2.n_verts(), 26);
        assert_eq!(mesh2.n_faces(), 48);
        assert_eq!(mesh2.n_elems(), 40);

        let v0 = Point::<3>::new(0.5, 0., 0.0);
        let v1 = Point::<3>::new(0., 0.01, 0.0);
        let v2 = Point::<3>::new(0., 0., 0.1);
        let m = AnisoMetric3d::from_sizes(&v0, &v1, &v2);
        let m = vec![m; mesh.n_verts() as usize];
        let m_vec = m.iter().copied().flatten().collect::<Vec<_>>();

        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".solb";

        mesh.write_solb(&m_vec, &fname)?;

        let (m2, n) = SimplexMesh::<3, Tetrahedron>::read_solb(&fname)?;
        assert_eq!(n, 6);

        let m2 = m2
            .chunks(n)
            .map(AnisoMetric3d::from_slice)
            .collect::<Vec<_>>();

        for (x, y) in m
            .iter()
            .copied()
            .flatten()
            .zip(m2.iter().copied().flatten())
        {
            assert!((x - y).abs() < 1e-10);
        }

        Ok(())
    }

    fn mean_bandwidth_e2v<const D: usize, E: Elem>(mesh: &SimplexMesh<D, E>) -> f64 {
        let mut mean = 0;
        for i_elem in 0..mesh.n_elems() {
            let e = mesh.elem(i_elem);
            mean += e.iter().max().unwrap() - e.iter().min().unwrap();
        }
        f64::from(mean) / f64::from(mesh.n_elems())
    }

    fn mean_bandwidth_e2e<const D: usize, E: Elem>(mesh: &SimplexMesh<D, E>) -> f64 {
        let e2e = mesh.get_elem_to_elems().unwrap();
        let mut mean = 0;
        for i_elem in 0..mesh.n_elems() {
            let n = e2e.row(i_elem as usize);
            mean += n
                .iter()
                .map(|i| i32::abs(*i as i32 - i_elem as i32))
                .max()
                .unwrap();
        }
        f64::from(mean) / f64::from(mesh.n_elems())
    }

    #[test]
    fn test_hilbert_2d() {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        mesh.compute_elem_to_elems();

        let mean_e2v_before = mean_bandwidth_e2v(&mesh);
        let mean_e2e_before = mean_bandwidth_e2e(&mesh);

        let (mut mesh, _, _, _) = mesh.reorder_hilbert();

        mesh.compute_elem_to_elems();
        let mean_e2v_after = mean_bandwidth_e2v(&mesh);
        let mean_e2e_after = mean_bandwidth_e2e(&mesh);

        assert!(
            mean_e2v_after < 0.11 * mean_e2v_before,
            "{mean_e2v_after} {mean_e2v_before}"
        );
        assert!(
            mean_e2e_after < 1.3 * mean_e2e_before,
            "{mean_e2e_after} {mean_e2e_before}"
        );
    }

    #[test]
    fn test_hilbert_3d() {
        let mut mesh = test_mesh_3d().split().split().split();
        mesh.compute_elem_to_elems();

        let mean_e2v_before = mean_bandwidth_e2v(&mesh);
        let mean_e2e_before = mean_bandwidth_e2e(&mesh);

        let (mut mesh, _, _, _) = mesh.reorder_hilbert();

        mesh.compute_elem_to_elems();
        let mean_e2v_after = mean_bandwidth_e2v(&mesh);
        let mean_e2e_after = mean_bandwidth_e2e(&mesh);

        assert!(
            mean_e2v_after < 0.5 * mean_e2v_before,
            "{mean_e2v_after} {mean_e2v_before}"
        );
        assert!(
            mean_e2e_after < 1.1 * mean_e2e_before,
            "{mean_e2e_after} {mean_e2e_before}"
        );
    }

    #[test]
    fn test_partition_hilbert_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        let (q, _) = mesh.partition_simple(PartitionType::Hilbert(4))?;

        assert!(q < 0.025, "failed, q = {q}");

        Ok(())
    }

    #[test]
    fn test_partition_hilbert_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();
        let (q, _) = mesh.partition_simple(PartitionType::Hilbert(4))?;

        assert!(q < 0.027, "failed, q = {q}");

        Ok(())
    }

    // #[cfg(feature = "scotch")]
    // #[test]
    // fn test_partition_scotch_2d() -> Result<()> {
    //     let mut mesh = test_mesh_2d().split().split().split().split().split();
    //     mesh.compute_elem_to_elems();
    //     mesh.partition_scotch(4)?;

    //     let q = mesh.partition_quality()?;
    //     assert!(q < 0.03, "failed, q = {q}");

    //     Ok(())
    // }

    // #[cfg(feature = "scotch")]
    // #[test]
    // fn test_partition_scotch_3d() -> Result<()> {
    //     let mut mesh = test_mesh_3d().split().split().split().split();
    //     mesh.compute_elem_to_elems();
    //     mesh.partition_scotch(4)?;

    //     let q = mesh.partition_quality()?;
    //     assert!(q < 0.025, "failed, q = {q}");

    //     Ok(())
    // }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_2d() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        let (q, _) = mesh.partition_simple(PartitionType::MetisRecursive(4))?;
        assert!(q < 0.03, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_3d() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();
        let (q, _) = mesh.partition_simple(PartitionType::MetisRecursive(4))?;

        assert!(q < 0.025, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_2d_kway() -> Result<()> {
        let mut mesh = test_mesh_2d().split().split().split().split().split();
        let (q, _) = mesh.partition_simple(PartitionType::MetisKWay(4))?;

        assert!(q < 0.03, "failed, q = {q}");

        Ok(())
    }

    #[cfg(feature = "metis")]
    #[test]
    fn test_partition_metis_3d_kway() -> Result<()> {
        let mut mesh = test_mesh_3d().split().split().split().split();
        let (q, _) = mesh.partition_simple(PartitionType::MetisKWay(4))?;

        assert!(q < 0.022, "failed, q = {q}");

        Ok(())
    }
}
