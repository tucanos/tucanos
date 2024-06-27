use crate::{
    mesh::{Point, SimplexMesh},
    topo_elems::Elem,
    Idx, Result, Tag,
};
use minimeshb::{reader::MeshbReader, writer::MeshbWriter};

// /// Reorder the entries (actually used only for symmetric tensors) to ensure consistency between
// /// with the conventions used the meshb format
// fn field_order(dim: usize, field_type: GmfFieldTypes) -> Vec<usize> {
//     match dim {
//         2 => match field_type {
//             GmfFieldTypes::Scalar => vec![0],
//             GmfFieldTypes::Vector => vec![0, 1],
//             GmfFieldTypes::Metric => vec![0, 2, 1],
//         },
//         3 => match field_type {
//             GmfFieldTypes::Scalar => vec![0],
//             GmfFieldTypes::Vector => vec![0, 1, 2],
//             GmfFieldTypes::Metric => vec![0, 2, 5, 1, 4, 3],
//         },
//         _ => unreachable!(),
//     }
// }

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    pub fn write_meshb(&self, file_name: &str) -> Result<()> {
        let mut writer = MeshbWriter::new(file_name, 2, D as u8)?;

        writer.write_vertices(self.verts().map(Into::into), (0..self.n_verts()).map(|_| 1))?;

        match E::N_VERTS {
            3 => {
                writer.write_triangles(
                    self.elems()
                        .map(|v| [v[0].into(), v[1].into(), v[2].into()]),
                    self.etags().map(Into::into),
                )?;
                writer.write_edges(
                    self.faces().map(|v| [v[0].into(), v[1].into()]),
                    self.ftags().map(Into::into),
                )?;
            }
            4 => {
                writer.write_tetrahedra(
                    self.elems()
                        .map(|v| [v[0].into(), v[1].into(), v[2].into(), v[3].into()]),
                    self.etags().map(Into::into),
                )?;
                writer.write_triangles(
                    self.faces()
                        .map(|v| [v[0].into(), v[1].into(), v[2].into()]),
                    self.ftags().map(Into::into),
                )?;
            }
            _ => unreachable!(),
        }
        writer.close();

        Ok(())
    }

    fn write_solb_it<const N: usize, F: FnMut(&[f64]) -> [f64; N]>(
        &self,
        arr: &[f64],
        file_name: &str,
        f: F,
    ) -> Result<()> {
        assert_eq!(arr.len(), N * self.n_verts() as usize);

        let mut writer = MeshbWriter::new(file_name, 2, D as u8)?;
        writer.write_solution(arr.chunks(N).map(f))?;
        writer.close();

        Ok(())
    }

    pub fn write_solb(&self, arr: &[f64], file_name: &str) -> Result<()> {
        let n_comp = arr.len() / self.n_verts() as usize;
        match D {
            2 => match n_comp {
                1 => self.write_solb_it::<1, _>(arr, file_name, |x| [x[0]])?,
                2 => self.write_solb_it::<2, _>(arr, file_name, |x| [x[0], x[1]])?,
                3 => self.write_solb_it::<3, _>(arr, file_name, |x| [x[0], x[2], x[1]])?,
                _ => unreachable!(),
            },
            3 => match n_comp {
                1 => self.write_solb_it::<1, _>(arr, file_name, |x| [x[0]])?,
                3 => self.write_solb_it::<3, _>(arr, file_name, |x| [x[0], x[1], x[2]])?,
                6 => self.write_solb_it::<6, _>(arr, file_name, |x| {
                    [x[0], x[3], x[1], x[5], x[4], x[2]]
                })?, // [0, 2, 5, 1, 4, 3]
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }

        Ok(())
    }

    pub fn read_meshb(file_name: &str) -> Result<Self> {
        let mut reader = MeshbReader::new(file_name)?;
        assert_eq!(reader.dimension(), D as u8);

        let it = reader.read_vertices::<D>()?;
        let mut verts = Vec::with_capacity(it.len());
        for (v, _) in it {
            verts.push(Point::<D>::from_iterator(v.iter().copied()));
        }

        let mut elems = Vec::new();
        let mut etags = Vec::new();
        let mut faces = Vec::new();
        let mut ftags = Vec::new();
        match E::N_VERTS {
            3 => {
                let it = reader.read_triangles()?;
                elems.reserve(it.len());
                etags.reserve(it.len());
                for (e, t) in it {
                    elems.push(E::from_iter(e.iter().map(|&i| i as Idx)));
                    etags.push(t as Tag);
                }
                let it = reader.read_edges()?;
                faces.reserve(it.len());
                ftags.reserve(it.len());
                for (e, t) in it {
                    faces.push(E::Face::from_iter(e.iter().map(|&i| i as Idx)));
                    ftags.push(t as Tag);
                }
            }
            4 => {
                let it = reader.read_tetrahedra()?;
                elems.reserve(it.len());
                etags.reserve(it.len());
                for (e, t) in it {
                    elems.push(E::from_iter(e.iter().map(|&i| i as Idx)));
                    etags.push(t as Tag);
                }
                let it = reader.read_triangles()?;
                faces.reserve(it.len());
                ftags.reserve(it.len());
                for (e, t) in it {
                    faces.push(E::Face::from_iter(e.iter().map(|&i| i as Idx)));
                    ftags.push(t as Tag);
                }
            }
            _ => unreachable!(),
        }

        Ok(Self::new(verts, elems, etags, faces, ftags))
    }

    fn read_solb_it<const N: usize, F: FnMut([f64; N]) -> [f64; N]>(
        mut reader: MeshbReader,
        f: F,
    ) -> Result<Vec<f64>> {
        let sol = reader.read_solution::<N>()?;
        Ok(sol.flat_map(f).collect())
    }

    pub fn read_solb(file_name: &str) -> Result<(Vec<f64>, usize)> {
        let mut reader = MeshbReader::new(file_name)?;
        let d = reader.dimension();
        assert_eq!(d, D as u8);
        let m = reader.get_solution_size()?;

        let res = match d {
            2 => match m {
                1 => Self::read_solb_it::<1, _>(reader, |x| [x[0]])?,
                2 => Self::read_solb_it::<2, _>(reader, |x| [x[0], x[1]])?,
                3 => Self::read_solb_it::<3, _>(reader, |x| [x[0], x[2], x[1]])?,
                _ => unreachable!(),
            },
            3 => match m {
                1 => Self::read_solb_it::<1, _>(reader, |x| [x[0]])?,
                3 => Self::read_solb_it::<3, _>(reader, |x| [x[0], x[1], x[2]])?,
                6 => Self::read_solb_it::<6, _>(reader, |x| [x[0], x[2], x[5], x[1], x[4], x[3]])?,
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };

        Ok((res, m))
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        mesh::{Point, SimplexMesh},
        metric::{AnisoMetric2d, AnisoMetric3d, Metric},
        test_meshes::{test_mesh_2d, test_mesh_3d},
        topo_elems::{Tetrahedron, Triangle},
        Result,
    };
    use tempfile::NamedTempFile;

    #[test]
    fn test_2d_ascii() -> Result<()> {
        let mesh = test_mesh_2d().split();

        assert_eq!(mesh.n_verts(), 9);
        assert_eq!(mesh.n_faces(), 8);
        assert_eq!(mesh.n_elems(), 8);

        let file = NamedTempFile::new().unwrap();
        let fname = file.path().to_str().unwrap().to_owned() + ".mesh";

        mesh.write_meshb(&fname)?;

        let mesh2 = SimplexMesh::<2, Triangle>::read_meshb(&fname)?;

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

        let mesh2 = SimplexMesh::<2, Triangle>::read_meshb(&fname)?;

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

        let mesh2 = SimplexMesh::<3, Tetrahedron>::read_meshb(&fname)?;

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

        let mesh2 = SimplexMesh::<3, Tetrahedron>::read_meshb(&fname)?;

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

    #[test]
    #[ignore]
    fn test_3d_simple3d() -> Result<()> {
        let mesh = SimplexMesh::<3, Tetrahedron>::read_meshb("data/simple3d.meshb")?;

        assert_eq!(mesh.n_verts(), 242118);
        assert_eq!(mesh.n_faces(), 44830);
        assert_eq!(mesh.n_elems(), 1380547);

        Ok(())
    }
}
