extern crate libmeshb_sys;
use self::libmeshb_sys::{
    GmfCloseMesh, GmfGetLin, GmfGotoKwd, GmfKwdCod, GmfOpenMesh, GmfRead, GmfSca, GmfSetKwd,
    GmfSetLin, GmfStatKwd, GmfSymMat, GmfVec, GmfWrite,
};
use crate::{
    mesh::{Point, SimplexMesh},
    topo_elems::Elem,
    Error, Idx, Result, Tag,
};
use log::debug;
use std::ffi::{c_int, CString};
use std::ptr::addr_of_mut;

/// Reader for .mesh(b) / .sol(b) files (interface to libMeshb)
pub struct GmfReader {
    file: i64,
    dim: c_int,
    version: c_int,
}

#[derive(Clone, Copy, Debug)]
pub enum GmfElementTypes {
    Vertex = GmfKwdCod::GmfVertices as isize,
    Edge = GmfKwdCod::GmfEdges as isize,
    Triangle = GmfKwdCod::GmfTriangles as isize,
    Tetrahedron = GmfKwdCod::GmfTetrahedra as isize,
}

#[derive(Clone, Copy)]
pub enum GmfFieldTypes {
    Scalar = GmfSca as isize,
    Vector = GmfVec as isize,
    Metric = GmfSymMat as isize,
}

/// Reorder the entries (actually used only for symmetric tensors) to ensure consistency between
/// with the conventions used the meshb format
fn field_order(dim: usize, field_type: GmfFieldTypes) -> Vec<usize> {
    match dim {
        2 => match field_type {
            GmfFieldTypes::Scalar => vec![0],
            GmfFieldTypes::Vector => vec![0, 1],
            GmfFieldTypes::Metric => vec![0, 2, 1],
        },
        3 => match field_type {
            GmfFieldTypes::Scalar => vec![0],
            GmfFieldTypes::Vector => vec![0, 1, 2],
            GmfFieldTypes::Metric => vec![0, 2, 5, 1, 4, 3],
        },
        _ => unreachable!(),
    }
}

impl GmfReader {
    /// Create a new file
    #[must_use]
    pub fn new(fname: &str) -> Self {
        debug!("Open {} (read)", fname);
        let mut dim: c_int = 0;
        let mut version: c_int = 0;
        let cfname = CString::new(fname).unwrap();
        let file = unsafe {
            GmfOpenMesh(
                cfname.as_ptr(),
                GmfRead as c_int,
                addr_of_mut!(version),
                addr_of_mut!(dim),
            )
        };
        assert!(file == 0 || version > 0, "Invalid version in {fname}");
        assert!(file == 0 || dim > 0, "Invalid dimension in {fname}");
        Self { file, dim, version }
    }

    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.file != 0
    }

    #[must_use]
    pub const fn is_invalid(&self) -> bool {
        self.file == 0
    }

    /// Get the dimension (# or components for the coordinates)
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim as usize
    }

    /// Read the vertices and return a vector of size (dim * # of vertices)
    #[must_use]
    pub fn read_vertices(&self) -> Vec<f64> {
        match self.version {
            1 => self.read_vertices_gen::<f32>(),
            2 => self.read_vertices_gen::<f64>(),
            3 => self.read_vertices_gen::<f64>(),
            4 => self.read_vertices_gen::<f64>(),
            _ => panic!("Unsupported meshb version: {}", self.version),
        }
    }

    fn read_vertices_gen<T1: TryInto<f64> + Default + Clone>(&self) -> Vec<f64>
    where
        <T1 as std::convert::TryInto<f64>>::Error: std::fmt::Debug,
    {
        assert!(self.is_valid());

        let n_nodes = unsafe { GmfStatKwd(self.file, GmfKwdCod::GmfVertices as c_int) };
        debug!("Read {} vertices", n_nodes);

        let mut res = Vec::with_capacity(self.dim as usize * n_nodes as usize);

        unsafe { GmfGotoKwd(self.file, GmfKwdCod::GmfVertices as c_int) };

        match self.dim {
            2 => {
                let mut tag: c_int = 0;
                let mut x: T1 = Default::default();
                let mut y: T1 = Default::default();
                for _ in 0..n_nodes {
                    unsafe {
                        GmfGetLin(
                            self.file,
                            GmfKwdCod::GmfVertices as c_int,
                            addr_of_mut!(x),
                            addr_of_mut!(y),
                            addr_of_mut!(tag),
                        )
                    };
                    res.push(x.clone().try_into().unwrap());
                    res.push(y.clone().try_into().unwrap());
                }
            }
            3 => {
                let mut tag: c_int = 0;
                let mut x: T1 = Default::default();
                let mut y: T1 = Default::default();
                let mut z: T1 = Default::default();
                for _ in 0..n_nodes {
                    unsafe {
                        GmfGetLin(
                            self.file,
                            GmfKwdCod::GmfVertices as c_int,
                            addr_of_mut!(x),
                            addr_of_mut!(y),
                            addr_of_mut!(z),
                            addr_of_mut!(tag),
                        )
                    };
                    res.push(x.clone().try_into().unwrap());
                    res.push(y.clone().try_into().unwrap());
                    res.push(z.clone().try_into().unwrap());
                }
            }
            _ => unreachable!(),
        }
        res
    }

    /// Read the element connectivity and element tag of a given type and return
    ///  - a connectivity vector of size (# of vertices per element * # of elements )
    ///  - a tag vector of size (# of elements)
    #[must_use]
    pub fn read_elements(&self, etype: GmfElementTypes) -> (Vec<Idx>, Vec<Tag>) {
        match self.version {
            1 => self.read_elements_gen::<i32>(etype),
            2 => self.read_elements_gen::<i32>(etype),
            3 => self.read_elements_gen::<i32>(etype),
            4 => self.read_elements_gen::<i64>(etype),
            _ => panic!("Unsupported meshb version: {}", self.version),
        }
    }

    fn read_elements_gen<T2: TryInto<Idx> + Default + Clone>(
        &self,
        etype: GmfElementTypes,
    ) -> (Vec<Idx>, Vec<Tag>)
    where
        <T2 as std::convert::TryInto<Idx>>::Error: std::fmt::Debug,
    {
        assert!(self.is_valid());

        let m = match etype {
            GmfElementTypes::Vertex => 1,
            GmfElementTypes::Edge => 2,
            GmfElementTypes::Triangle => 3,
            GmfElementTypes::Tetrahedron => 4,
        };

        let n_elems = unsafe { GmfStatKwd(self.file, etype as c_int) };
        debug!("Read {} elements ({:?})", n_elems, etype);

        let mut elems = Vec::with_capacity(m * n_elems as usize);
        let mut etags = Vec::with_capacity(n_elems as usize);

        unsafe { GmfGotoKwd(self.file, etype as c_int) };

        match etype {
            GmfElementTypes::Vertex => {
                let mut tag: c_int = 0;
                let mut i0: T2 = Default::default();
                for _ in 0..n_elems {
                    unsafe {
                        GmfGetLin(
                            self.file,
                            etype as c_int,
                            addr_of_mut!(i0),
                            addr_of_mut!(tag),
                        )
                    };
                    elems.push(i0.clone().try_into().unwrap() - 1);
                    etags.push(tag as Tag);
                }
            }
            GmfElementTypes::Edge => {
                let mut tag: c_int = 0;
                let mut i0: T2 = Default::default();
                let mut i1: T2 = Default::default();
                for _ in 0..n_elems {
                    unsafe {
                        GmfGetLin(
                            self.file,
                            etype as c_int,
                            addr_of_mut!(i0),
                            addr_of_mut!(i1),
                            addr_of_mut!(tag),
                        )
                    };
                    elems.push(i0.clone().try_into().unwrap() - 1);
                    elems.push(i1.clone().try_into().unwrap() - 1);
                    etags.push(tag as Tag);
                }
            }
            GmfElementTypes::Triangle => {
                let mut tag: c_int = 0;
                let mut i0: T2 = Default::default();
                let mut i1: T2 = Default::default();
                let mut i2: T2 = Default::default();
                for _ in 0..n_elems {
                    unsafe {
                        GmfGetLin(
                            self.file,
                            etype as c_int,
                            addr_of_mut!(i0),
                            addr_of_mut!(i1),
                            addr_of_mut!(i2),
                            addr_of_mut!(tag),
                        )
                    };
                    elems.push(i0.clone().try_into().unwrap() - 1);
                    elems.push(i1.clone().try_into().unwrap() - 1);
                    elems.push(i2.clone().try_into().unwrap() - 1);
                    etags.push(tag as Tag);
                }
            }
            GmfElementTypes::Tetrahedron => {
                let mut tag: c_int = 0;
                let mut i0: T2 = Default::default();
                let mut i1: T2 = Default::default();
                let mut i2: T2 = Default::default();
                let mut i3: T2 = Default::default();
                for _ in 0..n_elems {
                    unsafe {
                        GmfGetLin(
                            self.file,
                            etype as c_int,
                            addr_of_mut!(i0),
                            addr_of_mut!(i1),
                            addr_of_mut!(i2),
                            addr_of_mut!(i3),
                            addr_of_mut!(tag),
                        )
                    };
                    elems.push(i0.clone().try_into().unwrap() - 1);
                    elems.push(i1.clone().try_into().unwrap() - 1);
                    elems.push(i2.clone().try_into().unwrap() - 1);
                    elems.push(i3.clone().try_into().unwrap() - 1);
                    etags.push(tag as Tag);
                }
            }
        }
        (elems, etags)
    }

    /// Read the field defined at the vertices (for .sol(b) files) and return a
    ///  - vector of size (m * # of vertices)
    ///  - the number of components m
    #[must_use]
    pub fn read_solution(&self) -> (Vec<f64>, usize) {
        match self.version {
            1 => self.read_solution_gen::<f32>(),
            2 => self.read_solution_gen::<f64>(),
            3 => self.read_solution_gen::<f64>(),
            4 => self.read_solution_gen::<f64>(),
            _ => panic!("Unsupported meshb version: {}", self.version),
        }
    }

    fn read_solution_gen<T1: TryInto<f64> + Default + Clone + Copy>(&self) -> (Vec<f64>, usize)
    where
        <T1 as std::convert::TryInto<f64>>::Error: std::fmt::Debug,
    {
        let mut field_type: c_int = 0;
        let mut n_types: c_int = 0;
        let mut sol_size: c_int = 0;
        let n_verts = unsafe {
            GmfStatKwd(
                self.file,
                GmfKwdCod::GmfSolAtVertices as c_int,
                addr_of_mut!(n_types),
                addr_of_mut!(sol_size),
                addr_of_mut!(field_type),
            )
        };
        assert_eq!(n_types, 1);

        let (field_type, n_comp) = match field_type as u32 {
            x if x == GmfSca => (GmfFieldTypes::Scalar, 1),
            x if x == GmfVec => (GmfFieldTypes::Vector, self.dim),
            x if x == GmfSymMat => (GmfFieldTypes::Metric, (self.dim * (self.dim + 1)) / 2),
            _ => unreachable!("Field type {field_type} unknown: {GmfSca} {GmfVec} {GmfSymMat}"),
        };
        assert_eq!(sol_size, n_comp as c_int);

        debug!("Read {}x{} values", n_verts, n_comp);

        let mut res = Vec::with_capacity(n_comp as usize * n_verts as usize);
        let mut s: [T1; 6] = [Default::default(); 6];

        unsafe { GmfGotoKwd(self.file, GmfKwdCod::GmfSolAtVertices as c_int) };

        let order = field_order(self.dim as usize, field_type);

        for _ in 0..n_verts {
            unsafe {
                GmfGetLin(
                    self.file,
                    GmfKwdCod::GmfSolAtVertices as c_int,
                    addr_of_mut!(s),
                );
            }
            for j in order.iter().copied() {
                res.push(s[j].try_into().unwrap());
            }
        }

        (res, n_comp as usize)
    }
}

impl Drop for GmfReader {
    fn drop(&mut self) {
        if self.is_valid() {
            unsafe {
                GmfCloseMesh(self.file);
            }
        }
    }
}

/// Writer for .mesh(b) / .sol(b) files (interface to libMeshb)
/// file version 2 (int32 and float64) is used
pub struct GmfWriter {
    file: i64,
}

impl GmfWriter {
    /// Create a new file
    #[must_use]
    pub fn new(fname: &str, dim: usize) -> Self {
        debug!("Open {} (write)", fname);
        let dim = dim as c_int;
        let version = 2;

        let fname = CString::new(fname).unwrap();
        Self {
            file: unsafe { GmfOpenMesh(fname.as_ptr(), GmfWrite as c_int, version, dim) },
        }
    }

    /// Write a SimplexMesh
    pub fn write_mesh<const D: usize, E: Elem>(&mut self, mesh: &SimplexMesh<D, E>) {
        self.write_vertices::<D, _>(mesh.n_verts(), mesh.verts());
        self.write_elements::<E, _, _>(mesh.n_elems(), mesh.elems(), mesh.etags());
        self.write_elements::<E::Face, _, _>(mesh.n_faces(), mesh.faces(), mesh.ftags());
    }

    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.file != 0
    }

    #[must_use]
    pub const fn is_invalid(&self) -> bool {
        self.file == 0
    }

    fn write_vertices<const D: usize, I: Iterator<Item = Point<D>>>(&self, n_verts: Idx, verts: I) {
        debug!("Write {} vertices", n_verts);

        unsafe {
            GmfSetKwd(
                self.file,
                GmfKwdCod::GmfVertices as c_int,
                i64::from(n_verts),
            );
        }

        for p in verts {
            if D == 2 {
                unsafe {
                    GmfSetLin(self.file, GmfKwdCod::GmfVertices as c_int, p[0], p[1], 1);
                }
            } else if D == 3 {
                unsafe {
                    GmfSetLin(
                        self.file,
                        GmfKwdCod::GmfVertices as c_int,
                        p[0],
                        p[1],
                        p[2],
                        1,
                    );
                }
            } else {
                unreachable!();
            }
        }
    }

    fn write_elements<E: Elem, I1: Iterator<Item = E>, I2: Iterator<Item = Tag>>(
        &self,
        n_elems: Idx,
        elems: I1,
        etags: I2,
    ) {
        let etype = match E::N_VERTS {
            1 => GmfElementTypes::Vertex,
            2 => GmfElementTypes::Edge,
            3 => GmfElementTypes::Triangle,
            4 => GmfElementTypes::Tetrahedron,
            _ => unreachable!(),
        };
        debug!("Write {} elements ({:?})", n_elems, etype);

        unsafe {
            GmfSetKwd(self.file, etype as c_int, i64::from(n_elems));
        }

        for (e, t) in elems.zip(etags) {
            match etype {
                GmfElementTypes::Vertex => unsafe {
                    GmfSetLin(self.file, etype as c_int, e[0] + 1, i32::from(t));
                },

                GmfElementTypes::Edge => unsafe {
                    GmfSetLin(self.file, etype as c_int, e[0] + 1, e[1] + 1, i32::from(t));
                },
                GmfElementTypes::Triangle => unsafe {
                    GmfSetLin(
                        self.file,
                        etype as c_int,
                        e[0] + 1,
                        e[1] + 1,
                        e[2] + 1,
                        i32::from(t),
                    );
                },
                GmfElementTypes::Tetrahedron => unsafe {
                    GmfSetLin(
                        self.file,
                        etype as c_int,
                        e[0] + 1,
                        e[1] + 1,
                        e[2] + 1,
                        e[3] + 1,
                        i32::from(t),
                    );
                },
            };
        }
    }

    /// Write a solution defined at the vertices (.sol(b) file)
    pub fn write_solution(&mut self, arr: &[f64], dim: usize, n_comp: usize) {
        let n_verts = arr.len() / n_comp;
        debug!("Write {}x{} values", n_verts, n_comp);

        let field_type = match n_comp {
            1 => GmfFieldTypes::Scalar,
            x if x == dim => GmfFieldTypes::Vector,
            x if x == (dim * (dim + 1)) / 2 => GmfFieldTypes::Metric,
            _ => unreachable!(),
        };

        unsafe {
            let val = match field_type {
                GmfFieldTypes::Scalar => GmfSca,
                GmfFieldTypes::Vector => GmfVec,
                GmfFieldTypes::Metric => GmfSymMat,
            } as c_int;

            GmfSetKwd(
                self.file,
                GmfKwdCod::GmfSolAtVertices as c_int,
                n_verts as i64,
                1,
                &val,
            );
        }

        let order = field_order(dim, field_type);

        let mut vals = [0.0; 6];
        for i in 0..n_verts {
            let start = i * n_comp;
            let end = start + n_comp;
            let s = &arr[start..end];
            for (i, j) in order.iter().copied().enumerate() {
                vals[j] = s[i];
            }

            unsafe {
                GmfSetLin(self.file, GmfKwdCod::GmfSolAtVertices as c_int, &vals);
            }
        }
    }

    pub fn close(&mut self) {
        if self.is_valid() {
            unsafe {
                GmfCloseMesh(self.file);
            }
        }
    }
}

impl Drop for GmfWriter {
    fn drop(&mut self) {
        self.close();
    }
}

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    pub fn write_meshb(&self, file_name: &str) -> Result<()> {
        let mut writer = GmfWriter::new(file_name, D);

        if writer.is_invalid() {
            return Err(Error::from("Cannot open the result file"));
        }

        writer.write_mesh(self);

        Ok(())
    }

    pub fn write_solb(&self, arr: &[f64], file_name: &str) -> Result<()> {
        let mut writer = GmfWriter::new(file_name, D);
        if writer.is_invalid() {
            return Err(Error::from(&format!("Cannot open {file_name}")));
        }

        let n_comp = arr.len() / self.n_verts() as usize;
        writer.write_solution(arr, D, n_comp);

        Ok(())
    }

    pub fn read_meshb(file_name: &str) -> Result<Self> {
        let reader = GmfReader::new(file_name);
        if reader.is_invalid() {
            return Err(Error::from(&format!("Cannot open {file_name}")));
        }
        if reader.dim() != D {
            return Err(Error::from("Invalid dimension"));
        }

        let coords = reader.read_vertices();
        let (etype, ftype) = match E::N_VERTS {
            4 => (GmfElementTypes::Tetrahedron, GmfElementTypes::Triangle),
            3 => (GmfElementTypes::Triangle, GmfElementTypes::Edge),
            2 => (GmfElementTypes::Edge, GmfElementTypes::Vertex),
            _ => unreachable!(),
        };
        let (elems, etags) = reader.read_elements(etype);
        let (faces, ftags) = reader.read_elements(ftype);

        let n_verts = coords.len() / D;
        let verts = (0..n_verts)
            .map(|i| {
                let start = i * D;
                let end = start + D;
                Point::<D>::from_iterator(coords[start..end].iter().copied())
            })
            .collect();

        let n_elems = elems.len() / E::N_VERTS as usize;
        let elems = (0..n_elems)
            .map(|i| {
                let start = i * E::N_VERTS as usize;
                let end = start + E::N_VERTS as usize;
                E::from_slice(&elems[start..end])
            })
            .collect();

        let n_faces = faces.len() / E::Face::N_VERTS as usize;
        let faces = (0..n_faces)
            .map(|i| {
                let start = i * E::Face::N_VERTS as usize;
                let end = start + E::Face::N_VERTS as usize;
                E::Face::from_slice(&faces[start..end])
            })
            .collect();

        Ok(Self::new(verts, elems, etags, faces, ftags))
    }

    pub fn read_solb(file_name: &str) -> Result<(Vec<f64>, usize)> {
        let reader = GmfReader::new(file_name);
        if reader.is_invalid() {
            return Err(Error::from(&format!("Cannot open {file_name}")));
        }

        let (sol, m) = reader.read_solution();

        Ok((sol, m))
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
