extern crate libmeshb_sys;
use log::{debug, info};

use self::libmeshb_sys::{
    GmfCloseMesh, GmfGetLin, GmfGotoKwd, GmfKwdCod, GmfOpenMesh, GmfRead, GmfSca, GmfSetKwd,
    GmfSetLin, GmfStatKwd, GmfSymMat, GmfVec, GmfWrite,
};
use crate::{
    mesh::{Point, SimplexMesh},
    topo_elems::Elem,
    Error, Idx, Result, Tag,
};
use std::{
    convert::TryInto,
    ffi::{c_int, CString},
};

/// Reader for .mesh(b) / .sol(b) files (interface to libMeshb)
pub struct GmfReader {
    file: i64,
    dim: c_int,
    version: c_int,
}

#[derive(Clone, Copy, Debug)]
pub enum GmfElementTypes {
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
        info!("Open {} (read)", fname);
        let dim = 0;
        let version = 0;

        let fname = CString::new(fname).unwrap();
        Self {
            file: unsafe { GmfOpenMesh(fname.as_ptr(), GmfRead as c_int, &version, &dim) },
            dim,
            version,
        }
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
            _ => unreachable!(),
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
                let tag: c_int = 0;
                let x: T1 = Default::default();
                let y: T1 = Default::default();
                for _ in 0..n_nodes {
                    unsafe { GmfGetLin(self.file, GmfKwdCod::GmfVertices as c_int, &x, &y, &tag) };
                    res.push(x.clone().try_into().unwrap());
                    res.push(y.clone().try_into().unwrap());
                }
            }
            3 => {
                let tag: c_int = 0;
                let x: T1 = Default::default();
                let y: T1 = Default::default();
                let z: T1 = Default::default();
                for _ in 0..n_nodes {
                    unsafe {
                        GmfGetLin(self.file, GmfKwdCod::GmfVertices as c_int, &x, &y, &z, &tag)
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
    pub fn read_elements(&self, etype: GmfElementTypes) -> (Vec<Idx>, Vec<Tag>) {
        match self.version {
            1 => self.read_elements_gen::<i32>(etype),
            2 => self.read_elements_gen::<i32>(etype),
            3 => self.read_elements_gen::<i32>(etype),
            4 => self.read_elements_gen::<i64>(etype),
            _ => unreachable!(),
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
            GmfElementTypes::Edge => {
                let tag: c_int = 0;
                let i0: T2 = Default::default();
                let i1: T2 = Default::default();
                for _ in 0..n_elems {
                    unsafe { GmfGetLin(self.file, etype as c_int, &i0, &i1, &tag) };
                    elems.push(i0.clone().try_into().unwrap() - 1);
                    elems.push(i1.clone().try_into().unwrap() - 1);
                    etags.push(tag as Tag);
                }
            }
            GmfElementTypes::Triangle => {
                let tag: c_int = 0;
                let i0: T2 = Default::default();
                let i1: T2 = Default::default();
                let i2: T2 = Default::default();
                for _ in 0..n_elems {
                    unsafe { GmfGetLin(self.file, etype as c_int, &i0, &i1, &i2, &tag) };
                    elems.push(i0.clone().try_into().unwrap() - 1);
                    elems.push(i1.clone().try_into().unwrap() - 1);
                    elems.push(i2.clone().try_into().unwrap() - 1);
                    etags.push(tag as Tag);
                }
            }
            GmfElementTypes::Tetrahedron => {
                let tag: c_int = 0;
                let i0: T2 = Default::default();
                let i1: T2 = Default::default();
                let i2: T2 = Default::default();
                let i3: T2 = Default::default();
                for _ in 0..n_elems {
                    unsafe { GmfGetLin(self.file, etype as c_int, &i0, &i1, &i2, &i3, &tag) };
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
    pub fn read_solution(&self) -> (Vec<f64>, usize) {
        match self.version {
            1 => self.read_solution_gen::<f32>(),
            2 => self.read_solution_gen::<f64>(),
            3 => self.read_solution_gen::<f64>(),
            4 => self.read_solution_gen::<f64>(),
            _ => unreachable!(),
        }
    }

    fn read_solution_gen<T1: TryInto<f64> + Default + Clone + Copy>(&self) -> (Vec<f64>, usize)
    where
        <T1 as std::convert::TryInto<f64>>::Error: std::fmt::Debug,
    {
        let field_type: c_int = 0;
        let n_types: c_int = 0;
        let sol_size: c_int = 0;
        let n_verts = unsafe {
            GmfStatKwd(
                self.file,
                GmfKwdCod::GmfSolAtVertices as c_int,
                &n_types,
                &sol_size,
                &field_type,
            )
        };
        let (field_type, n_comp) = match field_type as u32 {
            x if x == GmfSca => (GmfFieldTypes::Scalar, 1),
            x if x == GmfVec => (GmfFieldTypes::Vector, self.dim),
            x if x == GmfSymMat => (GmfFieldTypes::Metric, (self.dim * (self.dim + 1)) / 2),
            _ => unreachable!(),
        };
        debug!("Read {}x{} values", n_verts, n_comp);

        let mut res = Vec::with_capacity(n_comp as usize * n_verts as usize);
        let s: [T1; 6] = [Default::default(); 6];

        unsafe { GmfGotoKwd(self.file, GmfKwdCod::GmfSolAtVertices as c_int) };

        let order = field_order(self.dim as usize, field_type);

        for _ in 0..n_verts {
            unsafe {
                GmfGetLin(self.file, GmfKwdCod::GmfSolAtVertices as c_int, &s);
            }
            for j in order.iter().cloned() {
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
    pub fn new(fname: &str, dim: usize) -> Self {
        info!("Open {} (write)", fname);
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

    pub fn is_valid(&self) -> bool {
        self.file != 0
    }

    pub fn is_invalid(&self) -> bool {
        self.file == 0
    }

    fn write_vertices<const D: usize, I: Iterator<Item = Point<D>>>(&self, n_verts: Idx, verts: I) {
        debug!("Write {} vertices", n_verts);

        unsafe {
            GmfSetKwd(self.file, GmfKwdCod::GmfVertices as c_int, n_verts as i64);
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
            2 => GmfElementTypes::Edge,
            3 => GmfElementTypes::Triangle,
            4 => GmfElementTypes::Tetrahedron,
            _ => unreachable!(),
        };
        debug!("Write {} elements ({:?})", n_elems, etype);

        unsafe {
            GmfSetKwd(self.file, etype as c_int, n_elems as i64);
        }

        for (e, t) in elems.zip(etags) {
            match etype {
                GmfElementTypes::Edge => unsafe {
                    GmfSetLin(self.file, etype as c_int, e[0] + 1, e[1] + 1, t as c_int);
                },
                GmfElementTypes::Triangle => unsafe {
                    GmfSetLin(
                        self.file,
                        etype as c_int,
                        e[0] + 1,
                        e[1] + 1,
                        e[2] + 1,
                        t as c_int,
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
                        t as c_int,
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
            let val = field_type as c_int;
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
            for (i, j) in order.iter().cloned().enumerate() {
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
            return Err(Error::from(&format!("Cannot open {}", file_name)));
        }

        let n_comp = arr.len() / self.n_verts() as usize;
        writer.write_solution(arr, D, n_comp);

        Ok(())
    }

    pub fn read_meshb(file_name: &str) -> Result<Self> {
        let reader = GmfReader::new(file_name);
        if reader.is_invalid() {
            return Err(Error::from(&format!("Cannot open {}", file_name)));
        }
        if reader.dim() != D {
            return Err(Error::from("Invalid dimension"));
        }

        let coords = reader.read_vertices();
        let (etype, ftype) = match E::N_VERTS {
            4 => (GmfElementTypes::Tetrahedron, GmfElementTypes::Triangle),
            3 => (GmfElementTypes::Triangle, GmfElementTypes::Edge),
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

        Ok(SimplexMesh::<D, E>::new(verts, elems, etags, faces, ftags))
    }
}

pub fn read_solb(file_name: &str) -> Result<(Vec<f64>, usize)> {
    let reader = GmfReader::new(file_name);
    if reader.is_invalid() {
        return Err(Error::from(&format!("Cannot open {}", file_name)));
    }

    let (sol, m) = reader.read_solution();

    Ok((sol, m))
}
