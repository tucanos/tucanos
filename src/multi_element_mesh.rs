use std::collections::HashMap;

use crate::{
    mesh::SimplexMesh,
    topo_elems::{Elem, Tetrahedron, Triangle},
    Error, FieldLocation, Idx, Mesh, Result, Tag,
};

/// Subdivision of standard elements to triangles and tetrahedra maintaining a consistent mesh. The algorithms are taken from
/// How to Subdivide Pyramids, Prisms and Hexahedra into Tetrahedra
/// Julien Dompierre Paul Labbé Marie-Gabrielle Vallet Ricardo Camarero

const INDIRECTION_PRI: [[usize; 6]; 6] = [
    [0, 1, 2, 3, 4, 5],
    [1, 2, 0, 4, 5, 3],
    [2, 0, 1, 5, 3, 4],
    [3, 5, 4, 0, 2, 1],
    [4, 3, 5, 1, 0, 2],
    [5, 4, 3, 2, 1, 0],
];

const INDIRECTION_HEX: [[usize; 8]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 0, 4, 5, 2, 3, 7, 6],
    [2, 1, 5, 6, 3, 0, 4, 7],
    [3, 0, 1, 2, 7, 4, 5, 6],
    [4, 0, 3, 7, 5, 1, 2, 6],
    [5, 1, 0, 4, 6, 2, 3, 7],
    [6, 2, 1, 5, 7, 3, 0, 4],
    [7, 3, 2, 6, 4, 0, 1, 5],
];

const ROTATION_HEX: [usize; 8] = [0, 0, 240, 120, 120, 240, 0, 0];
const PERM_120: [usize; 8] = [0, 4, 5, 1, 3, 7, 6, 2];
const PERM_240: [usize; 8] = [0, 3, 7, 4, 1, 2, 6, 5];

fn compare_edges(i0: Idx, i1: Idx, i2: Idx, i3: Idx) -> bool {
    Idx::min(i0, i1) < Idx::min(i2, i3)
}

fn argmin(arr: &[Idx]) -> usize {
    let mut imin = 0;
    let mut idx = arr[0];

    for (i, j) in arr.iter().skip(1).copied().enumerate() {
        if j < idx {
            idx = j;
            imin = i;
        }
    }

    imin
}

/// Convert a quadrangle into 2 triangles
fn quad2tris(quad: &[Idx]) -> [Triangle; 2] {
    let mut tri1 = Triangle::default();
    let mut tri2 = Triangle::default();

    if compare_edges(quad[0], quad[2], quad[1], quad[3]) {
        tri1[0] = quad[0];
        tri1[1] = quad[1];
        tri1[2] = quad[2];
        tri2[0] = quad[0];
        tri2[1] = quad[2];
        tri2[2] = quad[3];
    } else {
        tri1[0] = quad[1];
        tri1[1] = quad[2];
        tri1[2] = quad[3];
        tri2[0] = quad[1];
        tri2[1] = quad[3];
        tri2[2] = quad[0];
    }

    [tri1, tri2]
}

/// Convert a pyramid into 2 tetrahedra
fn pyr2tets(pyr: &[Idx]) -> [Tetrahedron; 2] {
    let mut tet1 = Tetrahedron::default();
    let mut tet2 = Tetrahedron::default();

    if compare_edges(pyr[0], pyr[2], pyr[1], pyr[3]) {
        tet1[0] = pyr[0];
        tet1[1] = pyr[1];
        tet1[2] = pyr[2];
        tet1[3] = pyr[4];
        tet2[0] = pyr[0];
        tet2[1] = pyr[2];
        tet2[2] = pyr[3];
        tet2[3] = pyr[4];
    } else {
        tet1[0] = pyr[1];
        tet1[1] = pyr[2];
        tet1[2] = pyr[3];
        tet1[3] = pyr[4];
        tet2[0] = pyr[1];
        tet2[1] = pyr[3];
        tet2[2] = pyr[0];
        tet2[3] = pyr[4];
    }

    [tet1, tet2]
}

/// Convert a prism into 3 tetrahedra
fn pri2tets(pri: &[Idx]) -> [Tetrahedron; 3] {
    let imin = argmin(pri);

    let mut idx = [0; 6];
    for i in 0..6 {
        idx[i] = pri[INDIRECTION_PRI[imin][i]];
    }

    let mut tet1 = Tetrahedron::default();
    let mut tet2 = Tetrahedron::default();
    let mut tet3 = Tetrahedron::default();
    if compare_edges(idx[1], idx[5], idx[2], idx[4]) {
        tet1[0] = idx[0];
        tet1[1] = idx[1];
        tet1[2] = idx[2];
        tet1[3] = idx[5];
        tet2[0] = idx[0];
        tet2[1] = idx[1];
        tet2[2] = idx[5];
        tet2[3] = idx[4];
        tet3[0] = idx[0];
        tet3[1] = idx[4];
        tet3[2] = idx[5];
        tet3[3] = idx[3];
    } else {
        tet1[0] = idx[0];
        tet1[1] = idx[1];
        tet1[2] = idx[2];
        tet1[3] = idx[4];
        tet2[0] = idx[0];
        tet2[1] = idx[4];
        tet2[2] = idx[2];
        tet2[3] = idx[5];
        tet3[0] = idx[0];
        tet3[1] = idx[4];
        tet3[2] = idx[5];
        tet3[3] = idx[3];
    }

    [tet1, tet2, tet3]
}

/// Convert a hex into 5 or 6 tetrahedra
#[allow(clippy::too_many_lines)]
fn hex2tets(hex: &[Idx]) -> ([Tetrahedron; 5], Option<Tetrahedron>) {
    let imin = argmin(hex);

    let mut idx = [0; 8];
    for i in 0..8 {
        idx[i] = hex[INDIRECTION_HEX[imin][i]];
    }

    let i1 = u32::from(compare_edges(idx[1], idx[6], idx[2], idx[5]));
    let i2 = u32::from(compare_edges(idx[3], idx[6], idx[2], idx[7]));
    let i3 = u32::from(compare_edges(idx[4], idx[6], idx[5], idx[7]));
    let flg = i1 + 2 * i2 + 4 * i3;
    let rot = ROTATION_HEX[flg as usize];

    let mut idx2 = idx;
    if rot == 120 {
        for i in 0..8 {
            idx2[i] = idx[PERM_120[i]];
        }
    } else if rot == 240 {
        for i in 0..8 {
            idx2[i] = idx[PERM_240[i]];
        }
    }
    let flg2 = i1 + i2 + i3;

    let mut tet1 = Tetrahedron::default();
    let mut tet2 = Tetrahedron::default();
    let mut tet3 = Tetrahedron::default();
    let mut tet4 = Tetrahedron::default();
    let mut tet5 = Tetrahedron::default();
    let mut tet6 = Tetrahedron::default();

    if flg2 == 0 {
        tet1[0] = idx2[0];
        tet1[1] = idx2[1];
        tet1[2] = idx2[2];
        tet1[3] = idx2[5];
        tet2[0] = idx2[0];
        tet2[1] = idx2[2];
        tet2[2] = idx2[7];
        tet2[3] = idx2[5];
        tet3[0] = idx2[0];
        tet3[1] = idx2[2];
        tet3[2] = idx2[3];
        tet3[3] = idx2[7];
        tet4[0] = idx2[0];
        tet4[1] = idx2[5];
        tet4[2] = idx2[7];
        tet4[3] = idx2[4];
        tet5[0] = idx2[2];
        tet5[1] = idx2[7];
        tet5[2] = idx2[5];
        tet5[3] = idx2[6];
        return ([tet1, tet2, tet3, tet4, tet5], None);
    } else if flg2 == 1 {
        tet1[0] = idx2[0];
        tet1[1] = idx2[5];
        tet1[2] = idx2[7];
        tet1[3] = idx2[4];
        tet2[0] = idx2[0];
        tet2[1] = idx2[1];
        tet2[2] = idx2[7];
        tet2[3] = idx2[5];
        tet3[0] = idx2[1];
        tet3[1] = idx2[6];
        tet3[2] = idx2[7];
        tet3[3] = idx2[5];
        tet4[0] = idx2[0];
        tet4[1] = idx2[7];
        tet4[2] = idx2[2];
        tet4[3] = idx2[3];
        tet5[0] = idx2[0];
        tet5[1] = idx2[7];
        tet5[2] = idx2[1];
        tet5[3] = idx2[2];
        tet6[0] = idx2[1];
        tet6[1] = idx2[7];
        tet6[2] = idx2[6];
        tet6[3] = idx2[2];
    } else if flg2 == 2 {
        tet1[0] = idx2[0];
        tet1[1] = idx2[4];
        tet1[2] = idx2[5];
        tet1[3] = idx2[6];
        tet2[0] = idx2[0];
        tet2[1] = idx2[3];
        tet2[2] = idx2[7];
        tet2[3] = idx2[6];
        tet3[0] = idx2[0];
        tet3[1] = idx2[7];
        tet3[2] = idx2[4];
        tet3[3] = idx2[6];
        tet4[0] = idx2[0];
        tet4[1] = idx2[1];
        tet4[2] = idx2[2];
        tet4[3] = idx2[5];
        tet5[0] = idx2[0];
        tet5[1] = idx2[3];
        tet5[2] = idx2[6];
        tet5[3] = idx2[2];
        tet6[0] = idx2[0];
        tet6[1] = idx2[6];
        tet6[2] = idx2[5];
        tet6[3] = idx2[2];
    } else if flg2 == 3 {
        tet1[0] = idx2[0];
        tet1[1] = idx2[2];
        tet1[2] = idx2[3];
        tet1[3] = idx2[6];
        tet2[0] = idx2[0];
        tet2[1] = idx2[3];
        tet2[2] = idx2[7];
        tet2[3] = idx2[6];
        tet3[0] = idx2[0];
        tet3[1] = idx2[7];
        tet3[2] = idx2[4];
        tet3[3] = idx2[6];
        tet4[0] = idx2[0];
        tet4[1] = idx2[5];
        tet4[2] = idx2[6];
        tet4[3] = idx2[4];
        tet5[0] = idx2[1];
        tet5[1] = idx2[5];
        tet5[2] = idx2[6];
        tet5[3] = idx2[0];
        tet6[0] = idx2[1];
        tet6[1] = idx2[6];
        tet6[2] = idx2[2];
        tet6[3] = idx2[0];
    }
    ([tet1, tet2, tet3, tet4, tet5], Some(tet6))
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ElementType {
    Edge,
    Triangle,
    Quadrangle,
    Tetrahedron,
    Pyramids,
    Prism,
    Hexahedron,
}

impl ElementType {
    fn n_verts(self) -> Idx {
        match self {
            ElementType::Edge => 2,
            ElementType::Triangle => 3,
            ElementType::Quadrangle => 4,
            ElementType::Tetrahedron => 4,
            ElementType::Pyramids => 5,
            ElementType::Prism => 6,
            ElementType::Hexahedron => 8,
        }
    }

    fn dim(self) -> Idx {
        match self {
            ElementType::Edge => 1,
            ElementType::Triangle => 2,
            ElementType::Quadrangle => 2,
            ElementType::Tetrahedron => 3,
            ElementType::Pyramids => 3,
            ElementType::Prism => 3,
            ElementType::Hexahedron => 3,
        }
    }
}

/// Mesh containing elements of any type
#[derive(Debug)]
pub struct MultiElementMesh {
    /// Mesh dimension
    pub dim: u32,
    /// Coordinates of the vertices (length = dim * # of vertices)
    pub coords: Vec<f64>,
    /// Pointer to the vertices of the i-th element (CSR storage) (length = # of elements + 1)
    pub ptr: Vec<Idx>,
    /// Element types (length = # of elements)
    pub etypes: Vec<ElementType>,
    /// Element tags (length = # of elements)
    pub etags: Vec<Tag>,
    /// Element to vertex connectivity
    pub conn: Vec<Idx>,
    /// Fields defined on the mesh
    pub data: HashMap<(String, FieldLocation), Vec<f64>>,
}

impl Mesh for MultiElementMesh {
    fn dim(&self) -> Idx {
        self.dim as Idx
    }

    fn cell_dim(&self) -> Idx {
        self.etypes
            .iter()
            .copied()
            .map(ElementType::dim)
            .max()
            .unwrap()
    }

    fn n_verts(&self) -> Idx {
        self.coords.len() as Idx / self.dim as Idx
    }

    fn n_elems(&self) -> Idx {
        let cdim = self.cell_dim();
        self.etypes.iter().filter(|x| x.dim() == cdim).count() as Idx
    }

    fn n_faces(&self) -> Idx {
        let fdim = self.cell_dim() - 1;
        self.etypes.iter().filter(|x| x.dim() == fdim).count() as Idx
    }
}

impl MultiElementMesh {
    /// Create a new empty `MultiElementMesh` of dimension dim
    #[must_use]
    pub fn new(dim: u32) -> Self {
        Self {
            dim,
            coords: Vec::default(),
            ptr: vec![0],
            etypes: Vec::default(),
            etags: Vec::default(),
            conn: Vec::default(),
            data: HashMap::default(),
        }
    }

    /// Set the mesh coordinates    
    pub fn set_verts(&mut self, coords: &[f64]) {
        self.coords.reserve(coords.len());
        self.coords.extend(coords.iter());
    }

    /// Add elements of type etype
    /// conn: connectivity array, size # of new elements * # or vertex per element etype
    /// tags: tag array, size # of new elements
    pub fn add_elems(&mut self, etype: ElementType, conn: &[Idx], tags: &[Tag]) {
        let m = etype.n_verts() as usize;
        let n = conn.len() / m;

        self.ptr.reserve(self.ptr.len() + n);
        self.etypes.reserve(self.conn.len() + conn.len());

        let start = self.ptr[self.ptr.len() - 1];
        for i in 0..n {
            self.ptr.push(start + (m * (i + 1)) as Idx);
            self.etypes.push(etype);
        }

        self.etags.extend_from_slice(tags);
        self.conn.extend_from_slice(conn);
    }

    /// Count the elements of a given type
    #[must_use]
    pub fn n_elems_of_type(&self, etype: ElementType) -> Idx {
        self.etypes.iter().filter(|x| **x == etype).count() as Idx
    }

    fn get_elem(&self, i: usize) -> (Tag, &[Idx]) {
        let tag = self.etags[i];
        let start = self.ptr[i] as usize;
        let end = self.ptr[i + 1] as usize;
        let el = &self.conn[start..end];
        (tag, el)
    }

    fn add_edg(&self, i: usize, elems: &mut Vec<Idx>, etags: &mut Vec<Tag>, cell_dim: Idx) -> Idx {
        assert_eq!(self.etypes[i], ElementType::Edge);
        let (tag, el) = self.get_elem(i);
        elems.extend(el.iter());
        etags.push(tag);
        if cell_dim == 1 {
            return 1;
        }
        0
    }

    fn add_tri(&self, i: usize, elems: &mut Vec<Idx>, etags: &mut Vec<Tag>, cell_dim: Idx) -> Idx {
        assert_eq!(self.etypes[i], ElementType::Triangle);
        let (tag, el) = self.get_elem(i);
        elems.extend(el.iter());
        etags.push(tag);
        if cell_dim == 2 {
            return 1;
        }
        0
    }

    fn add_quad(&self, i: usize, elems: &mut Vec<Idx>, etags: &mut Vec<Tag>, cell_dim: Idx) -> Idx {
        assert_eq!(self.etypes[i], ElementType::Quadrangle);
        let (tag, el) = self.get_elem(i);
        for t in quad2tris(el) {
            elems.extend(t.iter());
            etags.push(tag);
        }
        if cell_dim == 2 {
            return 2;
        }
        0
    }

    fn add_tet(&self, i: usize, elems: &mut Vec<Idx>, etags: &mut Vec<Tag>, cell_dim: Idx) -> Idx {
        assert_eq!(self.etypes[i], ElementType::Tetrahedron);
        let (tag, el) = self.get_elem(i);
        elems.extend(el.iter());
        etags.push(tag);
        if cell_dim == 3 {
            return 1;
        }
        0
    }

    fn add_pyr(&self, i: usize, elems: &mut Vec<Idx>, etags: &mut Vec<Tag>, cell_dim: Idx) -> Idx {
        assert_eq!(self.etypes[i], ElementType::Pyramids);
        let (tag, el) = self.get_elem(i);
        for t in pyr2tets(el) {
            elems.extend(t.iter());
            etags.push(tag);
        }
        if cell_dim == 3 {
            return 2;
        }
        0
    }

    fn add_pri(&self, i: usize, elems: &mut Vec<Idx>, etags: &mut Vec<Tag>, cell_dim: Idx) -> Idx {
        assert_eq!(self.etypes[i], ElementType::Prism);
        let (tag, el) = self.get_elem(i);
        for t in pri2tets(el) {
            elems.extend(t.iter());
            etags.push(tag);
        }
        if cell_dim == 3 {
            return 3;
        }
        0
    }

    fn add_hex(&self, i: usize, elems: &mut Vec<Idx>, etags: &mut Vec<Tag>, cell_dim: Idx) -> Idx {
        assert_eq!(self.etypes[i], ElementType::Hexahedron);
        let (tag, el) = self.get_elem(i);
        let (first_5, last) = hex2tets(el);
        for t in first_5 {
            elems.extend(t.iter());
            etags.push(tag);
        }
        if let Some(last) = last {
            elems.extend(last.iter());
            etags.push(tag);
            if cell_dim == 3 {
                return 6;
            }
        }
        if cell_dim == 3 {
            return 5;
        }
        0
    }

    pub fn add_field(&mut self, name: &str, loc: FieldLocation, data: Vec<f64>) -> Result<()> {
        if self.field_type(&data, loc).is_none() {
            return Err(Error::from("Invalid dimension"));
        }
        if self.get_field(name, loc).is_some() {
            return Err(Error::from("Field already present"));
        }
        self.data.insert((name.to_owned(), loc), data);
        Ok(())
    }

    pub fn get_field(&self, name: &str, loc: FieldLocation) -> Option<&Vec<f64>> {
        self.data.get(&(name.to_owned(), loc))
    }

    /// Convert all the elements to simplices an return a `SimplexMesh`
    /// The data is transfered to the new `SimplexMesh` (no interpolation or smoothing for P0 fields)
    #[allow(clippy::type_complexity)]
    pub fn to_simplices<const D: usize, E: Elem>(
        mut self,
    ) -> Result<(
        SimplexMesh<D, E>,
        HashMap<(String, FieldLocation), Vec<f64>>,
    )> {
        if self.dim != D as u32 {
            return Err("Invalid vertex dimension".into());
        }

        let cell_dim = self.cell_dim();
        if cell_dim > E::DIM {
            return Err("Invalid cell dimension".into());
        }
        let same_cell_dim = cell_dim == E::DIM;

        // Elements
        let mut edgs = Vec::new();
        let mut edg_tags = Vec::new();
        let mut tris = Vec::new();
        let mut tri_tags = Vec::new();
        let mut tets = Vec::new();
        let mut tet_tags = Vec::new();

        // Fields
        let mut data: HashMap<(String, FieldLocation), Vec<f64>> = HashMap::new();

        let node_data_names: Vec<_> = self
            .data
            .keys()
            .filter(|(_, loc)| *loc == FieldLocation::Vertex)
            .map(|(name, _)| name.clone())
            .collect();
        let cell_data_names: Vec<_> = self
            .data
            .keys()
            .filter(|(_, loc)| *loc == FieldLocation::Element)
            .map(|(name, _)| name.clone())
            .collect();

        // Get the node data
        for name in &node_data_names {
            let key = (name.clone(), FieldLocation::Vertex);
            let vec = self.data.remove(&key).unwrap();
            data.insert(key, vec);
        }

        // Initialize empty vectors for element data
        if same_cell_dim {
            for name in &cell_data_names {
                let key = (name.clone(), FieldLocation::Element);
                data.insert(key, Vec::new());
            }
        }

        let n = self.ptr.len() - 1;
        let mut i_elem = 0;
        for i in 0..n {
            let mut n_elems_created = 0;
            n_elems_created += match self.etypes.get(i).unwrap() {
                ElementType::Edge => self.add_edg(i, &mut edgs, &mut edg_tags, E::DIM),
                ElementType::Triangle => self.add_tri(i, &mut tris, &mut tri_tags, E::DIM),
                ElementType::Quadrangle => self.add_quad(i, &mut tris, &mut tri_tags, E::DIM),
                ElementType::Tetrahedron => self.add_tet(i, &mut tets, &mut tet_tags, E::DIM),
                ElementType::Pyramids => self.add_pyr(i, &mut tets, &mut tet_tags, E::DIM),
                ElementType::Prism => self.add_pri(i, &mut tets, &mut tet_tags, E::DIM),
                ElementType::Hexahedron => self.add_hex(i, &mut tets, &mut tet_tags, E::DIM),
            };
            if n_elems_created > 0 {
                for name in &cell_data_names {
                    let key = (name.clone(), FieldLocation::Element);
                    let in_vec = self.data.get(&key).unwrap();
                    let out_vec = data.get_mut(&key).unwrap();
                    for _ in 0..n_elems_created {
                        out_vec.push(in_vec[i_elem]);
                    }
                }
                i_elem += 1;
            }
        }

        if E::DIM == 3 {
            return Ok((
                SimplexMesh::new(self.coords, tets, tet_tags, tris, tri_tags),
                data,
            ));
        } else if E::DIM == 2 {
            return Ok((
                SimplexMesh::new(self.coords, tris, tri_tags, edgs, edg_tags),
                data,
            ));
        } else if E::DIM == 1 {
            return Ok((
                SimplexMesh::new(self.coords, edgs, edg_tags, Vec::new(), Vec::new()),
                data,
            ));
        }
        unreachable!();
    }
}

#[cfg(test)]
mod tests {
    use super::{ElementType, MultiElementMesh};
    use crate::{
        mesh::SimplexMesh,
        topo_elems::{Tetrahedron, Triangle},
        Idx, Mesh, Result,
    };

    #[test]
    fn test_2d() -> Result<()> {
        let coords = vec![0., 0., 1., 0., 1., 1., 0., 1., 1.5, 0.5];
        let quad = vec![0, 1, 2, 3];
        let tri = vec![1, 4, 2];
        let edgs = vec![0, 1, 1, 5, 5, 2, 2, 3, 3, 0];

        let mut msh = MultiElementMesh::new(2);
        msh.set_verts(&coords);
        msh.add_elems(ElementType::Quadrangle, &quad, &[1]);
        msh.add_elems(ElementType::Triangle, &tri, &[1]);
        msh.add_elems(ElementType::Edge, &edgs, &[1; 5]);
        msh.add_field(
            "f1",
            crate::FieldLocation::Vertex,
            vec![0., 1., 1., 0., 1.5],
        )?;
        msh.add_field("f2", crate::FieldLocation::Element, vec![1.0, 2.0])?;

        assert_eq!(msh.n_verts(), 5);
        assert_eq!(msh.n_elems(), 2);
        assert_eq!(msh.n_faces(), 5);

        let (msh, data): (SimplexMesh<2, Triangle>, _) = msh.to_simplices().unwrap();

        assert_eq!(msh.n_verts(), 5);
        assert_eq!(msh.n_elems(), 3);
        assert_eq!(msh.n_faces(), 5);

        let vol: f64 = (0..(msh.n_elems() as Idx)).map(|i| msh.elem_vol(i)).sum();
        assert!(f64::abs(vol - 1.25) < 1e-10);

        let arr = data
            .get(&("f1".to_string(), crate::FieldLocation::Vertex))
            .unwrap();
        assert_eq!(*arr, vec![0., 1., 1., 0., 1.5]);

        let arr = data
            .get(&("f2".to_string(), crate::FieldLocation::Element))
            .unwrap();
        assert_eq!(*arr, vec![1.0, 1.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_3d() -> Result<()> {
        let coords = vec![
            0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 1., 1., 1., 0.,
            1., 1., 0.5, 0.5, 1.5,
        ];
        let hexa = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let pyra = vec![4, 5, 6, 7, 8];
        let tris = vec![4, 5, 8, 5, 6, 8, 6, 7, 8, 7, 4, 8];
        let quads = vec![0, 1, 2, 3, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7];

        let mut msh = MultiElementMesh::new(3);
        msh.set_verts(&coords);
        msh.add_elems(ElementType::Hexahedron, &hexa, &[1]);
        msh.add_elems(ElementType::Pyramids, &pyra, &[1]);
        msh.add_elems(ElementType::Triangle, &tris, &[1; 4]);
        msh.add_elems(ElementType::Quadrangle, &quads, &[2; 5]);

        msh.add_field("f2", crate::FieldLocation::Element, vec![1.0, 2.0])?;

        assert_eq!(msh.n_verts(), 9);
        assert_eq!(msh.n_elems(), 2);
        assert_eq!(msh.n_faces(), 9);

        let (mut msh, data): (SimplexMesh<3, Tetrahedron>, _) = msh.to_simplices().unwrap();
        assert_eq!(msh.n_verts(), 9);
        assert_eq!(msh.n_elems(), 8);
        assert_eq!(msh.n_faces(), 14);

        // Build external faces
        msh.add_boundary_faces();
        assert_eq!(msh.n_verts(), 9);
        assert_eq!(msh.n_elems(), 8);
        assert_eq!(msh.n_faces(), 14);

        let vol: f64 = (0..(msh.n_elems() as Idx)).map(|i| msh.elem_vol(i)).sum();
        assert!(f64::abs(vol - 7. / 6.) < 1e-10);

        let arr = data
            .get(&("f2".to_string(), crate::FieldLocation::Element))
            .unwrap();
        assert_eq!(*arr, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]);

        Ok(())
    }

    #[test]
    fn test_3d_bdy() -> Result<()> {
        let coords = vec![
            0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 1., 1., 1., 0.,
            1., 1., 0.5, 0.5, 1.5,
        ];
        let hexa = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let pyra = vec![4, 5, 6, 7, 8];
        let tris = vec![4, 5, 8, 5, 6, 8, 6, 7, 8, 7, 4, 8];

        let mut msh = MultiElementMesh::new(3);
        msh.set_verts(&coords);
        msh.add_elems(ElementType::Hexahedron, &hexa, &[1]);
        msh.add_elems(ElementType::Pyramids, &pyra, &[1]);
        msh.add_elems(ElementType::Triangle, &tris, &[1; 4]);

        msh.add_field("f2", crate::FieldLocation::Element, vec![1.0, 2.0])?;

        assert_eq!(msh.n_verts(), 9);
        assert_eq!(msh.n_elems(), 2);
        assert_eq!(msh.n_faces(), 4);

        let (mut msh, _): (SimplexMesh<3, Tetrahedron>, _) = msh.to_simplices().unwrap();
        assert_eq!(msh.n_verts(), 9);
        assert_eq!(msh.n_elems(), 8);
        assert_eq!(msh.n_faces(), 4);

        // Build external faces
        msh.add_boundary_faces();
        assert_eq!(msh.n_verts(), 9);
        assert_eq!(msh.n_elems(), 8);
        assert_eq!(msh.n_faces(), 14);

        let bmsh = msh.boundary();
        assert_eq!(bmsh.n_verts(), 9);
        assert_eq!(bmsh.n_elems(), 14);
        assert_eq!(bmsh.n_faces(), 0);

        Ok(())
    }
}
