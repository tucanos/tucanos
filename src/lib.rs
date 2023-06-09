use core::fmt;
mod cavity;
pub mod curvature;
pub mod geom_elems;
pub mod geometry;
pub mod graph;
mod linalg;
#[allow(dead_code)]
#[allow(clippy::enum_variant_names)]
pub mod mesh;
mod mesh_interpolate;
mod mesh_ls;
pub mod mesh_metric;
mod mesh_ordering;
mod mesh_split;
pub mod mesh_stl;
pub mod mesh_vtk;
#[cfg(feature = "libmeshb-sys")]
pub mod meshb_io;
pub mod metric;
mod metric_reduction;
pub mod multi_element_mesh;
mod octree;
pub mod remesher;
mod stats;
pub mod test_meshes; // to suppress warnings!
pub mod topo_elems;
mod topology;
mod twovec;

const H_MAX: f64 = 1e8;
const H_MIN: f64 = 1e-8;
const ANISO_MAX: f64 = 1e5;

const S_MIN: f64 = 1. / (H_MAX * H_MAX);
const S_MAX: f64 = 1. / (H_MIN * H_MIN);
const S_RATIO_MAX: f64 = ANISO_MAX * ANISO_MAX;
// Errors
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "There is an error: {}", self.0)
    }
}

impl std::error::Error for Error {}

impl Error {
    #[must_use]
    pub fn from(msg: &str) -> Box<Self> {
        Box::new(Self(msg.into()))
    }
}

/// Vertex and element indices
pub type Idx = u32;

/// Topological tags
pub type Dim = i8;
pub type Tag = i16;
pub type TopoTag = (Dim, Tag);

/// Return the minimum of an iterator of f64
pub fn min_iter<I: Iterator<Item = f64>>(it: I) -> f64 {
    it.fold(f64::INFINITY, f64::min)
}

/// Return the maximum of an iterator of f64
pub fn max_iter<I: Iterator<Item = f64>>(it: I) -> f64 {
    it.fold(f64::NEG_INFINITY, f64::max)
}

/// Return the minimum and maximum of an iterator of f64
pub fn min_max_iter<I: Iterator<Item = f64>>(it: I) -> (f64, f64) {
    it.fold((f64::INFINITY, f64::NEG_INFINITY), |a, b| {
        (a.0.min(b), a.1.max(b))
    })
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub enum FieldLocation {
    Vertex,
    Element,
    Constant,
}

#[derive(Debug)]
pub enum FieldType {
    Scalar,
    Vector,
    SymTensor,
}

pub trait Mesh {
    fn n_verts(&self) -> Idx;
    fn n_elems(&self) -> Idx;
    fn n_faces(&self) -> Idx;
    fn dim(&self) -> Idx;
    fn cell_dim(&self) -> Idx;
    fn n_comps(&self, ftype: FieldType) -> Idx {
        match ftype {
            FieldType::Scalar => 1,
            FieldType::Vector => self.dim(),
            FieldType::SymTensor => self.dim() * (self.dim() + 1) / 2,
        }
    }
    fn field_type(&self, vec: &[f64], loc: FieldLocation) -> Option<FieldType> {
        let n = match loc {
            FieldLocation::Vertex => self.n_verts(),
            FieldLocation::Element => self.n_elems(),
            FieldLocation::Constant => 1,
        };
        let m = vec.len() as Idx;
        let size = m / n;
        assert!(m % n == 0);
        match size {
            1 => Some(FieldType::Scalar),
            x if x == self.dim() => Some(FieldType::Vector),
            x if x == (self.dim() * (self.dim() + 1) / 2) => Some(FieldType::SymTensor),
            _ => None,
        }
    }
}
