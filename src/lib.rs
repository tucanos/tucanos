use core::fmt;

use nalgebra::SVector;

pub mod boundary_mesh_2d;
pub mod boundary_mesh_3d;
pub mod dual_mesh;
pub mod dual_mesh_2d;
pub mod dual_mesh_3d;
pub mod graph;
mod least_squares;
pub mod mesh;
pub mod mesh_2d;
pub mod mesh_3d;
pub mod poly_mesh;
mod simplices;
mod to_simplices;
mod vtu_output;

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

pub type Tag = i16;

pub type Vertex<const D: usize> = SVector<f64, D>;

pub type Vert2d = Vertex<2>;
pub type Vert3d = Vertex<3>;

pub type Cell<const C: usize> = [usize; C];
pub type Face<const F: usize> = Cell<F>;

pub type Hexahedron = Cell<8>;
pub type Prism = Cell<6>;
pub type Pyramid = Cell<5>;
pub type Tetrahedron = Cell<4>;
pub type Quadrangle = Cell<4>;
pub type Triangle = Cell<3>;
pub type Edge = Cell<2>;
pub type Node = Cell<1>;

#[macro_export]
macro_rules! assert_delta {
    ($x:expr, $y:expr, $d:expr) => {
        assert!(
            ($x - $y).abs() < $d,
            "({} - {}).abs() = {}",
            $x,
            $y,
            ($x - $y).abs()
        )
    };
}
