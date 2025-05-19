//! Simplex meshes in 2D and 3D
use core::fmt;
use nalgebra::SVector;

pub mod boundary_mesh_2d;
pub mod boundary_mesh_3d;
pub mod dual_mesh;
pub mod dual_mesh_2d;
pub mod dual_mesh_3d;
pub mod extruded;
pub mod graph;
mod least_squares;
pub mod mesh;
pub mod mesh_2d;
pub mod mesh_3d;
pub mod poly_mesh;
pub mod simplices;
mod to_simplices;
mod vtu_output;

/// Result
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Error
#[derive(Debug)]
pub struct Error(String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "There is an error: {}", self.0)
    }
}

impl std::error::Error for Error {}

impl Error {
    /// Set the error message
    #[must_use]
    pub fn from(msg: &str) -> Box<Self> {
        Box::new(Self(msg.into()))
    }
}

/// Tag used for elements and face
pub type Tag = i16;

/// Vertex in D dimensions
pub type Vertex<const D: usize> = SVector<f64, D>;
/// Vertex in 2D
pub type Vert2d = Vertex<2>;
/// Vertex in 3D
pub type Vert3d = Vertex<3>;

/// Cell
pub type Cell<const C: usize> = [usize; C];
/// Face
pub type Face<const F: usize> = Cell<F>;

/// Hexahedron
pub type Hexahedron = Cell<8>;
/// Prism
pub type Prism = Cell<6>;
/// Pyramid
pub type Pyramid = Cell<5>;
/// Tetrahedron
pub type Tetrahedron = Cell<4>;
/// Quadrangle
pub type Quadrangle = Cell<4>;
/// Triangle
pub type Triangle = Cell<3>;
/// Edge
pub type Edge = Cell<2>;
/// Node
pub type Node = Cell<1>;

/// Assert that two floating point values are closer than a tolerance
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
