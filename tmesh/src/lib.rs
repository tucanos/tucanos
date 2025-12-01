//! Simplex meshes in 2D and 3D
use core::fmt;
use nalgebra::SVector;

pub mod dual;
pub mod extruded;
pub mod graph;
pub mod interpolate;
pub mod io;
pub mod mesh;
pub mod spatialindex;

pub use minimeshb;

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

#[cfg(all(feature = "32bit-tags", feature = "64bit-tags"))]
compile_error!("features `32bit-tags` and `64bit-tags` are mutually exclusive");
#[cfg(feature = "64bit-tags")]
/// Tag used for elements and face
pub type Tag = i64;
#[cfg(feature = "32bit-tags")]
/// Tag used for elements and face
pub type Tag = i32;
#[cfg(not(any(feature = "32bit-tags", feature = "64bit-tags")))]
/// Tag used for elements and face
pub type Tag = i16;

/// Vertex in D dimensions
pub type Vertex<const D: usize> = SVector<f64, D>;
/// Vertex in 2D
pub type Vert2d = Vertex<2>;
/// Vertex in 3D
pub type Vert3d = Vertex<3>;

/// Assert that two floating point values are closer than a tolerance
#[macro_export]
macro_rules! assert_delta {
    ($x:expr, $y:expr, $d:expr) => {
        assert!(
            ($x - $y).abs() < $d,
            "({:.3e} - {:.3e}).abs() = {:.3e}",
            $x,
            $y,
            ($x - $y).abs()
        )
    };
}
