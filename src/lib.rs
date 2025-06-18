use core::fmt;

pub mod geometry;
pub mod mesh;
pub mod metric;
pub mod remesher;

/// Isotropic or anisotropic remesher in 2 and 3 dimensions
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

/// Topological tags: dimension
pub type Dim = i8;
/// Topological tags: tag
#[cfg(all(feature = "32bit-tags", feature = "64bit-tags"))]
compile_error!("features `32bit-tags` and `64bit-tags` are mutually exclusive");
#[cfg(feature = "64bit-tags")]
pub type Tag = i64;
#[cfg(feature = "32bit-tags")]
pub type Tag = i32;
#[cfg(not(any(feature = "32bit-tags", feature = "64bit-tags")))]
pub type Tag = i16;
/// Topological tags: dimension and tag
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

// Set the log level for tests
pub fn init_log(level: &str) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level))
        .format_timestamp(None)
        .init();
}
