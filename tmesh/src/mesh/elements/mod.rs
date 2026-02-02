use std::convert::Infallible;
use std::fmt::Debug;
use std::hash::Hash;
use std::num::TryFromIntError;

pub(super) mod edge;
pub(super) mod ho_simplex;
pub(super) mod node;
pub(super) mod quadratic_edge;
pub(super) mod quadratic_tetrahedron;
pub(super) mod quadratic_triangle;
pub(super) mod quadratures;
pub(super) mod simplex;
pub(super) mod tetrahedron;
pub(super) mod to_simplices;
pub(super) mod triangle;
mod twovec;

pub trait Idx:
    TryInto<usize, Error = Self::ConvertError>
    + TryFrom<usize, Error = Self::ConvertError>
    + Hash
    + Clone
    + Copy
    + PartialOrd
    + Ord
    + Eq
    + Debug
    + Default
    + Send
    + Sync
    + 'static
{
    type ConvertError: std::error::Error + Debug;
}

impl Idx for usize {
    type ConvertError = Infallible;
}

impl Idx for u32 {
    type ConvertError = TryFromIntError;
}

impl Idx for i64 {
    type ConvertError = TryFromIntError;
}

impl Idx for i32 {
    type ConvertError = TryFromIntError;
}

/// Hexahedron
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Hexahedron<T: Idx>([T; 8]);

impl<T: Idx> Hexahedron<T> {
    #[must_use]
    pub fn new(indices: [usize; 8]) -> Self {
        Self(indices.map(|x| x.try_into().unwrap()))
    }

    pub fn get(&self, index: usize) -> usize {
        self.0[index].try_into().unwrap()
    }
}

impl<T: Idx> IntoIterator for Hexahedron<T> {
    type Item = usize;
    type IntoIter = std::iter::Map<std::array::IntoIter<T, 8>, fn(T) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| x.try_into().unwrap())
    }
}

impl<T: Idx> FromIterator<usize> for Hexahedron<T> {
    fn from_iter<T2: IntoIterator<Item = usize>>(iter: T2) -> Self {
        let mut res = Self::default();
        let mut count = 0;
        for (i, j) in iter.into_iter().enumerate() {
            assert!(i < 8);
            res.0[i] = j.try_into().unwrap();
            count += 1;
        }
        assert_eq!(count, 8);
        res
    }
}

/// Prism
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Prism<T: Idx>([T; 6]);

impl<T: Idx> Prism<T> {
    #[must_use]
    pub fn new(i0: usize, i1: usize, i2: usize, i3: usize, i4: usize, i5: usize) -> Self {
        Self([
            i0.try_into().unwrap(),
            i1.try_into().unwrap(),
            i2.try_into().unwrap(),
            i3.try_into().unwrap(),
            i4.try_into().unwrap(),
            i5.try_into().unwrap(),
        ])
    }

    pub fn get(&self, index: usize) -> usize {
        self.0[index].try_into().unwrap()
    }
}

impl<T: Idx> IntoIterator for Prism<T> {
    type Item = usize;
    type IntoIter = std::iter::Map<std::array::IntoIter<T, 6>, fn(T) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| x.try_into().unwrap())
    }
}

impl<T: Idx> FromIterator<usize> for Prism<T> {
    fn from_iter<T2: IntoIterator<Item = usize>>(iter: T2) -> Self {
        let mut res = Self::default();
        let mut count = 0;
        for (i, j) in iter.into_iter().enumerate() {
            assert!(i < 6);
            res.0[i] = j.try_into().unwrap();
            count += 1;
        }
        assert_eq!(count, 6);
        res
    }
}

/// Pyramid
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Pyramid<T: Idx>([T; 5]);

impl<T: Idx> Pyramid<T> {
    #[must_use]
    pub fn new(i0: usize, i1: usize, i2: usize, i3: usize, i4: usize) -> Self {
        Self([
            i0.try_into().unwrap(),
            i1.try_into().unwrap(),
            i2.try_into().unwrap(),
            i3.try_into().unwrap(),
            i4.try_into().unwrap(),
        ])
    }

    pub fn get(&self, index: usize) -> usize {
        self.0[index].try_into().unwrap()
    }
}

impl<T: Idx> IntoIterator for Pyramid<T> {
    type Item = usize;
    type IntoIter = std::iter::Map<std::array::IntoIter<T, 5>, fn(T) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| x.try_into().unwrap())
    }
}

impl<T: Idx> FromIterator<usize> for Pyramid<T> {
    fn from_iter<T2: IntoIterator<Item = usize>>(iter: T2) -> Self {
        let mut res = Self::default();
        let mut count = 0;
        for (i, j) in iter.into_iter().enumerate() {
            assert!(i < 5);
            res.0[i] = j.try_into().unwrap();
            count += 1;
        }
        assert_eq!(count, 5);
        res
    }
}

/// Quadrangle
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Quadrangle<T: Idx>([T; 4]);

impl<T: Idx> Quadrangle<T> {
    #[must_use]
    pub fn new(i0: usize, i1: usize, i2: usize, i3: usize) -> Self {
        Self([
            i0.try_into().unwrap(),
            i1.try_into().unwrap(),
            i2.try_into().unwrap(),
            i3.try_into().unwrap(),
        ])
    }

    pub fn get(&self, index: usize) -> usize {
        self.0[index].try_into().unwrap()
    }
}

impl<T: Idx> FromIterator<usize> for Quadrangle<T> {
    fn from_iter<T2: IntoIterator<Item = usize>>(iter: T2) -> Self {
        let mut res = Self::default();
        let mut count = 0;
        for (i, j) in iter.into_iter().enumerate() {
            assert!(i < 4);
            res.0[i] = j.try_into().unwrap();
            count += 1;
        }
        assert_eq!(count, 4);
        res
    }
}

impl<T: Idx> IntoIterator for Quadrangle<T> {
    type Item = usize;
    type IntoIter = std::iter::Map<std::array::IntoIter<T, 4>, fn(T) -> usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| x.try_into().unwrap())
    }
}
