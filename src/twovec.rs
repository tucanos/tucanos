use std::ops::Index;

/// A vec optimized for storing 2 elements
///
/// It's even smaller than the smallvec crate.
#[derive(Debug, Clone)]
pub struct Vec<T> {
    data: Data<T>,
}

#[derive(Debug, Clone)]
enum Data<T> {
    One(T),
    Two([T; 2]),
    // Box indirection to make Data smaller. Because of alignment
    // it make this enum 16B while an empty vec is 24B. Without
    // the Box it would have been 32B.
    #[allow(clippy::box_collection)]
    Many(Box<std::vec::Vec<T>>),
}

impl<T: Copy> Vec<T> {
    pub fn len(&self) -> usize {
        match &self.data {
            Data::One(_) => 1,
            Data::Two(_) => 2,
            Data::Many(v) => v.len(),
        }
    }

    pub fn with_single(v: T) -> Self {
        Self { data: Data::One(v) }
    }

    pub fn push(&mut self, v: T) {
        match &mut self.data {
            Data::One(o) => self.data = Data::Two([*o, v]),
            Data::Two(pair) => self.data = Data::Many(Box::new(vec![pair[0], pair[1], v])),
            Data::Many(vc) => vc.push(v),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        VecIter {
            data: self,
            position: 0,
        }
    }
}

impl<T> Index<usize> for Vec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match &self.data {
            Data::One(v) => {
                debug_assert!(index == 0);
                v
            }
            Data::Two(v) => &v[index],
            Data::Many(v) => &v[index],
        }
    }
}

impl<T: Copy> IntoIterator for Vec<T> {
    type Item = T;
    type IntoIter = OwnedVecIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        OwnedVecIter {
            data: self,
            position: 0,
        }
    }
}

pub struct VecIter<'a, T> {
    data: &'a Vec<T>,
    position: usize,
}

pub struct OwnedVecIter<T> {
    data: Vec<T>,
    position: usize,
}

impl<'a, T: Copy> Iterator for VecIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.data.len() {
            let r = &self.data[self.position];
            self.position += 1;
            Some(r)
        } else {
            None
        }
    }
}

impl<T: Copy> Iterator for OwnedVecIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.data.len() {
            let r = self.data[self.position];
            self.position += 1;
            Some(r)
        } else {
            None
        }
    }
}
