use crate::{
    mesh::{Elem, QuadraticElem},
    Idx, Tag,
};

pub trait FromNativePointer: Send {
    type PointerType: std::fmt::Debug;
    const SIZE: usize;
    fn from_ptr(ptr: *const Self::PointerType) -> Self;
}

pub trait FromNativePointerQuadratic: Send {
    type PointerType: std::fmt::Debug;
    const SIZE: usize;
    fn from_ptr(ptr: *const Self::PointerType) -> Self;
}

impl FromNativePointer for Tag {
    type PointerType = Self;
    const SIZE: usize = 1;

    fn from_ptr(ptr: *const Self::PointerType) -> Self {
        unsafe { *ptr }
    }
}

impl FromNativePointerQuadratic for Tag {
    type PointerType = Self;
    const SIZE: usize = 1;

    fn from_ptr(ptr: *const Self::PointerType) -> Self {
        unsafe { *ptr }
    }
}

impl<E: Elem> FromNativePointer for E {
    type PointerType = Idx;
    const SIZE: usize = E::N_VERTS as usize;

    fn from_ptr(ptr: *const Self::PointerType) -> Self {
        let pts = unsafe { std::slice::from_raw_parts(ptr, E::N_VERTS as usize) };
        E::from_slice(pts)
    }
}

impl<QE: QuadraticElem> FromNativePointerQuadratic for QE {
    type PointerType = Idx;
    const SIZE: usize = QE::N_VERTS as usize;

    fn from_ptr(ptr: *const Self::PointerType) -> Self {
        let pts = unsafe { std::slice::from_raw_parts(ptr, QE::N_VERTS as usize) };
        QE::from_slice(pts)
    }
}

impl<const D: usize> FromNativePointer for crate::mesh::Point<D> {
    type PointerType = f64;
    const SIZE: usize = D;
    fn from_ptr(ptr: *const Self::PointerType) -> Self {
        let s = unsafe { std::slice::from_raw_parts(ptr, 3) };
        Self::from_row_slice(s)
    }
}

impl<const D: usize> FromNativePointerQuadratic for crate::mesh::Point<D> {
    type PointerType = f64;
    const SIZE: usize = D;
    fn from_ptr(ptr: *const Self::PointerType) -> Self {
        let s = unsafe { std::slice::from_raw_parts(ptr, 3) };
        Self::from_row_slice(s)
    }
}

#[derive(Debug)]
enum VectorImpl<T: FromNativePointer> {
    Std(Vec<T>),
    Native((*const T::PointerType, usize)),
}

#[derive(Debug)]
enum VectorImplQuadratic<T: FromNativePointerQuadratic> {
    Std(Vec<T>),
    Native((*const T::PointerType, usize)),
}

impl<T: FromNativePointer> VectorImpl<T> {
    pub fn len(&self) -> usize {
        match &self {
            Self::Std(x) => x.len(),
            Self::Native((_, s)) => *s,
        }
    }
}

impl<T: FromNativePointerQuadratic> VectorImplQuadratic<T> {
    pub fn len(&self) -> usize {
        match &self {
            Self::Std(x) => x.len(),
            Self::Native((_, s)) => *s,
        }
    }
}

unsafe impl<T: FromNativePointer> Sync for VectorImpl<T> {}
unsafe impl<T: FromNativePointer> Send for VectorImpl<T> {}

unsafe impl<T: FromNativePointerQuadratic> Sync for VectorImplQuadratic<T> {}
unsafe impl<T: FromNativePointerQuadratic> Send for VectorImplQuadratic<T> {}

struct NativeIter<T: FromNativePointer> {
    cur: *const T::PointerType,
    end: *const T::PointerType,
}

struct NativeIterQuadratic<T: FromNativePointerQuadratic> {
    cur: *const T::PointerType,
    end: *const T::PointerType,
}

impl<T: FromNativePointer> Iterator for NativeIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len() > 0 {
            let r = T::from_ptr(self.cur);
            unsafe { self.cur = self.cur.add(T::SIZE) };
            Some(r)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let p = self.len();
        (p, Some(p))
    }
}

impl<T: FromNativePointerQuadratic> Iterator for NativeIterQuadratic<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len() > 0 {
            let r = T::from_ptr(self.cur);
            unsafe { self.cur = self.cur.add(T::SIZE) };
            Some(r)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let p = self.len();
        (p, Some(p))
    }
}

impl<T: FromNativePointer> ExactSizeIterator for NativeIter<T> {
    fn len(&self) -> usize {
        let o = unsafe { self.end.offset_from(self.cur) };
        debug_assert!(o >= 0, "{o}");
        o as usize / T::SIZE
    }
}

impl<T: FromNativePointerQuadratic> ExactSizeIterator for NativeIterQuadratic<T> {
    fn len(&self) -> usize {
        let o = unsafe { self.end.offset_from(self.cur) };
        debug_assert!(o >= 0, "{o}");
        o as usize / T::SIZE
    }
}

enum Iter<'a, T: FromNativePointer> {
    Std(std::iter::Copied<std::slice::Iter<'a, T>>),
    Native(NativeIter<T>),
}

enum IterQuadratic<'a, T: FromNativePointerQuadratic> {
    Std(std::iter::Copied<std::slice::Iter<'a, T>>),
    Native(NativeIterQuadratic<T>),
}

impl<T: Copy + FromNativePointer> Iterator for Iter<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Iter::Std(x) => x.next(),
            Iter::Native(x) => x.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Iter::Std(x) => x.size_hint(),
            Iter::Native(x) => x.size_hint(),
        }
    }
}

impl<T: Copy + FromNativePointerQuadratic> Iterator for IterQuadratic<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IterQuadratic::Std(x) => x.next(),
            IterQuadratic::Native(x) => x.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            IterQuadratic::Std(x) => x.size_hint(),
            IterQuadratic::Native(x) => x.size_hint(),
        }
    }
}

impl<T: Copy + FromNativePointer> ExactSizeIterator for Iter<'_, T> {
    fn len(&self) -> usize {
        match self {
            Iter::Std(x) => x.len(),
            Iter::Native(x) => x.len(),
        }
    }
}

impl<T: Copy + FromNativePointerQuadratic> ExactSizeIterator for IterQuadratic<'_, T> {
    fn len(&self) -> usize {
        match self {
            IterQuadratic::Std(x) => x.len(),
            IterQuadratic::Native(x) => x.len(),
        }
    }
}

#[derive(Debug)]
pub struct Vector<T: FromNativePointer> {
    data: VectorImpl<T>,
}

const VECTOR_MSG: &str =
    "This fonction is not supported with the C array backed mesh. Call tucanos_mesh_clone().";

impl<T: Copy + FromNativePointer> Vector<T> {
    pub fn as_std(&self) -> &Vec<T> {
        match &self.data {
            VectorImpl::Std(x) => x,
            VectorImpl::Native(_) => unimplemented!("{VECTOR_MSG}"),
        }
    }
    pub fn as_std_mut(&mut self) -> &mut Vec<T> {
        match &mut self.data {
            VectorImpl::Std(x) => x,
            VectorImpl::Native(_) => unimplemented!("{VECTOR_MSG}"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = T> + '_ {
        match &self.data {
            VectorImpl::Std(x) => Iter::Std(x.iter().copied()),
            VectorImpl::Native((p, s)) => Iter::Native(NativeIter {
                cur: *p,
                end: unsafe { p.add(*s * T::SIZE) },
            }),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
    }

    pub fn index(&self, i: Idx) -> T {
        let i = i as usize;
        match &self.data {
            VectorImpl::Std(x) => x[i],
            VectorImpl::Native((p, s)) => {
                debug_assert!(i < *s);
                let p = unsafe { p.add(i * T::SIZE) };
                T::from_ptr(p)
            }
        }
    }

    pub fn push(&mut self, v: T) {
        match &mut self.data {
            VectorImpl::Std(x) => x.push(v),
            VectorImpl::Native(_) => panic!("Cannot push to native vectors"),
        }
    }
}

impl<T: Clone + FromNativePointer + Copy> Clone for Vector<T> {
    fn clone(&self) -> Self {
        match &self.data {
            VectorImpl::Std(x) => Self {
                data: VectorImpl::Std(x.clone()),
            },
            VectorImpl::Native(_) => Self {
                data: VectorImpl::Std(self.iter().collect()),
            },
        }
    }
}
impl<T: FromNativePointer> Default for Vector<T> {
    fn default() -> Self {
        Self {
            data: VectorImpl::Std(Vec::new()),
        }
    }
}

impl<T: FromNativePointer> From<Vec<T>> for Vector<T> {
    fn from(value: Vec<T>) -> Self {
        Self {
            data: VectorImpl::Std(value),
        }
    }
}

impl<T: FromNativePointer> From<(*const T::PointerType, usize)> for Vector<T> {
    fn from(value: (*const T::PointerType, usize)) -> Self {
        Self {
            data: VectorImpl::Native(value),
        }
    }
}

#[derive(Debug)]
pub struct VectorQuadratic<T: FromNativePointerQuadratic> {
    data: VectorImplQuadratic<T>,
}

const VECTORQUADRATIC_MSG: &str =
    "This fonction is not supported with the C array backed mesh. Call tucanos_mesh_clone().";

impl<T: Copy + FromNativePointerQuadratic> VectorQuadratic<T> {
    pub fn as_std(&self) -> &Vec<T> {
        match &self.data {
            VectorImplQuadratic::Std(x) => x,
            VectorImplQuadratic::Native(_) => unimplemented!("{VECTOR_MSG}"),
        }
    }
    pub fn as_std_mut(&mut self) -> &mut Vec<T> {
        match &mut self.data {
            VectorImplQuadratic::Std(x) => x,
            VectorImplQuadratic::Native(_) => unimplemented!("{VECTOR_MSG}"),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = T> + '_ {
        match &self.data {
            VectorImplQuadratic::Std(x) => IterQuadratic::Std(x.iter().copied()),
            VectorImplQuadratic::Native((p, s)) => IterQuadratic::Native(NativeIterQuadratic {
                cur: *p,
                end: unsafe { p.add(*s * T::SIZE) },
            }),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.data.len() == 0
    }

    pub fn index(&self, i: Idx) -> T {
        let i = i as usize;
        match &self.data {
            VectorImplQuadratic::Std(x) => x[i],
            VectorImplQuadratic::Native((p, s)) => {
                debug_assert!(i < *s);
                let p = unsafe { p.add(i * T::SIZE) };
                T::from_ptr(p)
            }
        }
    }

    pub fn push(&mut self, v: T) {
        match &mut self.data {
            VectorImplQuadratic::Std(x) => x.push(v),
            VectorImplQuadratic::Native(_) => panic!("Cannot push to native vectors"),
        }
    }
}

impl<T: Clone + FromNativePointerQuadratic + Copy> Clone for VectorQuadratic<T> {
    fn clone(&self) -> Self {
        match &self.data {
            VectorImplQuadratic::Std(x) => Self {
                data: VectorImplQuadratic::Std(x.clone()),
            },
            VectorImplQuadratic::Native(_) => Self {
                data: VectorImplQuadratic::Std(self.iter().collect()),
            },
        }
    }
}
impl<T: FromNativePointerQuadratic> Default for VectorQuadratic<T> {
    fn default() -> Self {
        Self {
            data: VectorImplQuadratic::Std(Vec::new()),
        }
    }
}

impl<T: FromNativePointerQuadratic> From<Vec<T>> for VectorQuadratic<T> {
    fn from(value: Vec<T>) -> Self {
        Self {
            data: VectorImplQuadratic::Std(value),
        }
    }
}

impl<T: FromNativePointerQuadratic> From<(*const T::PointerType, usize)> for VectorQuadratic<T> {
    fn from(value: (*const T::PointerType, usize)) -> Self {
        Self {
            data: VectorImplQuadratic::Native(value),
        }
    }
}
