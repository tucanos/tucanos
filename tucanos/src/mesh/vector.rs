use crate::{Idx, Tag, mesh::Elem};

pub trait FromNativePointer: Send {
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

impl<E: Elem> FromNativePointer for E {
    type PointerType = Idx;
    const SIZE: usize = E::N_VERTS as usize;

    fn from_ptr(ptr: *const Self::PointerType) -> Self {
        let pts = unsafe { std::slice::from_raw_parts(ptr, E::N_VERTS as usize) };
        E::from_slice(pts)
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

#[derive(Debug)]
enum VectorImpl<T: FromNativePointer> {
    Std(Vec<T>),
    Native((*const T::PointerType, usize)),
}

impl<T: FromNativePointer> VectorImpl<T> {
    pub const fn len(&self) -> usize {
        match &self {
            Self::Std(x) => x.len(),
            Self::Native((_, s)) => *s,
        }
    }
}

unsafe impl<T: FromNativePointer> Sync for VectorImpl<T> {}
unsafe impl<T: FromNativePointer> Send for VectorImpl<T> {}

#[derive(Clone)]
struct NativeIter<T: FromNativePointer + Clone>
where
    T::PointerType: Clone,
{
    cur: *const T::PointerType,
    end: *const T::PointerType,
}

impl<T: FromNativePointer + Clone> Iterator for NativeIter<T>
where
    T::PointerType: Clone,
{
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

impl<T: FromNativePointer + Clone> ExactSizeIterator for NativeIter<T>
where
    T::PointerType: Clone,
{
    fn len(&self) -> usize {
        let o = unsafe { self.end.offset_from(self.cur) };
        debug_assert!(o >= 0, "{o}");
        o as usize / T::SIZE
    }
}

#[derive(Clone)]
enum Iter<'a, T: FromNativePointer + Clone>
where
    T::PointerType: Clone,
{
    Std(std::iter::Copied<std::slice::Iter<'a, T>>),
    Native(NativeIter<T>),
}

impl<T: Copy + FromNativePointer> Iterator for Iter<'_, T>
where
    T::PointerType: Clone,
{
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

impl<T: Copy + FromNativePointer> ExactSizeIterator for Iter<'_, T>
where
    T::PointerType: Clone,
{
    fn len(&self) -> usize {
        match self {
            Iter::Std(x) => x.len(),
            Iter::Native(x) => x.len(),
        }
    }
}

#[derive(Debug)]
pub struct Vector<T: FromNativePointer> {
    data: VectorImpl<T>,
}

const VECTOR_MSG: &str =
    "This fonction is not supported with the C array backed mesh. Call tucanos_mesh_clone().";

impl<T: Copy + FromNativePointer> Vector<T>
where
    T::PointerType: Clone,
{
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

    pub const fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = T> + Clone + '_ {
        match &self.data {
            VectorImpl::Std(x) => Iter::Std(x.iter().copied()),
            VectorImpl::Native((p, s)) => Iter::Native(NativeIter {
                cur: *p,
                end: unsafe { p.add(*s * T::SIZE) },
            }),
        }
    }
    pub const fn is_empty(&self) -> bool {
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

    pub fn index_mut(&mut self, i: Idx) -> &mut T {
        let i = i as usize;
        match &mut self.data {
            VectorImpl::Std(x) => &mut x[i],
            VectorImpl::Native(_) => panic!("Cannot use index_mut native vectors"),
        }
    }

    pub fn push(&mut self, v: T) {
        match &mut self.data {
            VectorImpl::Std(x) => x.push(v),
            VectorImpl::Native(_) => panic!("Cannot push to native vectors"),
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        match &mut self.data {
            VectorImpl::Std(x) => x.reserve(additional),
            VectorImpl::Native(_) => panic!("Cannot reserve native vectors"),
        }
    }

    pub fn clear(&mut self) {
        match &mut self.data {
            VectorImpl::Std(x) => x.clear(),
            VectorImpl::Native(_) => panic!("Cannot clear native vectors"),
        }
    }
}

impl<T: Clone + FromNativePointer + Copy> Clone for Vector<T>
where
    T::PointerType: Clone,
{
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
