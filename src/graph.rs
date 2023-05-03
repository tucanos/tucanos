use std::hash::BuildHasherDefault;

use crate::Idx;
use rustc_hash::FxHashMap;

pub trait Connectivity {
    fn elem(&self, i: Idx) -> &[Idx];
    fn n_elems(&self) -> Idx;
    fn max_node(&self) -> Idx;
    fn n(&self) -> Idx;
}

#[derive(Debug, Default, Clone)]
pub struct ElemGraph {
    pub n: Idx,
    pub elems: Vec<Idx>,
}

impl ElemGraph {
    #[must_use]
    pub fn new(n: Idx, n_elems: Idx) -> Self {
        Self {
            n,
            elems: Vec::with_capacity((n * n_elems) as usize),
        }
    }

    pub fn from<I: ExactSizeIterator<Item = Idx>>(n: Idx, els: I) -> Self {
        debug_assert_eq!(els.len() % n as usize, 0);
        Self {
            n,
            elems: els.collect(),
        }
    }

    pub fn add_elem(&mut self, e: &[Idx]) {
        debug_assert_eq!(e.len(), self.n as usize);
        self.elems.extend(e.iter());
    }

    pub fn reindex(&mut self) -> FxHashMap<Idx, Idx> {
        let mut map = FxHashMap::with_hasher(BuildHasherDefault::default());
        let mut next = 0;
        for e in &mut self.elems {
            if let Some(i0) = map.get(e) {
                *e = *i0;
            } else {
                map.insert(*e, next);
                *e = next;
                next += 1;
            }
        }
        map
    }
}

impl Connectivity for ElemGraph {
    fn elem(&self, i: Idx) -> &[Idx] {
        let start = (self.n * i) as usize;
        let end = start + self.n as usize;
        &self.elems[start..end]
    }

    fn n_elems(&self) -> Idx {
        debug_assert_eq!(self.elems.len() % self.n as usize, 0);
        (self.elems.len() / self.n as usize) as Idx
    }

    fn max_node(&self) -> Idx {
        self.elems.iter().copied().max().unwrap()
    }

    fn n(&self) -> Idx {
        self.n
    }
}

#[derive(Debug)]
pub struct ElemGraphInterface<'a> {
    n: Idx,
    elems: &'a [Idx],
}

impl<'a> ElemGraphInterface<'a> {
    #[must_use]
    pub fn new(n: Idx, elems: &'a [Idx]) -> Self {
        Self { n, elems }
    }
}

impl<'a> Connectivity for ElemGraphInterface<'a> {
    fn elem(&self, i: Idx) -> &[Idx] {
        let start = (self.n * i) as usize;
        let end = start + self.n as usize;
        &self.elems[start..end]
    }

    fn n_elems(&self) -> Idx {
        debug_assert_eq!(self.elems.len() % self.n as usize, 0);
        (self.elems.len() / self.n as usize) as Idx
    }

    fn max_node(&self) -> Idx {
        self.elems.iter().copied().max().unwrap()
    }

    fn n(&self) -> Idx {
        self.n
    }
}

#[derive(Debug, Default, Clone)]
pub struct CSRGraph {
    pub ptr: Vec<Idx>,
    pub indices: Vec<Idx>,
    pub m: Idx,
}

impl CSRGraph {
    fn set_ptr<G: Connectivity>(g: &G) -> Self {
        let nv = g.max_node() as usize + 1;
        let ne = g.n_elems() as usize;

        let mut res = Self {
            ptr: vec![0; nv + 1],
            indices: vec![Idx::MAX; g.n() as usize * ne],
            m: 0,
        };

        for i in 0..ne {
            for e in g.elem(i as Idx).iter().copied() {
                res.ptr[e as usize + 1] += 1;
            }
        }

        for i in 0..nv {
            res.ptr[i + 1] += res.ptr[i];
        }

        res
    }

    pub fn sort(&mut self) {
        let n = self.ptr.len() - 1;
        for i in 0..n {
            let start = self.ptr[i] as usize;
            let end = self.ptr[i + 1] as usize;
            self.indices[start..end].sort_unstable();
        }
    }

    pub fn new<G: Connectivity>(g: &G) -> Self {
        assert_eq!(g.n(), 2);
        let mut res = Self::set_ptr(g);
        let ne = g.n_elems() as usize;
        res.m = res.n();

        for i in 0..ne {
            let e = g.elem(i as Idx);
            let i0 = e[0] as usize;
            let i1 = e[1] as usize;
            let mut ok = false;
            for j in res.ptr[i0]..res.ptr[i0 + 1] {
                if res.indices[j as usize] == Idx::MAX {
                    res.indices[j as usize] = i1 as Idx;
                    ok = true;
                    break;
                }
            }
            assert!(ok);
            let mut ok = false;
            for j in res.ptr[i1]..res.ptr[i1 + 1] {
                if res.indices[j as usize] == Idx::MAX {
                    res.indices[j as usize] = i0 as Idx;
                    ok = true;
                    break;
                }
            }
            assert!(ok);
        }
        res.sort();
        res
    }

    pub fn transpose<G: Connectivity>(g: &G) -> Self {
        let mut res = Self::set_ptr(g);
        let ne = g.n_elems();
        res.m = ne;
        for i in 0..ne {
            let e = g.elem(i);
            for i_vert in e.iter().copied() {
                let start = res.ptr[i_vert as usize];
                let end = res.ptr[i_vert as usize + 1];
                let mut ok = false;
                for j in start..end {
                    if res.indices[j as usize] == Idx::MAX {
                        res.indices[j as usize] = i;
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
        }
        res.sort();
        res
    }

    #[must_use]
    pub fn n(&self) -> Idx {
        self.ptr.len() as Idx - 1
    }

    #[must_use]
    pub fn m(&self) -> Idx {
        self.m
    }

    #[must_use]
    pub fn n_edges(&self) -> Idx {
        self.indices.len() as Idx
    }

    #[must_use]
    pub fn row(&self, i: Idx) -> &[Idx] {
        let start = self.ptr[i as usize] as usize;
        let end = self.ptr[i as usize + 1] as usize;
        &self.indices[start..end]
    }
}

#[derive(Debug)]
pub struct ConnectedComponents {
    vtag: Vec<u16>,
}

impl ConnectedComponents {
    #[must_use]
    pub fn new(g: &CSRGraph) -> Self {
        assert_eq!(g.n(), g.m());

        let mut res = Self {
            vtag: vec![u16::MAX; g.n() as usize],
        };
        res.compute(g);
        res
    }

    fn compute_from(&mut self, g: &CSRGraph, start: Idx, component: u16) {
        for i in g.row(start).iter().copied() {
            if self.vtag[i as usize] == u16::MAX {
                self.vtag[i as usize] = component;
                self.compute_from(g, i, component);
            }
        }
    }

    fn compute(&mut self, g: &CSRGraph) {
        let mut start = 0;
        let mut component = 0;
        while start < g.n() {
            self.compute_from(g, start, component);
            while start < g.n() && self.vtag[start as usize] != u16::MAX {
                start += 1;
            }
            component += 1;
            assert!(component < u16::MAX);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create() {
        let mut g = ElemGraph::new(2, 4);
        g.add_elem(&[0, 1]);
        g.add_elem(&[1, 2]);
        g.add_elem(&[2, 0]);
        g.add_elem(&[5, 6]);
        assert_eq!(g.max_node(), 6);
        assert_eq!(g.n_elems(), 4);
    }

    #[test]
    fn test_reindex() {
        let mut g = ElemGraph::new(2, 4);
        g.add_elem(&[0, 1]);
        g.add_elem(&[1, 2]);
        g.add_elem(&[2, 0]);
        g.add_elem(&[5, 6]);
        g.reindex();
        assert_eq!(g.max_node(), 4);
        assert_eq!(g.n_elems(), 4);
    }

    #[test]
    fn test_csr_edges() {
        let mut g = ElemGraph::new(2, 4);
        g.add_elem(&[0, 1]);
        g.add_elem(&[1, 2]);
        g.add_elem(&[2, 0]);
        g.add_elem(&[3, 4]);
        let g = CSRGraph::new(&g);
        assert_eq!(g.n(), 5);
        assert_eq!(g.m(), 5);
        assert_eq!(g.n_edges(), 8);
        let edgs = g.row(0);
        assert_eq!(*edgs, [1, 2]);
        let edgs = g.row(1);
        assert_eq!(*edgs, [0, 2]);
        let edgs = g.row(2);
        assert_eq!(*edgs, [0, 1]);
        let edgs = g.row(3);
        assert_eq!(*edgs, [4]);
        let edgs = g.row(4);
        assert_eq!(*edgs, [3]);
    }

    #[test]
    fn test_csr_triangles() {
        let mut g = ElemGraph::new(3, 4);
        g.add_elem(&[0, 1, 2]);
        g.add_elem(&[0, 2, 3]);
        let g = CSRGraph::transpose(&g);
        assert_eq!(g.n(), 4);
        assert_eq!(g.m(), 2);

        let edgs = g.row(0);
        assert_eq!(*edgs, [0, 1]);
        let edgs = g.row(1);
        assert_eq!(*edgs, [0]);
        let edgs = g.row(2);
        assert_eq!(*edgs, [0, 1]);
        let edgs = g.row(3);
        assert_eq!(*edgs, [1]);
    }

    #[test]
    fn test_csr_triangles_2() {
        let tmp = [0, 1, 2, 0, 2, 3];

        let g = ElemGraphInterface::new(3, &tmp);
        let g = CSRGraph::transpose(&g);
        assert_eq!(g.n(), 4);
        assert_eq!(g.m(), 2);

        let edgs = g.row(0);
        assert_eq!(*edgs, [0, 1]);
        let edgs = g.row(1);
        assert_eq!(*edgs, [0]);
        let edgs = g.row(2);
        assert_eq!(*edgs, [0, 1]);
        let edgs = g.row(3);
        assert_eq!(*edgs, [1]);
    }

    #[test]
    fn test_cc() {
        let mut g = ElemGraph::new(2, 4);
        g.add_elem(&[0, 1]);
        g.add_elem(&[1, 2]);
        g.add_elem(&[2, 0]);
        g.add_elem(&[3, 4]);
        let g = CSRGraph::new(&g);
        let components = ConnectedComponents::new(&g).vtag;
        assert_eq!(components, [0, 0, 0, 1, 1]);
    }
}
