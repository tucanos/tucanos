use super::Edge;
use crate::{Tag, mesh::Simplex};
use rustc_hash::FxHashMap;

pub fn split_edgs<C: Simplex>(
    elems_and_tags: impl ExactSizeIterator<Item = (C, Tag)>,
    edges: &FxHashMap<Edge<C::T>, usize>,
) -> (Vec<C>, Vec<Tag>) {
    assert_eq!(C::N_VERTS, 2);
    let new_n_elems = 2 * elems_and_tags.len();
    let mut elems = Vec::with_capacity(new_n_elems);
    let mut etags = Vec::with_capacity(new_n_elems);

    for (e, tag) in elems_and_tags {
        let edg = Edge::new(e.get(0), e.get(1)).sorted();
        let i = edges.get(&edg).unwrap();

        elems.push(C::from_iter([e.get(0), *i]));
        etags.push(tag);

        elems.push(C::from_iter([*i, e.get(1)]));
        etags.push(tag);
    }

    (elems, etags)
}

pub fn split_tris<C: Simplex>(
    elems_and_tags: impl ExactSizeIterator<Item = (C, Tag)>,
    edges: &FxHashMap<Edge<C::T>, usize>,
) -> (Vec<C>, Vec<Tag>) {
    assert_eq!(C::N_VERTS, 3);

    let new_n_elems = 4 * elems_and_tags.len();
    let mut elems = Vec::with_capacity(new_n_elems);
    let mut etags = Vec::with_capacity(new_n_elems);

    for (e, tag) in elems_and_tags {
        let edg = Edge::new(e.get(0), e.get(1)).sorted();
        let i2 = edges.get(&edg).unwrap();

        let edg = Edge::new(e.get(1), e.get(2)).sorted();
        let i0 = edges.get(&edg).unwrap();

        let edg = Edge::new(e.get(2), e.get(0)).sorted();
        let i1 = edges.get(&edg).unwrap();

        elems.push(C::from_iter([e.get(0), *i2, *i1]));
        etags.push(tag);

        elems.push(C::from_iter([*i2, e.get(1), *i0]));
        etags.push(tag);

        elems.push(C::from_iter([*i2, *i0, *i1]));
        etags.push(tag);

        elems.push(C::from_iter([*i1, *i0, e.get(2)]));
        etags.push(tag);
    }

    (elems, etags)
}

pub fn split_tets<C: Simplex>(
    elems_and_tags: impl ExactSizeIterator<Item = (C, Tag)>,
    edges: &FxHashMap<Edge<C::T>, usize>,
) -> (Vec<C>, Vec<Tag>) {
    assert_eq!(C::N_VERTS, 4);

    let new_n_elems = 8 * elems_and_tags.len();
    let mut elems = Vec::with_capacity(new_n_elems);
    let mut etags = Vec::with_capacity(new_n_elems);

    for (e, tag) in elems_and_tags {
        let edg = Edge::new(e.get(0), e.get(1)).sorted();
        let i4 = edges.get(&edg);
        let edg = Edge::new(e.get(1), e.get(2)).sorted();
        let i5 = edges.get(&edg);
        let edg = Edge::new(e.get(2), e.get(0)).sorted();
        let i6 = edges.get(&edg);
        let edg = Edge::new(e.get(0), e.get(3)).sorted();
        let i7 = edges.get(&edg);
        let edg = Edge::new(e.get(1), e.get(3)).sorted();
        let i8 = edges.get(&edg);
        let edg = Edge::new(e.get(2), e.get(3)).sorted();
        let i9 = edges.get(&edg);

        let ids = [
            e.get(0),
            e.get(1),
            e.get(2),
            e.get(3),
            *i4.unwrap(),
            *i5.unwrap(),
            *i6.unwrap(),
            *i7.unwrap(),
            *i8.unwrap(),
            *i9.unwrap(),
        ];

        elems.push(C::from_iter([ids[0], ids[4], ids[6], ids[7]]));
        etags.push(tag);

        elems.push(C::from_iter([ids[4], ids[1], ids[5], ids[8]]));
        etags.push(tag);

        elems.push(C::from_iter([ids[4], ids[5], ids[6], ids[7]]));
        etags.push(tag);

        elems.push(C::from_iter([ids[4], ids[5], ids[7], ids[8]]));
        etags.push(tag);

        elems.push(C::from_iter([ids[8], ids[5], ids[7], ids[9]]));
        etags.push(tag);

        elems.push(C::from_iter([ids[6], ids[5], ids[9], ids[7]]));
        etags.push(tag);

        elems.push(C::from_iter([ids[7], ids[8], ids[9], ids[3]]));
        etags.push(tag);

        elems.push(C::from_iter([ids[6], ids[5], ids[2], ids[9]]));
        etags.push(tag);
    }

    (elems, etags)
}
