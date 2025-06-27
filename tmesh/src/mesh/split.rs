use super::{Cell, Edge};
use crate::Tag;
use rustc_hash::FxHashMap;

pub fn split_edgs<const C: usize, I: ExactSizeIterator<Item = (Cell<C>, Tag)>>(
    elems_and_tags: I,
    edges: &FxHashMap<Edge, usize>,
) -> (Vec<Cell<C>>, Vec<Tag>) {
    assert_eq!(C, 2);
    let new_n_elems = 2 * elems_and_tags.len();
    let mut elems = Vec::with_capacity(new_n_elems);
    let mut etags = Vec::with_capacity(new_n_elems);

    let mut new = [0; C];
    for (e, tag) in elems_and_tags {
        let mut edg = [e[0], e[1]];
        edg.sort_unstable();
        let i = edges.get(&edg).unwrap();

        new.copy_from_slice(&[e[0], *i]);
        elems.push(new);
        etags.push(tag);

        new.copy_from_slice(&[*i, e[1]]);
        elems.push(new);
        etags.push(tag);
    }

    (elems, etags)
}

pub fn split_tris<const C: usize, I: ExactSizeIterator<Item = (Cell<C>, Tag)>>(
    elems_and_tags: I,
    edges: &FxHashMap<Edge, usize>,
) -> (Vec<Cell<C>>, Vec<Tag>) {
    assert_eq!(C, 3);

    let new_n_elems = 4 * elems_and_tags.len();
    let mut elems = Vec::with_capacity(new_n_elems);
    let mut etags = Vec::with_capacity(new_n_elems);

    let mut new = [0; C];
    for (e, tag) in elems_and_tags {
        let mut edg = [e[0], e[1]];
        edg.sort_unstable();
        let i2 = edges.get(&edg).unwrap();
        let mut edg = [e[1], e[2]];
        edg.sort_unstable();
        let i0 = edges.get(&edg).unwrap();
        let mut edg = [e[2], e[0]];
        edg.sort_unstable();
        let i1 = edges.get(&edg).unwrap();

        new.copy_from_slice(&[e[0], *i2, *i1]);
        elems.push(new);
        etags.push(tag);

        new.copy_from_slice(&[*i2, e[1], *i0]);
        elems.push(new);
        etags.push(tag);

        new.copy_from_slice(&[*i2, *i0, *i1]);
        elems.push(new);
        etags.push(tag);

        new.copy_from_slice(&[*i1, *i0, e[2]]);
        elems.push(new);
        etags.push(tag);
    }

    (elems, etags)
}

pub fn split_tets<const C: usize, I: ExactSizeIterator<Item = (Cell<C>, Tag)>>(
    elems_and_tags: I,
    edges: &FxHashMap<Edge, usize>,
) -> (Vec<Cell<C>>, Vec<Tag>) {
    assert_eq!(C, 4);

    let new_n_elems = 8 * elems_and_tags.len();
    let mut elems = Vec::with_capacity(new_n_elems);
    let mut etags = Vec::with_capacity(new_n_elems);

    let mut new = [0; C];
    for (e, tag) in elems_and_tags {
        let mut edg = [e[0], e[1]];
        edg.sort_unstable();
        let i4 = edges.get(&edg);
        let mut edg = [e[1], e[2]];
        edg.sort_unstable();
        let i5 = edges.get(&edg);
        let mut edg = [e[2], e[0]];
        edg.sort_unstable();
        let i6 = edges.get(&edg);
        let mut edg = [e[0], e[3]];
        edg.sort_unstable();
        let i7 = edges.get(&edg);
        let mut edg = [e[1], e[3]];
        edg.sort_unstable();
        let i8 = edges.get(&edg);
        let mut edg = [e[2], e[3]];
        edg.sort_unstable();
        let i9 = edges.get(&edg);

        let ids = [
            e[0],
            e[1],
            e[2],
            e[3],
            *i4.unwrap(),
            *i5.unwrap(),
            *i6.unwrap(),
            *i7.unwrap(),
            *i8.unwrap(),
            *i9.unwrap(),
        ];

        new.copy_from_slice(&[ids[0], ids[4], ids[6], ids[7]]);
        elems.push(new);
        etags.push(tag);

        new.copy_from_slice(&[ids[4], ids[1], ids[5], ids[8]]);
        elems.push(new);
        etags.push(tag);

        new.copy_from_slice(&[ids[4], ids[5], ids[6], ids[7]]);
        elems.push(new);
        etags.push(tag);

        new.copy_from_slice(&[ids[4], ids[5], ids[7], ids[8]]);
        elems.push(new);
        etags.push(tag);

        new.copy_from_slice(&[ids[8], ids[5], ids[7], ids[9]]);
        elems.push(new);
        etags.push(tag);

        new.copy_from_slice(&[ids[6], ids[5], ids[9], ids[7]]);
        elems.push(new);
        etags.push(tag);

        new.copy_from_slice(&[ids[7], ids[8], ids[9], ids[3]]);
        elems.push(new);
        etags.push(tag);

        new.copy_from_slice(&[ids[6], ids[5], ids[2], ids[9]]);
        elems.push(new);
        etags.push(tag);
    }

    (elems, etags)
}
