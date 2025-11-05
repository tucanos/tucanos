use super::Edge;
use crate::{
    Tag,
    mesh::{Idx, Simplex, Tetrahedron, Triangle},
};
use rustc_hash::FxHashMap;

pub fn split_edgs<T: Idx, C: Simplex<T>, I: ExactSizeIterator<Item = (C, Tag)>>(
    elems_and_tags: I,
    edges: &FxHashMap<Edge<T>, T>,
) -> (Vec<C>, Vec<Tag>) {
    assert_eq!(C::N_VERTS, 2);
    let new_n_elems = 2 * elems_and_tags.len();
    let mut elems = Vec::with_capacity(new_n_elems);
    let mut etags = Vec::with_capacity(new_n_elems);

    for (e, tag) in elems_and_tags {
        let edg = Edge([e[0], e[1]]).sorted();
        let i = edges.get(&edg).unwrap();

        elems.push(C::from_other(Edge([e[0], *i])));
        etags.push(tag);

        elems.push(C::from_other(Edge([*i, e[1]])));
        etags.push(tag);
    }

    (elems, etags)
}

pub fn split_tris<T: Idx, C: Simplex<T>, I: ExactSizeIterator<Item = (C, Tag)>>(
    elems_and_tags: I,
    edges: &FxHashMap<Edge<T>, T>,
) -> (Vec<C>, Vec<Tag>) {
    assert_eq!(C::N_VERTS, 3);

    let new_n_elems = 4 * elems_and_tags.len();
    let mut elems = Vec::with_capacity(new_n_elems);
    let mut etags = Vec::with_capacity(new_n_elems);

    for (e, tag) in elems_and_tags {
        let edg = Edge([e[0], e[1]]).sorted();
        let i2 = edges.get(&edg).unwrap();

        let edg = Edge([e[1], e[2]]).sorted();
        let i0 = edges.get(&edg).unwrap();

        let edg = Edge([e[2], e[0]]).sorted();
        let i1 = edges.get(&edg).unwrap();

        elems.push(C::from_other(Triangle([e[0], *i2, *i1])));
        etags.push(tag);

        elems.push(C::from_other(Triangle([*i2, e[1], *i0])));
        etags.push(tag);

        elems.push(C::from_other(Triangle([*i2, *i0, *i1])));
        etags.push(tag);

        elems.push(C::from_other(Triangle([*i1, *i0, e[2]])));
        etags.push(tag);
    }

    (elems, etags)
}

pub fn split_tets<T: Idx, C: Simplex<T>, I: ExactSizeIterator<Item = (C, Tag)>>(
    elems_and_tags: I,
    edges: &FxHashMap<Edge<T>, T>,
) -> (Vec<C>, Vec<Tag>) {
    assert_eq!(C::N_VERTS, 4);

    let new_n_elems = 8 * elems_and_tags.len();
    let mut elems = Vec::with_capacity(new_n_elems);
    let mut etags = Vec::with_capacity(new_n_elems);

    for (e, tag) in elems_and_tags {
        let edg = Edge([e[0], e[1]]).sorted();
        let i4 = edges.get(&edg);
        let edg = Edge([e[1], e[2]]).sorted();
        let i5 = edges.get(&edg);
        let edg = Edge([e[2], e[0]]).sorted();
        let i6 = edges.get(&edg);
        let edg = Edge([e[0], e[3]]).sorted();
        let i7 = edges.get(&edg);
        let edg = Edge([e[1], e[3]]).sorted();
        let i8 = edges.get(&edg);
        let edg = Edge([e[2], e[3]]).sorted();
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

        elems.push(C::from_other(Tetrahedron([ids[0], ids[4], ids[6], ids[7]])));
        etags.push(tag);

        elems.push(C::from_other(Tetrahedron([ids[4], ids[1], ids[5], ids[8]])));
        etags.push(tag);

        elems.push(C::from_other(Tetrahedron([ids[4], ids[5], ids[6], ids[7]])));
        etags.push(tag);

        elems.push(C::from_other(Tetrahedron([ids[4], ids[5], ids[7], ids[8]])));
        etags.push(tag);

        elems.push(C::from_other(Tetrahedron([ids[8], ids[5], ids[7], ids[9]])));
        etags.push(tag);

        elems.push(C::from_other(Tetrahedron([ids[6], ids[5], ids[9], ids[7]])));
        etags.push(tag);

        elems.push(C::from_other(Tetrahedron([ids[7], ids[8], ids[9], ids[3]])));
        etags.push(tag);

        elems.push(C::from_other(Tetrahedron([ids[6], ids[5], ids[2], ids[9]])));
        etags.push(tag);
    }

    (elems, etags)
}
