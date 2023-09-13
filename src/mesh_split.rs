use crate::{
    mesh::{Point, SimplexMesh},
    topo_elems::Elem,
    Idx, Tag,
};
use log::info;
use rustc_hash::FxHashMap;
use std::hash::BuildHasherDefault;

impl<const D: usize, E: Elem> SimplexMesh<D, E> {
    fn split_edgs<E2: Elem, I: Iterator<Item = (E2, Tag)>>(
        n_elems: Idx,
        elems_and_tags: I,
        edges: &FxHashMap<[Idx; 2], Idx>,
    ) -> (Vec<E2>, Vec<Tag>) {
        let new_n_elems = 2 * n_elems;
        let mut elems = Vec::with_capacity(new_n_elems as usize);
        let mut etags = Vec::with_capacity(new_n_elems as usize);

        for (e, tag) in elems_and_tags {
            let mut edg = [e[0], e[1]];
            edg.sort_unstable();
            let i = edges.get(&edg).unwrap();

            elems.push(E2::from_slice(&[e[0], *i]));
            etags.push(tag);

            elems.push(E2::from_slice(&[*i, e[1]]));
            etags.push(tag);
        }

        (elems, etags)
    }

    fn split_tris<E2: Elem, I: Iterator<Item = (E2, Tag)>>(
        n_elems: Idx,
        elems_and_tags: I,
        edges: &FxHashMap<[Idx; 2], Idx>,
    ) -> (Vec<E2>, Vec<Tag>) {
        let new_n_elems = 4 * n_elems;
        let mut elems = Vec::with_capacity(new_n_elems as usize);
        let mut etags = Vec::with_capacity(new_n_elems as usize);

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

            elems.push(E2::from_slice(&[e[0], *i2, *i1]));
            etags.push(tag);

            elems.push(E2::from_slice(&[*i2, e[1], *i0]));
            etags.push(tag);

            elems.push(E2::from_slice(&[*i2, *i0, *i1]));
            etags.push(tag);

            elems.push(E2::from_slice(&[*i1, *i0, e[2]]));
            etags.push(tag);
        }

        (elems, etags)
    }

    fn split_tets<E2: Elem, I: Iterator<Item = (E2, Tag)>>(
        n_elems: Idx,
        elems_and_tags: I,
        edges: &FxHashMap<[Idx; 2], Idx>,
    ) -> (Vec<E2>, Vec<Tag>) {
        let new_n_elems = 8 * n_elems;
        let mut elems = Vec::with_capacity((E::N_VERTS * new_n_elems) as usize);
        let mut etags = Vec::with_capacity(new_n_elems as usize);

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

            elems.push(E2::from_slice(&[ids[0], ids[4], ids[6], ids[7]]));
            etags.push(tag);

            elems.push(E2::from_slice(&[ids[4], ids[1], ids[5], ids[8]]));
            etags.push(tag);

            elems.push(E2::from_slice(&[ids[4], ids[5], ids[6], ids[7]]));
            etags.push(tag);

            elems.push(E2::from_slice(&[ids[4], ids[5], ids[7], ids[8]]));
            etags.push(tag);

            elems.push(E2::from_slice(&[ids[8], ids[5], ids[7], ids[9]]));
            etags.push(tag);

            elems.push(E2::from_slice(&[ids[6], ids[5], ids[9], ids[7]]));
            etags.push(tag);

            elems.push(E2::from_slice(&[ids[7], ids[8], ids[9], ids[3]]));
            etags.push(tag);

            elems.push(E2::from_slice(&[ids[6], ids[5], ids[2], ids[9]]));
            etags.push(tag);
        }

        (elems, etags)
    }

    /// Create a new mesh by splitting all the mesh elements and faces
    /// TODO: transfer the vertex and cell data
    #[must_use]
    pub fn split(&self) -> Self {
        info!("Split all the elements uniformly");

        let mut edges: FxHashMap<[Idx; 2], Idx> =
            FxHashMap::with_hasher(BuildHasherDefault::default());
        let mut i_edg = self.n_verts() as Idx;
        for e in self.elems() {
            for i in 0..E::N_EDGES {
                let mut edg = e.edge(i);
                edg.sort_unstable();
                let tmp = edges.get(&edg);
                if tmp.is_none() {
                    edges.insert(edg, i_edg);
                    i_edg += 1;
                }
            }
        }

        let new_n_verts = self.n_verts() + edges.len() as Idx;
        let mut verts = Vec::with_capacity(new_n_verts as usize);
        verts.extend(self.verts());
        verts.resize(new_n_verts as usize, Point::<D>::zeros());

        for (edg, &i) in &edges {
            let p0 = self.vert(edg[0]);
            let p1 = self.vert(edg[1]);
            verts[i as usize] = 0.5 * (p0 + p1);
        }

        if E::N_VERTS == 3 {
            let (elems, etags) =
                Self::split_tris(self.n_elems(), self.elems().zip(self.etags()), &edges);
            let (faces, ftags) =
                Self::split_edgs(self.n_faces(), self.faces().zip(self.ftags()), &edges);
            Self::new(verts, elems, etags, faces, ftags)
        } else {
            let (elems, etags) =
                Self::split_tets(self.n_elems(), self.elems().zip(self.etags()), &edges);
            let (faces, ftags) =
                Self::split_tris(self.n_faces(), self.faces().zip(self.ftags()), &edges);
            Self::new(verts, elems, etags, faces, ftags)
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        geom_elems::GElem,
        test_meshes::{test_mesh_2d, test_mesh_3d},
        topo_elems::{get_face_to_elem, Elem},
        Idx,
    };

    #[test]
    fn test_split_2d() {
        let mesh = test_mesh_2d();

        let mesh = mesh.split();

        assert_eq!(mesh.n_verts(), 4 + 5);
        assert_eq!(mesh.n_elems(), 8);
        assert_eq!(mesh.n_faces(), 8);
    }

    #[test]
    fn test_split_3d() {
        let mesh = test_mesh_3d();

        let mesh = mesh.split();

        assert_eq!(mesh.n_verts(), 8 + 6 + 12);
        assert_eq!(mesh.n_faces(), 12 * 4);
        assert_eq!(mesh.n_elems(), 5 * 8);

        let v: f64 = mesh.vol();
        assert!(f64::abs(v - 1.0) < 1e-12);

        let f2e = get_face_to_elem(mesh.elems());
        let mut areas = [0.0; 7];
        for (i_face, (mut f, tag)) in mesh.faces().zip(mesh.ftags()).enumerate() {
            f.sort();
            assert!(f2e.get(&f).is_some());
            areas[tag as usize] += mesh.gface(mesh.face(i_face as Idx)).vol();
        }
        for a in areas.iter().skip(1) {
            assert!((*a - 1.).abs() < 1e-12);
        }
    }
}
