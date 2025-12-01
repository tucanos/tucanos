//! Boundary of `Mesh3d`
use crate::{
    Result, Vert3d, Vertex,
    mesh::{GenericMesh, Mesh, QuadraticEdge, QuadraticTriangle, Simplex, Triangle, elements::Idx},
};
use std::{fs::OpenOptions, iter::once};

/// Triangle mesh in 3d
pub type BoundaryMesh3d = GenericMesh<3, Triangle<usize>>;
pub type QuadraticBoundaryMesh3d = GenericMesh<3, QuadraticTriangle<usize>>;

/// Create a `Mesh<3, Triangle<_>>` of a sphere
#[must_use]
pub fn sphere_mesh<M: Mesh<3, Triangle<impl Idx>>>(r: f64, n: usize) -> M {
    let mut res = M::empty();

    res.add_verts(
        [
            Vert3d::new(0., 0., 1.),
            Vert3d::new(1., 0., 0.),
            Vert3d::new(0., 1., 0.),
            Vert3d::new(-1., 0., 0.),
            Vert3d::new(0., -1., 0.),
            Vert3d::new(0., 0., -1.),
        ]
        .iter()
        .copied(),
    );

    res.add_elems_and_tags(
        [
            (Triangle::new(0, 1, 2), 1),
            (Triangle::new(0, 2, 3), 1),
            (Triangle::new(0, 3, 4), 1),
            (Triangle::new(0, 4, 1), 1),
            (Triangle::new(5, 2, 1), 1),
            (Triangle::new(5, 3, 2), 1),
            (Triangle::new(5, 4, 3), 1),
            (Triangle::new(5, 1, 4), 1),
        ]
        .iter()
        .copied(),
    );

    for _ in 0..n {
        res = res.split();
        res.verts_mut().for_each(|x| *x *= r / x.norm());
    }

    res
}

/// Create a `Mesh<3, QuadraticTriangle<_>>` of a sphere
#[must_use]
pub fn quadratic_sphere_mesh<M: Mesh<3, QuadraticTriangle<impl Idx>>>(r: f64, n: usize) -> M {
    let mut res: M = to_quadratic_triangle_mesh(&sphere_mesh::<BoundaryMesh3d>(r, n));
    res.verts_mut().for_each(|x| *x *= r / x.norm());

    res
}

#[must_use]
pub fn to_quadratic_triangle_mesh<
    const D: usize,
    T: Idx,
    M: Mesh<D, QuadraticTriangle<T>>,
    T2: Idx,
>(
    msh: &impl Mesh<D, Triangle<T2>>,
) -> M {
    let edges = msh.edges();

    let mut res = M::empty();
    res.add_verts(msh.verts());
    let mut new_verts = vec![Vertex::zeros(); edges.len()];
    for (&e, &i) in &edges {
        new_verts[i] = 0.5 * (msh.vert(e.get(0)) + msh.vert(e.get(1)));
    }
    res.add_verts(new_verts.iter().copied());

    for (t, tag) in msh.elems().zip(msh.etags()) {
        res.add_elems(
            once(QuadraticTriangle::new(
                t.get(0),
                t.get(1),
                t.get(2),
                edges.get(&t.face(2).sorted()).unwrap() + msh.n_verts(),
                edges.get(&t.face(0).sorted()).unwrap() + msh.n_verts(),
                edges.get(&t.face(1).sorted()).unwrap() + msh.n_verts(),
            )),
            once(tag),
        );
    }

    for (e, tag) in msh.faces().zip(msh.ftags()) {
        res.add_faces(
            once(QuadraticEdge::new(
                e.get(0),
                e.get(1),
                edges.get(&e.sorted()).unwrap() + msh.n_verts(),
            )),
            once(tag),
        );
    }

    res
}

/// Read a stl file
pub fn read_stl<M: Mesh<3, Triangle<impl Idx>>>(file_name: &str) -> Result<M> {
    let mut file = OpenOptions::new().read(true).open(file_name).unwrap();
    let stl = stl_io::read_stl(&mut file).unwrap();

    let mut verts = Vec::with_capacity(stl.vertices.len());
    verts.extend(
        stl.vertices
            .iter()
            .map(|v| Vert3d::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]))),
    );

    let mut elems = Vec::with_capacity(3 * stl.faces.len());
    elems.extend(stl.faces.iter().map(|v| Triangle::from_iter(v.vertices)));
    let etags = vec![1; stl.faces.len()];
    let faces = Vec::new();
    let ftags = Vec::new();

    Ok(M::new(&verts, &elems, &etags, &faces, &ftags))
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use crate::{
        assert_delta,
        mesh::{
            BoundaryMesh3d, GSimplex, Mesh, Mesh3d, QuadraticBoundaryMesh3d, box_mesh,
            quadratic_sphere_mesh, sphere_mesh,
        },
    };
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_box() {
        let msh = box_mesh::<Mesh3d>(1.0, 10, 2.0, 15, 1.0, 20);

        let (mut bdy, _): (BoundaryMesh3d, _) = msh.boundary();

        let faces = bdy.all_faces();
        let tags = bdy.tag_internal_faces(&faces);
        assert_eq!(tags.len(), 12);
        bdy.check(&faces).unwrap();

        let vol = bdy.gelems().map(|ge| ge.vol()).sum::<f64>();
        assert_delta!(vol, 10.0, 1e-12);
    }

    #[test]
    fn test_integrate() {
        let msh = box_mesh::<Mesh3d>(1.0, 10, 2.0, 15, 1.0, 20);

        let f = msh.par_verts().map(|v| v[0]).collect::<Vec<_>>();

        let tag = 3;
        let (bdy, ids): (BoundaryMesh3d, _) = msh.extract_faces(|t| t == tag);
        let f_bdy = ids.iter().map(|&i| f[i]).collect::<Vec<_>>();

        let val = bdy.integrate(&f_bdy, |_| 1.0);
        assert_delta!(val, 1.0, 1e-12);

        let val = bdy.integrate(&f_bdy, |x| x);
        assert_delta!(val, 0.5, 1e-12);

        let nrm = bdy.norm(&f_bdy);
        assert_delta!(nrm, 1.0 / 3.0_f64.sqrt(), 1e-12);
    }

    #[test]
    fn test_sphere() {
        let n = 4;

        let msh: BoundaryMesh3d = sphere_mesh(1.0, n + 1);

        let qmsh: QuadraticBoundaryMesh3d = quadratic_sphere_mesh(1.0, n);

        assert_delta!(msh.vol(), 4.0 * PI, 0.02);
        assert_delta!(qmsh.vol(), 4.0 * PI, 0.00004);

        let d = msh
            .gelems()
            .map(|ge| (ge.center().norm() - 1.0).abs())
            .fold(0.0, f64::max);
        let qd = qmsh
            .gelems()
            .map(|ge| (ge.center().norm() - 1.0).abs())
            .fold(0.0, f64::max);

        assert_delta!(d, 0.0, 0.02);
        assert_delta!(qd, 0.0, 0.000006);

        let qmsh: QuadraticBoundaryMesh3d = quadratic_sphere_mesh(1.0, 1);

        qmsh.write_meshb("qsphere.meshb").unwrap();
    }
}
