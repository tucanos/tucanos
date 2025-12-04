use crate::H_MAX;
use log::debug;
use nalgebra::{SMatrix, SVector, SymmetricEigen};
use tmesh::{
    Result, Vertex,
    io::VTUFile,
    mesh::{Edge, GSimplex, GenericMesh, Idx, Mesh, Simplex, Triangle},
};

/// Compute the surface normals at the mesh vertices
/// NB: it should not be used across non smooth patches
/// TODO: use a different weighting:
/// "This contrasts with the weights used for
/// estimating normals, for which we take wf,p to be the area of
/// f divided by the squares of the lengths of the two edges that
/// touch vertex p. As shown by Max (1999), this produces more
/// accurate normal estimates than other weighting approaches,
/// and is exact for vertices that lie on a sphere."
#[must_use]
pub fn compute_vertex_normals<const D: usize, M: Mesh<D>>(mesh: &M) -> Vec<Vertex<D>> {
    debug!("Compute the surface normals at the vertices");

    let mut normals = vec![Vertex::<D>::zeros(); mesh.n_verts()];

    for e in mesh.elems() {
        let ge = mesh.gelem(&e);
        let n = ge.normal();
        for i_vert in e {
            normals[i_vert] += n;
        }
    }

    for n in &mut normals {
        n.normalize_mut();
    }

    normals
}

fn bound_curvature(x: f64) -> f64 {
    let b = 1. / H_MAX;
    if x < -b {
        x
    } else if x < 0.0 {
        -b
    } else if x < b {
        b
    } else {
        x
    }
}

/// Compute the curvature of a line mesh and return the principal direction of the
/// curvature tensor (scaled with the principal curvature) for each element
/// (2D version of <https://gfx.cs.princeton.edu/pubs/Rusinkiewicz_2004_ECA/curvpaper.pdf>)
/// NB: curvature should not be computed across non smooth patches
/// NB: curvature estimation is not accurate near the boundaires
#[must_use]
pub fn compute_curvature_tensor_2d<T: Idx>(smesh: &impl Mesh<2, C = Edge<T>>) -> Vec<Vertex<2>> {
    debug!("Compute curvature tensors (2d)");
    let vertex_normals = compute_vertex_normals(smesh);

    let mut elem_u = Vec::with_capacity(smesh.n_verts());

    for e in smesh.elems() {
        // Compute the element-based curvature
        let n0 = &vertex_normals[e.get(0)];
        let n1 = &vertex_normals[e.get(1)];
        let dn = n1 - n0;

        // local basis
        let mut u = smesh.vert(e.get(1)) - smesh.vert(e.get(0));
        let nrm = u.norm();
        let eu = nrm;
        u /= nrm;
        let nu = dn.dot(&u);
        let c = nu / eu;

        u *= bound_curvature(c);

        elem_u.push(u);
    }

    elem_u
}

/// Compute the curvature of a surface mesh and return the principal direction of the
/// curvature tensor (scaled with the principal curvature) for each element
/// the algorithm is taken from <https://gfx.cs.princeton.edu/pubs/Rusinkiewicz_2004_ECA/curvpaper.pdf>
/// NB: curvature should not be computed across non smooth patches
/// NB: curvature estimation is not accurate near the boundaires
#[must_use]
pub fn compute_curvature_tensor_3d<T: Idx>(
    smesh: &impl Mesh<3, C = Triangle<T>>,
) -> (Vec<Vertex<3>>, Vec<Vertex<3>>) {
    debug!("Compute curvature tensors");
    let vertex_normals = compute_vertex_normals(smesh);

    let mut elem_u = Vec::with_capacity(smesh.n_verts());
    let mut elem_v = Vec::with_capacity(smesh.n_verts());

    for e in smesh.elems() {
        // Compute the element-based curvature
        let ge = smesh.gelem(&e);
        let n = ge.normal().normalize();

        let edgs = [
            ge.face(0).as_vec(),
            ge.face(1).as_vec(),
            ge.face(2).as_vec(),
        ];
        let n0 = &vertex_normals[e.get(0)];
        let n1 = &vertex_normals[e.get(1)];
        let n2 = &vertex_normals[e.get(2)];
        let dn = [n2 - n1, n0 - n2, n1 - n0];

        // local basis
        let mut u = edgs[0];
        u.normalize_mut();
        let mut v = n.cross(&u);

        let mut mat = SMatrix::<f64, 6, 3>::zeros();
        let mut rhs = SMatrix::<f64, 6, 1>::zeros();

        for i in 0..3 {
            let eu = edgs[i].dot(&u);
            let ev = edgs[i].dot(&v);
            let nu = dn[i].dot(&u);
            let nv = dn[i].dot(&v);

            let row = SMatrix::<f64, 1, 3>::new(eu, ev, 0.0);
            let irow = 2 * i;
            mat.set_row(irow, &row);
            rhs[irow] = nu;

            let row = SMatrix::<f64, 1, 3>::new(0.0, eu, ev);
            let irow = 2 * i + 1;
            mat.set_row(irow, &row);
            rhs[irow] = nv;
        }

        let qr = mat.qr();
        qr.q_tr_mul(&mut rhs);
        let mut rhs = SVector::<f64, 3>::new(rhs[0], rhs[1], rhs[2]);
        assert!(qr.r().solve_upper_triangular_mut(&mut rhs));

        // Compute the principal directions
        let mat = SMatrix::<f64, 2, 2>::new(rhs[0], rhs[1], rhs[1], rhs[2]);
        let eig = SymmetricEigen::new(mat);

        // Make sure that the pricipal curvatures are non-zero
        let ev0 = if eig.eigenvalues[0].abs() < 1e-16 {
            1e-16
        } else {
            eig.eigenvalues[0]
        };
        let ev1 = if eig.eigenvalues[1].abs() < 1e-16 {
            1e-16
        } else {
            eig.eigenvalues[1]
        };

        if ev0 > ev1 {
            u = eig.eigenvectors[0] * u + eig.eigenvectors[1] * v;
            v = n.cross(&u);
            u.normalize_mut();
            v.normalize_mut();

            u *= bound_curvature(ev0);
            v *= bound_curvature(ev1);
        } else {
            v = eig.eigenvectors[0] * u + eig.eigenvectors[1] * v;
            u = -n.cross(&v);

            v.normalize_mut();
            u.normalize_mut();

            u *= bound_curvature(ev1);
            v *= bound_curvature(ev0);
        }

        elem_u.push(u);
        elem_v.push(v);
    }

    (elem_u, elem_v)
}

pub fn fix_curvature<const D: usize, M: Mesh<D>>(
    smesh: &M,
    u: &mut [Vertex<D>],
    mut v: Option<&mut [Vertex<D>]>,
) {
    debug!("Fix curvature tensors near the patch boundaries");

    if D == 2 {
        assert!(v.is_none());
    } else {
        assert!(v.is_some());
    }

    let (_, bdy_ids) = smesh.boundary::<GenericMesh<D, <M::C as Simplex>::FACE>>();
    let mut bdy_flag = vec![false; smesh.n_verts()];
    for i in bdy_ids {
        bdy_flag[i] = true;
    }

    let e2e = smesh.element_pairs(&smesh.all_faces());
    let mut flg = Vec::with_capacity(smesh.n_elems());

    let mut to_fix = 0;
    for e in smesh.elems() {
        let is_boundary = e.into_iter().any(|i| bdy_flag[i]);
        flg.push(is_boundary);
        if is_boundary {
            to_fix += 1;
        }
    }

    let mut n_iter = 0;
    let etags: Vec<_> = smesh.etags().collect();
    while to_fix > 0 {
        debug!("iteration {}: {} elements to fix", n_iter + 1, to_fix);
        let mut fixed = Vec::with_capacity(to_fix);
        for i_elem in 0..smesh.n_elems() {
            if flg[i_elem] {
                // Use any of the valid neighbors
                // TODO: take the closest
                let valid_neighbors = e2e
                    .row(i_elem)
                    .iter()
                    .copied()
                    .filter(|i| !flg[*i])
                    .filter(|i| etags[*i] == etags[i_elem]);
                let mut min_dist = f64::MAX;
                let e = smesh.elem(i_elem);
                let c = smesh.gelem(&e).center();
                let mut i_neighbor = None;
                for i in valid_neighbors {
                    let e2 = smesh.elem(i);
                    let c2 = smesh.gelem(&e2).center();
                    let dist = (c2 - c).norm();
                    if dist < min_dist {
                        min_dist = dist;
                        i_neighbor = Some(i);
                    }
                }

                if let Some(i_neighbor) = i_neighbor {
                    if D == 2 {
                        u[i_elem].normalize_mut();
                        u[i_elem] *= u[i_neighbor].norm();
                    } else {
                        let v = v.as_mut().unwrap();

                        // Make sure that u,v are in the element plane
                        let mut n = u[i_elem].cross(&v[i_elem]);
                        n.normalize_mut();

                        let mut v_new = v[i_neighbor];
                        v_new.normalize_mut();
                        let mut u_new = v_new.cross(&n);
                        u_new.normalize_mut();
                        v_new = n.cross(&u_new);
                        v_new.normalize_mut();

                        u[i_elem] = u_new * u[i_neighbor].norm();
                        v[i_elem] = v_new * v[i_neighbor].norm();

                        to_fix -= 1;
                        fixed.push(i_elem);
                    }
                }
            }
        }
        if fixed.is_empty() {
            // No element was fixed
            if to_fix > 0 {
                debug!(
                    "stop at iteration {}, {} elements cannot be fixed",
                    n_iter + 1,
                    to_fix
                );
            }
            break;
        }
        for i in fixed {
            flg[i] = false;
        }
        n_iter += 1;
    }
}

pub trait HasCurvature<const D: usize>: Mesh<D> {
    fn compute_curvature(&self) -> (Vec<Vertex<D>>, Option<Vec<Vertex<D>>>);

    fn write_curvature(&self, fname: &str) -> Result<()> {
        let (u, v) = self.compute_curvature();

        let mut writer = VTUFile::from_mesh(self, tmesh::io::VTUEncoding::Binary);
        writer.add_point_data("u", D, u.iter().flatten().copied());
        writer.add_point_data("v", D, v.as_ref().unwrap().iter().flatten().copied());
        writer.export(fname)?;

        Ok(())
    }
}

impl<T: Idx, M: Mesh<2, C = Edge<T>>> HasCurvature<2> for M {
    fn compute_curvature(&self) -> (Vec<Vertex<2>>, Option<Vec<Vertex<2>>>) {
        let mut u = compute_curvature_tensor_2d(self);
        fix_curvature(self, &mut u, None);

        (u, None)
    }
}

impl<T: Idx, M: Mesh<3, C = Triangle<T>>> HasCurvature<3> for M {
    fn compute_curvature(&self) -> (Vec<Vertex<3>>, Option<Vec<Vertex<3>>>) {
        let (mut u, mut v) = compute_curvature_tensor_3d(self);
        fix_curvature(self, &mut u, Some(&mut v));
        (u, Some(v))
    }
}

#[cfg(test)]
mod tests {
    use tmesh::{
        Vertex,
        mesh::{BoundaryMesh2d, Edge, GSimplex, GenericMesh, Mesh, Node},
    };

    use crate::{
        H_MAX,
        geometry::MeshedGeometry,
        mesh::{MeshTopology, test_meshes::test_mesh_2d},
    };

    use super::{compute_curvature_tensor_2d, compute_curvature_tensor_3d, fix_curvature};

    #[test]
    fn test_curvature_circle() {
        let r_in = 0.1;
        let r_out = 0.5;

        let mesh = test_mesh_2d().split().split().split().split().split();
        let (mut mesh, _) = mesh.boundary::<BoundaryMesh2d>();

        mesh.verts_mut().for_each(|p| {
            let r = r_in + (r_out - r_in) * p[0];
            let theta = 3.0 * p[1];
            let x = r * f64::cos(theta);
            let y = r * f64::sin(theta);
            *p = Vertex::<2>::new(x, y);
        });

        let u = compute_curvature_tensor_2d(&mesh);

        // Get the indices of corner vertices
        mesh.fix().unwrap();

        let mut bdy_flag = vec![false; mesh.n_verts()];
        for i in mesh.boundary::<GenericMesh<2, Node<usize>>>().1 {
            bdy_flag[i] = true;
        }

        for (i_elem, (e, t)) in mesh.elems().zip(mesh.etags()).enumerate() {
            let u = u[i_elem];
            let ge = mesh.gelem(&e);
            let c = ge.center();

            let is_boundary = e.into_iter().any(|i| bdy_flag[i]);
            if !is_boundary {
                match t {
                    1 | 3 => {
                        assert!(u.norm() < 1.1 / H_MAX);
                        assert!(u.norm() > 1e-17);
                    }
                    2 => {
                        assert!(f64::abs(u.norm() - 1. / r_out) < 1e-6 / r_out);
                        assert!(f64::abs(c.dot(&u)) < 1e-2);
                    }
                    4 => {
                        assert!(f64::abs(u.norm() - 1. / r_in) < 1e-6 / r_in);
                        assert!(f64::abs(c.dot(&u)) < 1e-2);
                    }
                    _ => {
                        unreachable!();
                    }
                }
            }
        }
    }

    #[test]
    fn test_curvature_circle_fixed() {
        let r_in = 0.1;
        let r_out = 0.5;

        let mesh = test_mesh_2d().split().split().split().split().split();
        let (mut mesh, _) = mesh.boundary::<BoundaryMesh2d>();

        mesh.verts_mut().for_each(|p| {
            let r = r_in + (r_out - r_in) * p[0];
            let theta = 3.0 * p[1];
            let x = r * f64::cos(theta);
            let y = r * f64::sin(theta);
            *p = Vertex::<2>::new(x, y);
        });

        mesh.fix().unwrap();

        let mut u = compute_curvature_tensor_2d(&mesh);
        fix_curvature(&mesh, &mut u, None);

        for (i_elem, (e, t)) in mesh.elems().zip(mesh.etags()).enumerate() {
            let u = u[i_elem];
            let ge = mesh.gelem(&e);
            let c = ge.center();

            match t {
                1 | 3 => {
                    assert!(u.norm() < 1.1 / H_MAX);
                    assert!(u.norm() > 1e-17);
                }
                2 => {
                    assert!(f64::abs(u.norm() - 1. / r_out) < 1e-6 / r_out);
                    assert!(f64::abs(c.dot(&u)) < 1e-2);
                }
                4 => {
                    assert!(f64::abs(u.norm() - 1. / r_in) < 1e-6 / r_in);
                    assert!(f64::abs(c.dot(&u)) < 1e-2);
                }
                _ => {
                    unreachable!();
                }
            }
        }
    }

    #[test]
    fn test_curvature_cylinder() {
        let radius = 0.1;

        let mesh = test_mesh_2d().split().split().split().split().split();

        let verts = mesh
            .verts()
            .map(|p| {
                let z = p[0];
                let theta = 3.0 * p[1];
                let x = radius * f64::cos(theta);
                let y = radius * f64::sin(theta);
                Vertex::<3>::new(x, y, z)
            })
            .collect();

        let mut surf = GenericMesh::from_vecs(
            verts,
            mesh.elems().collect::<Vec<_>>(),
            mesh.etags().collect::<Vec<_>>(),
            Vec::new(),
            Vec::new(),
        );
        // Get the indices of boundary vertices
        surf.fix().unwrap();
        let mut bdy_flag = vec![false; surf.n_verts()];
        for i in surf.boundary::<GenericMesh<3, Edge<usize>>>().1 {
            bdy_flag[i] = true;
        }

        let (u, v) = compute_curvature_tensor_3d(&surf);

        for (i_elem, e) in surf.elems().enumerate() {
            let mut u = u[i_elem];
            let mut v = v[i_elem];
            let ge = surf.gelem(&e);
            let c = ge.center();

            let is_boundary = e.into_iter().any(|i| bdy_flag[i]);
            if !is_boundary {
                assert!(u.norm() < 1.1 / H_MAX);
                assert!(u.norm() > 1e-17);
                let u_ref = Vertex::<3>::new(0.0, 0.0, 1.0);
                u.normalize_mut();
                let cos_a = u.dot(&u_ref);
                assert!(f64::abs(cos_a) > 0.98);

                assert!(f64::abs(v.norm() - 1. / radius) < 1e-6 / radius);
                let mut v_ref = Vertex::<3>::new(-c[1], c[0], 0.0);
                v_ref.normalize_mut();
                v.normalize_mut();
                let cos_a = v.dot(&v_ref);
                assert!(f64::abs(cos_a) > 0.98);
            }
        }
    }

    #[test]
    fn test_curvature_cylinder_fixed() {
        let radius = 0.1;

        let mesh = test_mesh_2d().split().split().split().split().split();

        let verts = mesh
            .verts()
            .map(|p| {
                let z = p[0];
                let theta = 3.0 * p[1];
                let x = radius * f64::cos(theta);
                let y = radius * f64::sin(theta);
                Vertex::<3>::new(x, y, z)
            })
            .collect();

        let mut surf = GenericMesh::from_vecs(
            verts,
            mesh.elems().collect::<Vec<_>>(),
            mesh.etags().collect::<Vec<_>>(),
            Vec::new(),
            Vec::new(),
        );
        surf.fix().unwrap();

        let (mut u, mut v) = compute_curvature_tensor_3d(&surf);
        fix_curvature(&surf, &mut u, Some(&mut v));

        for (i_elem, e) in surf.elems().enumerate() {
            let mut u = u[i_elem];
            let mut v = v[i_elem];
            let ge = surf.gelem(&e);
            let c = ge.center();

            assert!(u.norm() < 1.1 / H_MAX);
            assert!(u.norm() > 1e-17);
            let u_ref = Vertex::<3>::new(0.0, 0.0, 1.0);
            u.normalize_mut();
            let cos_a = u.dot(&u_ref);
            assert!(f64::abs(cos_a) > 0.98);

            assert!(f64::abs(v.norm() - 1. / radius) < 1e-3 / radius);
            let mut v_ref = Vertex::<3>::new(-c[1], c[0], 0.0);
            v_ref.normalize_mut();
            v.normalize_mut();
            let cos_a = v.dot(&v_ref);
            assert!(f64::abs(cos_a) > 0.98);
        }
    }

    #[test]
    fn test_curvature_cylinder_fixed_geom() {
        let radius = 0.1;

        let mesh = test_mesh_2d().split().split().split().split().split();

        let verts = mesh
            .verts()
            .map(|p| {
                let z = p[0];
                let theta = 2.0 * p[1];
                let x = radius * f64::cos(theta);
                let y = radius * f64::sin(theta);
                Vertex::<3>::new(x, y, z)
            })
            .collect();

        let mut surf = GenericMesh::from_vecs(
            verts,
            mesh.elems().collect::<Vec<_>>(),
            mesh.etags().collect::<Vec<_>>(),
            Vec::new(),
            Vec::new(),
        );
        surf.fix().unwrap();

        let geom = MeshedGeometry::new(&surf, &MeshTopology::new(&surf), surf.clone()).unwrap();

        for (e, tag) in surf.elems().zip(surf.etags()) {
            let ge = surf.gelem(&e);
            let (mut u, v) = geom.curvature(&ge.center(), tag);
            let mut v = v.unwrap();

            assert!(u.norm() < 1.1 / H_MAX);
            assert!(u.norm() > 1e-17);
            let u_ref = Vertex::<3>::new(0.0, 0.0, 1.0);
            u.normalize_mut();

            let cos_a = u.dot(&u_ref);
            assert!(f64::abs(cos_a) > 0.98, "{cos_a}");

            assert!(f64::abs(v.norm() - 1. / radius) < 1e-3 / radius);
            let c = ge.center();
            let mut v_ref = Vertex::<3>::new(-c[1], c[0], 0.0);
            v_ref.normalize_mut();
            v.normalize_mut();
            let cos_a = v.dot(&v_ref);

            assert!(f64::abs(cos_a) > 0.98, "{cos_a}");
        }
    }
}
