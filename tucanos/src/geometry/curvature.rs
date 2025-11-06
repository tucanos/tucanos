use crate::{H_MAX, Result, mesh::SimplexMesh};
use log::debug;
use nalgebra::{SMatrix, SVector, SymmetricEigen};
use tmesh::{
    Vert2d, Vert3d, Vertex,
    mesh::{Edge, GSimplex, Idx, Mesh, Simplex, Triangle},
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
pub fn compute_vertex_normals<T: Idx, const D: usize, C: Simplex<T>>(
    mesh: &SimplexMesh<T, D, C>,
) -> Vec<Vertex<D>> {
    debug!("Compute the surface normals at the vertices");

    let mut normals = vec![Vertex::<D>::zeros(); mesh.n_verts().try_into().unwrap()];

    for e in mesh.elems() {
        let ge = mesh.gelem(&e);
        let n = ge.normal();
        let w = ge.vol();
        for i_vert in e {
            normals[i_vert.try_into().unwrap()] += w * n;
        }
    }

    for i_vert in 0..mesh.n_verts().try_into().unwrap() {
        normals[i_vert].normalize_mut();
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
pub fn compute_curvature_tensor_2d<T: Idx>(smesh: &SimplexMesh<T, 2, Edge<T>>) -> Vec<Vert2d> {
    debug!("Compute curvature tensors (2d)");
    let vertex_normals = compute_vertex_normals(smesh);

    let mut elem_u = Vec::with_capacity(smesh.n_verts().try_into().unwrap());

    for e in smesh.elems() {
        // Compute the element-based curvature
        let n0 = &vertex_normals[e[0].try_into().unwrap()];
        let n1 = &vertex_normals[e[1].try_into().unwrap()];
        let dn = n1 - n0;

        // local basis
        let mut u = smesh.vert(e[1]) - smesh.vert(e[0]);
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
pub fn compute_curvature_tensor<T: Idx>(
    smesh: &SimplexMesh<T, 3, Triangle<T>>,
) -> (Vec<Vert3d>, Vec<Vert3d>) {
    debug!("Compute curvature tensors");
    let vertex_normals = compute_vertex_normals(smesh);

    // // arrays for the least squares problem
    // let nrows = 6;
    // let ncols = 3;

    let mut elem_u = Vec::with_capacity(smesh.n_verts().try_into().unwrap());
    let mut elem_v = Vec::with_capacity(smesh.n_verts().try_into().unwrap());

    for e in smesh.elems() {
        // Compute the element-based curvature
        let ge = smesh.gelem(&e);
        let n = ge.normal();

        let edgs = [ge.edge(0_usize), ge.edge(1_usize), ge.edge(2_usize)];
        let n0 = &vertex_normals[e[0].try_into().unwrap()];
        let n1 = &vertex_normals[e[1].try_into().unwrap()];
        let n2 = &vertex_normals[e[2].try_into().unwrap()];
        let dn = [n1 - n0, n2 - n0, n2 - n1];

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

pub fn fix_curvature<T: Idx, const D: usize, C: Simplex<T>>(
    smesh: &SimplexMesh<T, D, C>,
    u: &mut [Vertex<D>],
    mut v: Option<&mut [Vertex<D>]>,
) -> Result<()> {
    debug!("Fix curvature tensors near the patch boundaries");

    if D == 2 {
        assert!(v.is_none());
    } else {
        assert!(v.is_some());
    }

    let (_, bdy_ids) = smesh.boundary::<SimplexMesh<T, D, C::FACE>>();
    let mut bdy_flag = vec![false; smesh.n_verts().try_into().unwrap()];
    for i in bdy_ids {
        bdy_flag[i.try_into().unwrap()] = true;
    }

    let e2e = smesh.get_elem_to_elems()?;
    let mut flg = Vec::with_capacity(smesh.n_elems().try_into().unwrap());

    let mut to_fix = 0;
    for e in smesh.elems() {
        let is_boundary = e.into_iter().any(|i| bdy_flag[i.try_into().unwrap()]);
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
        for i_elem in 0..smesh.n_elems().try_into().unwrap() {
            if flg[i_elem] {
                // Use any of the valid neighbors
                // TODO: take the closest
                let valid_neighbors = e2e
                    .row(i_elem.try_into().unwrap())
                    .iter()
                    .copied()
                    .filter(|&i| !flg[i.try_into().unwrap()])
                    .filter(|&i| etags[i.try_into().unwrap()] == etags[i_elem]);
                let mut min_dist = f64::MAX;
                let e = smesh.elem(i_elem.try_into().unwrap());
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
                    let i_neighbor = i_neighbor.try_into().unwrap();
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

    Ok(())
}

pub trait HasCurvature<const D: usize> {
    fn compute_curvature(&self) -> (Vec<Vertex<D>>, Option<Vec<Vertex<D>>>);
}

impl<T: Idx> HasCurvature<2> for SimplexMesh<T, 2, Edge<T>> {
    fn compute_curvature(&self) -> (Vec<Vert2d>, Option<Vec<Vert2d>>) {
        let mut u = compute_curvature_tensor_2d(self);
        fix_curvature(self, &mut u, None).unwrap();

        (u, None)
    }
}

impl<T: Idx> HasCurvature<3> for SimplexMesh<T, 3, Triangle<T>> {
    fn compute_curvature(&self) -> (Vec<Vert3d>, Option<Vec<Vert3d>>) {
        let (mut u, mut v) = compute_curvature_tensor(self);
        fix_curvature(self, &mut u, Some(&mut v)).unwrap();
        (u, Some(v))
    }
}

#[cfg(test)]
mod tests {
    use tmesh::{
        Vert2d, Vert3d,
        mesh::{Edge, GSimplex, Mesh, MutMesh},
    };

    use crate::{
        H_MAX, Result,
        mesh::{SimplexMesh, test_meshes::test_mesh_2d},
    };

    use super::{compute_curvature_tensor, compute_curvature_tensor_2d, fix_curvature};

    #[test]
    fn test_curvature_circle() {
        let r_in = 0.1;
        let r_out = 0.5;

        let mesh = test_mesh_2d().split().split().split().split().split();
        let (mut mesh, ids) = mesh.boundary::<SimplexMesh<u32, 2, Edge<u32>>>();

        mesh.verts_mut().for_each(|p| {
            let r = r_in + (r_out - r_in) * p[0];
            let theta = 3.0 * p[1];
            let x = r * f64::cos(theta);
            let y = r * f64::sin(theta);
            *p = Vert2d::new(x, y);
        });

        let u = compute_curvature_tensor_2d(&mesh);

        // Get the indices of corner vertices
        mesh.fix();

        let mut bdy_flag = vec![false; mesh.n_verts().try_into().unwrap()];
        for i in ids {
            bdy_flag[i as usize] = true;
        }

        for (i_elem, (e, t)) in mesh.elems().zip(mesh.etags()).enumerate() {
            let u = u[i_elem];
            let ge = mesh.gelem(&e);
            let c = ge.center();

            let is_boundary = e.into_iter().any(|i| bdy_flag[i as usize]);
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
    fn test_curvature_circle_fixed() -> Result<()> {
        let r_in = 0.1;
        let r_out = 0.5;

        let mesh = test_mesh_2d().split().split().split().split().split();
        let (mut mesh, _) = mesh.boundary::<SimplexMesh<u32, 2, Edge<u32>>>();

        mesh.verts_mut().for_each(|p| {
            let r = r_in + (r_out - r_in) * p[0];
            let theta = 3.0 * p[1];
            let x = r * f64::cos(theta);
            let y = r * f64::sin(theta);
            *p = Vert2d::new(x, y);
        });

        mesh.compute_elem_to_elems();
        mesh.fix();

        let mut u = compute_curvature_tensor_2d(&mesh);
        fix_curvature(&mesh, &mut u, None)?;

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

        Ok(())
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
                Vert3d::new(x, y, z)
            })
            .collect();

        let mut surf = SimplexMesh::new_with_vec(
            verts,
            mesh.elems().collect::<Vec<_>>(),
            mesh.etags().collect::<Vec<_>>(),
            Vec::new(),
            Vec::new(),
        );
        // Get the indices of boundary vertices
        surf.fix();
        let mut bdy_flag = vec![false; surf.n_verts().try_into().unwrap()];
        let (_, ids) = surf.boundary::<SimplexMesh<u32, 3, Edge<u32>>>();
        for i in ids {
            bdy_flag[i as usize] = true;
        }

        let (u, v) = compute_curvature_tensor(&surf);

        for (i_elem, e) in surf.elems().enumerate() {
            let mut u = u[i_elem];
            let mut v = v[i_elem];
            let ge = surf.gelem(&e);
            let c = ge.center();

            let is_boundary = e.into_iter().any(|i| bdy_flag[i as usize]);
            if !is_boundary {
                assert!(u.norm() < 1.1 / H_MAX);
                assert!(u.norm() > 1e-17);
                let u_ref = Vert3d::new(0.0, 0.0, 1.0);
                u.normalize_mut();
                let cos_a = u.dot(&u_ref);
                assert!(f64::abs(cos_a) > 0.98);

                assert!(f64::abs(v.norm() - 1. / radius) < 1e-6 / radius);
                let mut v_ref = Vert3d::new(-c[1], c[0], 0.0);
                v_ref.normalize_mut();
                v.normalize_mut();
                let cos_a = v.dot(&v_ref);
                assert!(f64::abs(cos_a) > 0.98);
            }
        }
    }

    #[test]
    fn test_curvature_cylinder_fixed() -> Result<()> {
        let radius = 0.1;

        let mesh = test_mesh_2d().split().split().split().split().split();

        let verts = mesh
            .verts()
            .map(|p| {
                let z = p[0];
                let theta = 3.0 * p[1];
                let x = radius * f64::cos(theta);
                let y = radius * f64::sin(theta);
                Vert3d::new(x, y, z)
            })
            .collect();

        let mut surf = SimplexMesh::new_with_vec(
            verts,
            mesh.elems().collect::<Vec<_>>(),
            mesh.etags().collect::<Vec<_>>(),
            Vec::new(),
            Vec::new(),
        );
        surf.fix();
        surf.compute_elem_to_elems();

        let (mut u, mut v) = compute_curvature_tensor(&surf);
        fix_curvature(&surf, &mut u, Some(&mut v))?;

        for (i_elem, e) in surf.elems().enumerate() {
            let mut u = u[i_elem];
            let mut v = v[i_elem];
            let ge = surf.gelem(&e);
            let c = ge.center();

            assert!(u.norm() < 1.1 / H_MAX);
            assert!(u.norm() > 1e-17);
            let u_ref = Vert3d::new(0.0, 0.0, 1.0);
            u.normalize_mut();
            let cos_a = u.dot(&u_ref);
            assert!(f64::abs(cos_a) > 0.98);

            assert!(f64::abs(v.norm() - 1. / radius) < 1e-3 / radius);
            let mut v_ref = Vert3d::new(-c[1], c[0], 0.0);
            v_ref.normalize_mut();
            v.normalize_mut();
            let cos_a = v.dot(&v_ref);
            assert!(f64::abs(cos_a) > 0.98);
        }

        Ok(())
    }
}
