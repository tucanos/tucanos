use crate::{
    geom_elems::GElem,
    linalg::lapack_qr_least_squares,
    mesh::{Point, SimplexMesh},
    topo_elems::{Edge, Elem, Triangle},
    Idx, Mesh, Result, H_MAX,
};
use log::{debug, warn};
use nalgebra::{SMatrix, SymmetricEigen};

/// Compute the surface normals at the mesh vertices
/// NB: it should not be used across non smooth patches
/// TODO: use a different weighting:
/// "This contrasts with the weights used for
/// estimating normals, for which we take wf,p to be the area of
/// f divided by the squares of the lengths of the two edges that
/// touch vertex p. As shown by Max (1999), this produces more
/// accurate normal estimates than other weighting approaches,
/// and is exact for vertices that lie on a sphere."
pub fn compute_vertex_normals<const D: usize, E: Elem>(mesh: &SimplexMesh<D, E>) -> Vec<Point<D>> {
    debug!("Compute the surface normals at the vertices");

    let mut normals = vec![Point::<D>::zeros(); mesh.n_verts() as usize];

    for e in mesh.elems() {
        let ge = mesh.gelem(e);
        let n = ge.normal();
        let w = ge.vol();
        for i_vert in e.into_iter() {
            normals[i_vert as usize] += w * n;
        }
    }

    for i_vert in 0..mesh.n_verts() {
        normals[i_vert as usize].normalize_mut();
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
pub fn compute_curvature_tensor_2d(smesh: &SimplexMesh<2, Edge>) -> Vec<Point<2>> {
    debug!("Compute curvature tensors (2d)");
    let vertex_normals = compute_vertex_normals(smesh);

    let mut elem_u = Vec::with_capacity(smesh.n_verts() as usize);

    for e in smesh.elems() {
        // Compute the element-based curvature
        let n0 = &vertex_normals[e[0] as usize];
        let n1 = &vertex_normals[e[1] as usize];
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
pub fn compute_curvature_tensor(
    smesh: &SimplexMesh<3, Triangle>,
) -> (Vec<Point<3>>, Vec<Point<3>>) {
    debug!("Compute curvature tensors");
    let vertex_normals = compute_vertex_normals(smesh);

    // arrays for the least squares problem
    let nrows = 6;
    let ncols = 3;
    let mut buf = vec![0.; nrows * (ncols + 3)];
    let (a, b) = buf.split_at_mut(nrows * ncols);
    let (b, work) = b.split_at_mut(nrows);

    let mut elem_u = Vec::with_capacity(smesh.n_verts() as usize);
    let mut elem_v = Vec::with_capacity(smesh.n_verts() as usize);

    for e in smesh.elems() {
        // Compute the element-based curvature
        let ge = smesh.gelem(e);
        let n = ge.normal();

        let edgs = [ge.edge(0), ge.edge(1), ge.edge(2)];
        let n0 = &vertex_normals[e[0] as usize];
        let n1 = &vertex_normals[e[1] as usize];
        let n2 = &vertex_normals[e[2] as usize];
        let dn = [n1 - n0, n2 - n0, n2 - n1];

        // local basis
        let mut u = edgs[0];
        u.normalize_mut();
        let mut v = n.cross(&u);

        for i in 0..3 {
            let eu = edgs[i].dot(&u);
            let ev = edgs[i].dot(&v);
            let nu = dn[i].dot(&u);
            let nv = dn[i].dot(&v);
            let irow = 2 * i;
            let row = &mut a[irow..];
            row[0] = eu;
            row[nrows] = ev;
            row[2 * nrows] = 0.;
            b[irow] = nu;
            let irow = 2 * i + 1;
            let row = &mut a[irow..];
            row[0] = 0.0;
            row[nrows] = eu;
            row[2 * nrows] = ev;
            b[irow] = nv;
        }

        let res: [f64; 3] = lapack_qr_least_squares(a, b, work).unwrap();

        // Compute the principal directions
        let mat = SMatrix::<f64, 2, 2>::new(res[0], res[1], res[1], res[2]);
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

pub fn fix_curvature<const D: usize, E: Elem>(
    smesh: &SimplexMesh<D, E>,
    u: &mut [Point<D>],
    mut v: Option<&mut [Point<D>]>,
) -> Result<()> {
    debug!("Fix curvature tensors near the patch boundaries");

    if D == 2 {
        assert!(v.is_none());
    } else {
        assert!(v.is_some());
    }

    let (_, bdy_ids) = smesh.boundary();
    let mut bdy_flag = vec![false; smesh.n_verts() as usize];
    bdy_ids
        .into_iter()
        .for_each(|i| bdy_flag[i as usize] = true);

    let e2e = smesh.get_elem_to_elems()?;
    let mut flg = Vec::with_capacity(smesh.n_elems() as usize);

    let mut to_fix = 0;
    for e in smesh.elems() {
        let is_boundary = e.iter().copied().any(|i| bdy_flag[i as usize]);
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
        for i_elem in 0..smesh.n_elems() as usize {
            if flg[i_elem] {
                // Use any of the valid neighbors
                // TODO: take the closest
                let valid_neighbors = e2e
                    .row(i_elem as Idx)
                    .iter()
                    .copied()
                    .filter(|i| !flg[*i as usize])
                    .filter(|i| etags[*i as usize] == etags[i_elem]);
                let mut min_dist = f64::MAX;
                let e = smesh.elem(i_elem as Idx);
                let c = smesh.gelem(e).center();
                let mut i_neighbor = None;
                for i in valid_neighbors {
                    let e2 = smesh.elem(i);
                    let c2 = smesh.gelem(e2).center();
                    let dist = (c2 - c).norm();
                    if dist < min_dist {
                        min_dist = dist;
                        i_neighbor = Some(i);
                    }
                }

                if let Some(i_neighbor) = i_neighbor {
                    if D == 2 {
                        u[i_elem].normalize_mut();
                        u[i_elem] *= u[i_neighbor as usize].norm();
                    } else {
                        let v = v.as_mut().unwrap();

                        // Make sure that u,v are in the element plane
                        let mut n = u[i_elem].cross(&v[i_elem]);
                        n.normalize_mut();

                        let mut v_new = v[i_neighbor as usize];
                        v_new.normalize_mut();
                        let mut u_new = v_new.cross(&n);
                        u_new.normalize_mut();
                        v_new = n.cross(&u_new);
                        v_new.normalize_mut();

                        u[i_elem] = u_new * u[i_neighbor as usize].norm();
                        v[i_elem] = v_new * v[i_neighbor as usize].norm();

                        to_fix -= 1;
                        fixed.push(i_elem);
                    }
                }
            }
        }
        if fixed.is_empty() {
            // No element was fixed
            if to_fix > 0 {
                warn!(
                    "stop at iteration {}, {} elements cannot be fixed",
                    n_iter + 1,
                    to_fix
                );
            }
            break;
        }
        fixed.iter().for_each(|&i| flg[i] = false);
        n_iter += 1;
    }

    Ok(())
}

pub trait HasCurvature<const D: usize> {
    fn compute_curvature(&self) -> (Vec<Point<D>>, Option<Vec<Point<D>>>);
}

impl HasCurvature<2> for SimplexMesh<2, Edge> {
    fn compute_curvature(&self) -> (Vec<Point<2>>, Option<Vec<Point<2>>>) {
        let mut u = compute_curvature_tensor_2d(self);
        fix_curvature(self, &mut u, None).unwrap();

        (u, None)
    }
}

impl HasCurvature<3> for SimplexMesh<3, Triangle> {
    fn compute_curvature(&self) -> (Vec<Point<3>>, Option<Vec<Point<3>>>) {
        let (mut u, mut v) = compute_curvature_tensor(self);
        fix_curvature(self, &mut u, Some(&mut v)).unwrap();
        (u, Some(v))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        geom_elems::GElem,
        mesh::{Point, SimplexMesh},
        test_meshes::test_mesh_2d,
        topo_elems::Triangle,
        Mesh, Result, H_MAX,
    };

    use super::{compute_curvature_tensor, compute_curvature_tensor_2d, fix_curvature};

    #[test]
    fn test_curvature_circle() {
        let r_in = 0.1;
        let r_out = 0.5;

        let mesh = test_mesh_2d().split().split().split().split().split();
        let (mut mesh, _) = mesh.boundary();

        mesh.mut_verts().for_each(|p| {
            let r = r_in + (r_out - r_in) * p[0];
            let theta = 3.0 * p[1];
            let x = r * f64::cos(theta);
            let y = r * f64::sin(theta);
            *p = Point::<2>::new(x, y)
        });

        let u = compute_curvature_tensor_2d(&mesh);

        // Get the indices of corner vertices
        mesh.add_boundary_faces();
        let (_, bdy_ids) = mesh.boundary();
        let mut bdy_flag = vec![false; mesh.n_verts() as usize];
        bdy_ids
            .into_iter()
            .for_each(|i| bdy_flag[i as usize] = true);

        for (i_elem, (e, t)) in mesh.elems().zip(mesh.etags()).enumerate() {
            let u = u[i_elem];
            let ge = mesh.gelem(e);
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
        let (mut mesh, _) = mesh.boundary();

        mesh.mut_verts().for_each(|p| {
            let r = r_in + (r_out - r_in) * p[0];
            let theta = 3.0 * p[1];
            let x = r * f64::cos(theta);
            let y = r * f64::sin(theta);
            *p = Point::<2>::new(x, y)
        });

        mesh.compute_elem_to_elems();
        mesh.add_boundary_faces();

        let mut u = compute_curvature_tensor_2d(&mesh);
        fix_curvature(&mesh, &mut u, None)?;

        for (i_elem, (e, t)) in mesh.elems().zip(mesh.etags()).enumerate() {
            let u = u[i_elem];
            let ge = mesh.gelem(e);
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
                Point::<3>::new(x, y, z)
            })
            .collect();

        let mut surf = SimplexMesh::<3, Triangle>::new(
            verts,
            mesh.elems().collect::<Vec<_>>(),
            mesh.etags().collect::<Vec<_>>(),
            Vec::new(),
            Vec::new(),
        );
        // Get the indices of boundary vertices
        surf.add_boundary_faces();
        let (_, bdy_ids) = surf.boundary();
        let mut bdy_flag = vec![false; surf.n_verts() as usize];
        bdy_ids
            .into_iter()
            .for_each(|i| bdy_flag[i as usize] = true);

        let (u, v) = compute_curvature_tensor(&surf);

        for (i_elem, e) in surf.elems().enumerate() {
            let mut u = u[i_elem];
            let mut v = v[i_elem];
            let ge = surf.gelem(e);
            let c = ge.center();

            let is_boundary = e.into_iter().any(|i| bdy_flag[i as usize]);
            if !is_boundary {
                assert!(u.norm() < 1.1 / H_MAX);
                assert!(u.norm() > 1e-17);
                let u_ref = Point::<3>::new(0.0, 0.0, 1.0);
                u.normalize_mut();
                let cos_a = u.dot(&u_ref);
                assert!(f64::abs(cos_a) > 0.98);

                assert!(f64::abs(v.norm() - 1. / radius) < 1e-6 / radius);
                let mut v_ref = Point::<3>::new(-c[1], c[0], 0.0);
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
                Point::<3>::new(x, y, z)
            })
            .collect();

        let mut surf = SimplexMesh::<3, Triangle>::new(
            verts,
            mesh.elems().collect::<Vec<_>>(),
            mesh.etags().collect::<Vec<_>>(),
            Vec::new(),
            Vec::new(),
        );
        surf.add_boundary_faces();
        surf.compute_elem_to_elems();

        let (mut u, mut v) = compute_curvature_tensor(&surf);
        fix_curvature(&surf, &mut u, Some(&mut v))?;

        for (i_elem, e) in surf.elems().enumerate() {
            let mut u = u[i_elem];
            let mut v = v[i_elem];
            let ge = surf.gelem(e);
            let c = ge.center();

            assert!(u.norm() < 1.1 / H_MAX);
            assert!(u.norm() > 1e-17);
            let u_ref = Point::<3>::new(0.0, 0.0, 1.0);
            u.normalize_mut();
            let cos_a = u.dot(&u_ref);
            assert!(f64::abs(cos_a) > 0.98);

            assert!(f64::abs(v.norm() - 1. / radius) < 1e-3 / radius);
            let mut v_ref = Point::<3>::new(-c[1], c[0], 0.0);
            v_ref.normalize_mut();
            v.normalize_mut();
            let cos_a = v.dot(&v_ref);
            assert!(f64::abs(cos_a) > 0.98);
        }

        Ok(())
    }
}
