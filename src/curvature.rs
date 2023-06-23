use crate::{
    geom_elems::GElem,
    linalg::lapack_qr_least_squares,
    mesh::{Point, SimplexMesh},
    topo_elems::{Elem, Triangle},
    Error, Idx, Mesh, Result, H_MAX,
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

    for (i_elem, e) in mesh.elems().enumerate() {
        let ge = mesh.gelem(i_elem as Idx);
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

    for (i_elem, e) in smesh.elems().enumerate() {
        // Compute the element-based curvature
        let ge = smesh.gelem(i_elem as Idx);
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

        // change of basis
        let bound = |x| {
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
        };
        if ev0 > ev1 {
            u = eig.eigenvectors[0] * u + eig.eigenvectors[1] * v;
            v = n.cross(&u);
            u.normalize_mut();
            v.normalize_mut();

            u *= bound(ev0);
            v *= bound(ev1);
        } else {
            v = eig.eigenvectors[0] * u + eig.eigenvectors[1] * v;
            u = -n.cross(&v);

            v.normalize_mut();
            u.normalize_mut();

            u *= bound(ev1);
            v *= bound(ev0);
        }

        elem_u.push(u);
        elem_v.push(v);
    }

    (elem_u, elem_v)
}

pub fn fix_curvature(
    smesh: &SimplexMesh<3, Triangle>,
    u: &mut [Point<3>],
    v: &mut [Point<3>],
) -> Result<()> {
    debug!("Fix curvature tensors near the patch boundaries");

    if smesh.elem_to_elems.is_none() {
        return Err(Error::from(
            "element to elements connectivity not available",
        ));
    }

    let (_, bdy_ids) = smesh.boundary();
    let mut bdy_flag = vec![false; smesh.n_verts() as usize];
    bdy_ids
        .into_iter()
        .for_each(|i| bdy_flag[i as usize] = true);

    let e2e = smesh.elem_to_elems.as_ref().unwrap();
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
                    .filter(|i| smesh.etags[*i as usize] == smesh.etags[i_elem]);
                let mut min_dist = f64::MAX;
                let c = smesh.elem_center(i_elem as Idx);
                let mut i_neighbor = None;
                for i in valid_neighbors {
                    let c2 = smesh.elem_center(i);
                    let dist = (c2 - c).norm();
                    if dist < min_dist {
                        min_dist = dist;
                        i_neighbor = Some(i);
                    }
                }

                if let Some(i_neighbor) = i_neighbor {
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

#[cfg(test)]
mod tests {
    use crate::{
        geom_elems::GElem,
        mesh::{Point, SimplexMesh},
        test_meshes::test_mesh_2d,
        topo_elems::Triangle,
        Idx, Mesh, Result, H_MAX,
    };

    use super::{compute_curvature_tensor, fix_curvature};

    #[test]
    fn test_curvature_cylinder() {
        let radius = 0.1;

        let mesh = test_mesh_2d().split().split().split().split().split();

        let mut coords = Vec::new();

        for pt in mesh.verts() {
            let z = pt[0];
            let theta = 3.0 * pt[1];
            let x = radius * f64::cos(theta);
            let y = radius * f64::sin(theta);
            coords.push(x);
            coords.push(y);
            coords.push(z);
        }

        let mut surf =
            SimplexMesh::<3, Triangle>::new(coords, mesh.elems, mesh.etags, vec![0; 0], vec![0; 0]);
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
            let ge = surf.gelem(i_elem as Idx);
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

        let mut coords = Vec::new();

        for pt in mesh.verts() {
            let z = pt[0];
            let theta = 3.0 * pt[1];
            let x = radius * f64::cos(theta);
            let y = radius * f64::sin(theta);
            coords.push(x);
            coords.push(y);
            coords.push(z);
        }

        let mut surf =
            SimplexMesh::<3, Triangle>::new(coords, mesh.elems, mesh.etags, vec![0; 0], vec![0; 0]);
        surf.add_boundary_faces();
        surf.compute_elem_to_elems();

        let (mut u, mut v) = compute_curvature_tensor(&surf);
        fix_curvature(&surf, &mut u, &mut v)?;

        for i_elem in 0..surf.n_elems() as usize {
            let mut u = u[i_elem];
            let mut v = v[i_elem];
            let ge = surf.gelem(i_elem as Idx);
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
