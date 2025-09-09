use std::collections::HashMap;

use crate::{
    Vertex,
    graph::CSRGraph,
    mesh::{Cell, Face, Mesh, MutMesh, Simplex},
};
use coupe::nalgebra::{DMatrix, DVector};
use log::debug;
use rustc_hash::{FxBuildHasher, FxHashSet};

pub struct MeshDeformation<const D: usize> {
    a: f64,
    verts: Vec<Vertex<D>>,
    weights: DMatrix<f64>,
}

impl<const D: usize> MeshDeformation<D> {
    // fn phi(a: f64, x: &Vertex<D>, y: &Vertex<D>) -> f64 {
    //     let e = y - x;
    //     (-e.norm_squared() / (a * a)).exp()
    // }

    fn phi(a: f64, x: &Vertex<D>, y: &Vertex<D>) -> f64 {
        let e = y - x;
        let eta = e.norm() / a;
        if eta < 1.0 {
            (1.0 - eta).powi(4) * (4.0 * eta + 1.0)
        } else {
            0.0
        }
    }

    pub fn new<I: ExactSizeIterator<Item = (Vertex<D>, Vertex<D>)> + Clone>(
        constraints: &I,
    ) -> Self {
        let n = constraints.len();
        assert_ne!(n, 0);
        debug!("Compute mesh deformation with {n} constraints");
        let a = 4.0
            * constraints
                .clone()
                .map(|(_, x)| x.norm())
                .fold(0.0, f64::max);
        let verts = constraints.clone().map(|(x, _)| x).collect::<Vec<_>>();

        let mut mat = DMatrix::<f64>::zeros(n, n);

        debug!("Assemble the {n}x{n} matrix");
        for i in 0..n {
            *mat.index_mut((i, i)) = 1.0;
            for j in i + 1..n {
                let val = Self::phi(a, &verts[j], &verts[i]);
                *mat.index_mut((i, j)) = val;
                *mat.index_mut((j, i)) = val;
            }
        }
        debug!("Compute the Cholesky decomposition");
        let chol = mat.cholesky().unwrap();
        let mut weights = DMatrix::<f64>::zeros(n, D);
        for i_dim in 0..D {
            let mut rhs = DVector::<f64>::zeros(n);
            rhs.iter_mut()
                .zip(constraints.clone())
                .for_each(|(rhs, (_, x))| *rhs = x[i_dim]);
            weights.set_column(i_dim, &chol.solve(&rhs));
        }

        Self { a, verts, weights }
    }

    pub fn from_mesh<const C: usize, const F: usize, M: Mesh<D, C, F>>(
        msh: &M,
        deform: &[Vertex<D>],
    ) -> Self
    where
        Cell<C>: Simplex<C>,
        Cell<F>: Simplex<F>,
    {
        let constraints = msh.verts().zip(deform.iter().copied());
        Self::new(&constraints)
    }

    #[must_use]
    pub fn deform(&self, x: &Vertex<D>) -> Vertex<D> {
        self.weights
            .row_iter()
            .zip(self.verts.iter())
            .map(|(w, y)| {
                let w = Vertex::<D>::from_column_slice(w.clone_owned().as_slice());
                Self::phi(self.a, x, y) * w
            })
            .fold(Vertex::<D>::zeros(), |a, b| a + b)
    }
}

/// Locally deform the mesh to move vertex `i`, return `true` if deformation is valid
pub fn deform_mesh_local<
    const D: usize,
    const C: usize,
    const F: usize,
    M: MutMesh<D, C, F>,
    S: ::std::hash::BuildHasher,
>(
    msh: &mut M,
    all_faces: &HashMap<Face<F>, [usize; 3], S>,
    v2e: &CSRGraph,
    i: usize,
    d: &Vertex<D>,
    n_steps: usize,
) -> bool
where
    Cell<C>: Simplex<C>,
    Cell<F>: Simplex<F>,
{
    let x = msh.vert(i);
    let d_max = 3.0 * d.norm();

    // Flag the elements that have a face which distance to x is less than d_max
    let elem_to_faces = M::elem_to_faces();
    let mut elems = v2e.row(i).to_vec();
    let mut faces_visited = FxHashSet::with_hasher(FxBuildHasher);

    let mut added_elems = elems.clone();

    let mut bdy_faces = Vec::new();
    loop {
        let mut new_elems = Vec::new();
        for &i_elem in &added_elems {
            let e = msh.elem(i_elem);
            for face in &elem_to_faces {
                let mut tmp = [0; F];
                for (i, &j) in face.iter().enumerate() {
                    tmp[i] = e[j];
                }
                tmp.sort_unstable();
                if faces_visited.insert(tmp) {
                    let [_, i0, i1] = all_faces.get(&tmp).unwrap();
                    if *i0 == usize::MAX || *i1 == usize::MAX {
                        bdy_faces.push(tmp);
                    } else {
                        if *i0 == i_elem && !elems.contains(i1) {
                            if Face::<F>::distance(&msh.gface(&tmp), &x) < d_max {
                                new_elems.push(*i1);
                            } else {
                                bdy_faces.push(tmp);
                            }
                        }

                        if *i1 == i_elem && !elems.contains(i0) {
                            if Face::<F>::distance(&msh.gface(&tmp), &x) < d_max {
                                new_elems.push(*i0);
                            } else {
                                bdy_faces.push(tmp);
                            }
                        }
                    }
                }
            }
        }
        if new_elems.is_empty() {
            break;
        }
        elems.extend_from_slice(&new_elems);
        added_elems = new_elems;
    }

    let fac = 1.0 / n_steps as f64;
    let d_step = fac * d;

    // Get the constraints
    for _ in 0..n_steps {
        let idx_constraints = bdy_faces
            .iter()
            .flatten()
            .copied()
            .collect::<FxHashSet<_>>();
        let constraints = idx_constraints.iter().map(|&i_vert| {
            if i_vert == i {
                (msh.vert(i), d_step)
            } else {
                (msh.vert(i_vert), Vertex::<D>::zeros())
            }
        });

        // Compute the deformation
        let deform = MeshDeformation::new(&constraints);

        // Get the vertex to be deformed
        let idx_verts = elems
            .iter()
            .flat_map(|&i| msh.elem(i))
            .collect::<FxHashSet<_>>();
        let idx_verts = Vec::from_iter(idx_verts);
        let old_verts = idx_verts.iter().map(|&i| msh.vert(i)).collect::<Vec<_>>();

        for &i_vert in &idx_verts {
            let v = msh.vert(i_vert);
            if i == i_vert {
                msh.set_vert(i_vert, v + d_step);
            } else if !idx_constraints.contains(&i_vert) {
                msh.set_vert(i_vert, v + deform.deform(&v));
            }
        }

        // msh.write_vtk("local.vtu").unwrap();

        // Check that the mesh is valid, otherwise revert to the original vertices
        if elems.iter().any(|&i_elem| {
            let e = msh.elem(i_elem);
            let ge = msh.gelem(&e);
            Cell::<C>::vol(&ge) < 0.
        }) {
            for (&i_vert, &old_pos) in idx_verts.iter().zip(old_verts.iter()) {
                msh.set_vert(i_vert, old_pos);
            }
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashSet;

    use crate::{
        Vert2d, Vert3d,
        mesh::{BoundaryMesh2d, Mesh, Mesh2d, Mesh3d, MutMesh, box_mesh, rectangle_mesh},
    };

    use super::{MeshDeformation, deform_mesh_local};

    #[test]
    fn test_deform_2d() {
        let mut mesh = rectangle_mesh::<Mesh2d>(1.0, 9, 1.0, 9);
        let ids = mesh.faces().flatten().collect::<FxHashSet<_>>();
        let constraints = ids.iter().map(|&i| {
            let x = mesh.vert(i);
            if x[1].abs() < 1e-10 {
                (x, Vert2d::new(0.0, 0.25 * 4.0 * x[0] * (1.0 - x[0])))
            } else {
                (x, Vert2d::zeros())
            }
        });
        let deform = MeshDeformation::new(&constraints);
        mesh.verts_mut().for_each(|x| *x += deform.deform(x));

        let faces = mesh.all_faces();
        mesh.check(&faces).unwrap();
    }

    #[test]
    fn test_deform_2d_2() {
        let mut mesh = rectangle_mesh::<Mesh2d>(1.0, 9, 1.0, 9);
        let (bdy, _): (BoundaryMesh2d, _) = mesh.boundary();
        let h = bdy
            .verts()
            .map(|x| {
                if x[1].abs() < 1e-10 {
                    Vert2d::new(0.0, 0.25 * 4.0 * x[0] * (1.0 - x[0]))
                } else {
                    Vert2d::zeros()
                }
            })
            .collect::<Vec<_>>();
        let deform = MeshDeformation::from_mesh(&bdy, &h);
        mesh.verts_mut().for_each(|x| *x += deform.deform(x));

        let faces = mesh.all_faces();
        mesh.check(&faces).unwrap();
    }

    #[test]
    fn test_local_deform_2d() {
        let mut mesh = rectangle_mesh::<Mesh2d>(1.0, 9, 1.0, 9);

        let all_faces = mesh.all_faces();
        let v2e = mesh.vertex_to_elems();
        let d = Vert2d::new(0.0, 0.25);

        assert!(deform_mesh_local(&mut mesh, &all_faces, &v2e, 4, &d, 1));

        let faces = mesh.all_faces();
        mesh.check(&faces).unwrap();
    }

    #[test]
    fn test_local_deform_2d_2() {
        let mut mesh = rectangle_mesh::<Mesh2d>(1.0, 9, 1.0, 9);

        let all_faces = mesh.all_faces();
        let v2e = mesh.vertex_to_elems();
        let d = Vert2d::new(0.1, 0.25);

        assert!(deform_mesh_local(&mut mesh, &all_faces, &v2e, 4, &d, 2));

        let faces = mesh.all_faces();
        mesh.check(&faces).unwrap();
    }

    #[test]
    fn test_deform_3d() {
        let mut mesh = box_mesh::<Mesh3d>(1.0, 9, 1.0, 9, 1.0, 9);
        let ids = mesh.faces().flatten().collect::<FxHashSet<_>>();
        let constraints = ids.iter().map(|&i| {
            let x = mesh.vert(i);
            if x[1].abs() < 1e-10 {
                (
                    x,
                    Vert3d::new(
                        0.0,
                        0.25 * 16.0 * x[0] * (1.0 - x[0]) * x[2] * (1.0 - x[2]),
                        0.0,
                    ),
                )
            } else {
                (x, Vert3d::zeros())
            }
        });
        let deform = MeshDeformation::new(&constraints);
        mesh.verts_mut().for_each(|x| *x += deform.deform(x));

        let faces = mesh.all_faces();
        mesh.check(&faces).unwrap();

        // mesh.write_vtk("deform3d.vtu").unwrap();
    }
}
