use log::debug;
use nalgebra::SMatrix;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::{
    Vertex,
    graph::CSRGraph,
    mesh::{GSimplex, Mesh, Simplex},
};

/// Compute the gradient of a scalar field defined at the mesh vertices
/// as the L2 projection of the element-based gradient. The gradient at vertex
/// $`i`$ is approximated as
/// ```math
/// \nabla u_i = \frac{\sum_{K_j \ni i} |K_j| \nabla u_{K_j}}{\sum_{K_j \ni i} |K_j|}
/// ```
/// where the sum is considered over all elements that contain vertex $`i`$.
///
pub fn gradient_l2proj<const D: usize, C: Simplex, M: Mesh<D, C>>(
    msh: &M,
    v2e: &CSRGraph,
    f: &[f64],
) -> Vec<f64> {
    debug!("Compute gradient using L2 projection");

    assert_eq!(f.len(), msh.n_verts());

    let mut tmp = vec![0.0; D * msh.n_elems()];

    let mut vol = vec![0.0; msh.n_elems()];

    tmp.par_chunks_mut(D)
        .zip(msh.par_elems())
        .zip(vol.par_iter_mut())
        .for_each(|((g, e), v)| {
            let ge = msh.gelem(&e);
            let mut grad = Vertex::<D>::zeros();
            for i_face in 0..C::N_FACES {
                let i_vert = e.get(i_face);
                let gf = ge.face(i_face);
                grad += f[i_vert] * gf.normal();
            }
            g.iter_mut().zip(grad.iter()).for_each(|(x, y)| *x = *y);
            *v = ge.vol();
        });

    let fac = match D {
        2 => -6.0,
        3 => -12.0,
        _ => unreachable!(),
    };

    let mut res = vec![0.0; D * msh.n_verts()];
    res.par_chunks_mut(D).enumerate().for_each(|(i_vert, g)| {
        let mut v = 0.0;
        for &i_elem in v2e.row(i_vert) {
            v += vol[i_elem];
            for i in 0..D {
                g[i] += tmp[D * i_elem + i];
            }
        }
        v /= C::N_VERTS as f64;
        for x in g.iter_mut() {
            *x /= fac * v;
        }
    });

    res
}

/// Compute the hessian of a scalar field defined at the mesh vertices
/// applying twice the L2 projection gradient.
/// NB: this actually does not converge the the hessian!
pub fn hessian_l2proj<const D: usize, C: Simplex, M: Mesh<D, C>>(
    msh: &M,
    v2e: &CSRGraph,
    gradf: &[f64],
) -> Vec<f64> {
    debug!("Compute hessian using L2 projection");
    assert_eq!(gradf.len(), D * msh.n_verts());

    let indices = if D == 2 {
        vec![0, 3, 1]
    } else if D == 3 {
        vec![0, 4, 8, 1, 5, 2]
    } else {
        unreachable!()
    };
    let n = indices.len();

    let mut tmp = vec![0.0; n * msh.n_elems()];

    let mut vol = vec![0.0; msh.n_elems()];

    tmp.par_chunks_mut(n)
        .zip(msh.par_elems())
        .zip(vol.par_iter_mut())
        .for_each(|((h, e), v)| {
            let ge = msh.gelem(&e);
            let mut hess = SMatrix::<f64, D, D>::zeros();
            for i_face in 0..C::N_FACES {
                let i_vert = e.get(i_face);
                let start = D * i_vert;
                let end = start + D;
                let grad = &gradf[start..end];
                let gf = ge.face(i_face);
                let n = gf.normal();
                for i in 0..D {
                    for j in 0..D {
                        hess[D * i + j] += grad[i] * n[j];
                    }
                }
            }
            hess = 0.5 * (hess + hess.transpose());

            for (i, &j) in indices.iter().enumerate() {
                h[i] = hess[j];
            }

            *v = ge.vol();
        });

    let mut res = vec![0.0; n * msh.n_verts()];

    let fac = match D {
        2 => -6.0,
        3 => -12.0,
        _ => unreachable!(),
    };
    res.par_chunks_mut(n).enumerate().for_each(|(i_vert, g)| {
        let mut v = 0.0;
        for &i_elem in v2e.row(i_vert) {
            v += vol[i_elem];

            for i in 0..n {
                g[i] += tmp[n * i_elem + i];
            }
        }
        v /= C::N_VERTS as f64;

        for x in g.iter_mut() {
            *x /= fac * v;
        }
    });
    res
}
