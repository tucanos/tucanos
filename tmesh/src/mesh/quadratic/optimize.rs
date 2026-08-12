use rustc_hash::{FxBuildHasher, FxHashSet};

use crate::Vert3d;
use crate::mesh::QuadraticGTetrahedron;

use super::super::{AdaptiveBoundsQuadraticTetrahedron, GSimplex, Mesh, QuadraticMesh3d};
use super::nelder_mead::{NelderMeadParams, minimize_nelder_mead};

fn get_element_and_mid_edge_indices(
    msh: &QuadraticMesh3d,
    threshold: f64,
) -> (Vec<usize>, Vec<usize>) {
    let lu = AdaptiveBoundsQuadraticTetrahedron::lagrange_to_bezier();

    let mut ho_vert_ids = FxHashSet::with_hasher(FxBuildHasher);
    let mut tmp = vec![false; msh.n_verts()];
    let mut count = 0;
    msh.elems().for_each(|e| {
        let ge = msh.gelem(&e);
        let (_, (min, max)) =
            AdaptiveBoundsQuadraticTetrahedron::new(&ge, &lu).compute_bounds(None);

        assert!(max > 0.0, "Element has non-positive max J: {max}");

        if min < threshold * max {
            e.into_iter().for_each(|i| tmp[i] = true);
            for i in e.into_iter().skip(4) {
                ho_vert_ids.insert(i);
            }
            count += 1;
        }
    });
    println!("Flagged {count} elements with min J < {threshold} * max J");
    println!("Flagged {} high-order vertices", ho_vert_ids.len());

    let elem_ids = msh
        .elems()
        .enumerate()
        .filter(|(_, e)| e.into_iter().any(|i| tmp[i]))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();

    (elem_ids, ho_vert_ids.into_iter().collect::<Vec<_>>())
}

fn compute_min_max(
    gelems: impl ExactSizeIterator<Item = QuadraticGTetrahedron<3>>,
) -> Vec<(f64, f64)> {
    let lu = AdaptiveBoundsQuadraticTetrahedron::lagrange_to_bezier();
    gelems
        .map(|ge| {
            let (_, (min, max)) =
                AdaptiveBoundsQuadraticTetrahedron::new(&ge, &lu).compute_bounds(None);
            (min, max)
        })
        .collect::<Vec<_>>()
}

pub fn optimize_quadratic_mesh(msh: &mut QuadraticMesh3d, is_fixed_vertex: &[bool]) {
    let (elem_ids, ho_vert_ids) = get_element_and_mid_edge_indices(msh, 0.0);

    let min_max = compute_min_max(elem_ids.iter().map(|&i| msh.gelem(&msh.elem(i))));
    let count = min_max.iter().filter(|&&(min, _)| min < 0.0).count();
    let max = min_max
        .iter()
        .map(|&(min, max)| max / min)
        .fold(0.0, f64::max);

    println!(
        "Before: {count} elements with negative J after optimization, max J ratio = {max:.3e}"
    );

    for iter in 0..10 {
        for &i_vert in &ho_vert_ids {
            if is_fixed_vertex[i_vert] {
                continue;
            }
            let eids = elem_ids
                .iter()
                .filter(|&&eid| msh.elem(eid).into_iter().any(|j| j == i_vert))
                .copied()
                .collect::<Vec<_>>();

            let opt = {
                let min_max = compute_min_max(eids.iter().map(|&i| msh.gelem(&msh.elem(i))));
                let b = 1.1
                    * min_max
                        .iter()
                        .map(|&(min, _)| min)
                        .fold(f64::INFINITY, f64::min);
                let h = eids
                    .iter()
                    .map(|&i| msh.gelem(&msh.elem(i)).linear().radius())
                    .fold(f64::MAX, f64::min);

                let func = |x: &Vert3d| {
                    let mut cost = 0.0;
                    let lu = AdaptiveBoundsQuadraticTetrahedron::lagrange_to_bezier();
                    for &i_elem in &eids {
                        let e = msh.elem(i_elem);
                        let mut ge = msh.gelem(&e);
                        let j = e.into_iter().position(|j| j == i_vert).unwrap();
                        ge.set(j, *x);
                        let (_, (min, max)) =
                            AdaptiveBoundsQuadraticTetrahedron::new(&ge, &lu).compute_bounds(None);
                        let f = (max / (min - b)).ln();
                        cost += f;
                    }
                    cost
                };

                let params = NelderMeadParams {
                    initial: msh.vert(i_vert),
                    step: h,
                    max_iters: 100,
                    tolerance: 1e-6,
                };
                minimize_nelder_mead(func, params)
            };
            msh.set_vert(i_vert, opt);
        }
        let min_max = compute_min_max(elem_ids.iter().map(|&i| msh.gelem(&msh.elem(i))));
        let count = min_max.iter().filter(|&&(min, _)| min < 0.0).count();
        let max = min_max
            .iter()
            .map(|&(min, max)| max / min)
            .fold(0.0, f64::max);
        println!(
            "Iteration {iter}: {count} elements with negative J after optimization, max J ratio = {max:.3e}"
        );
        if count == 0 {
            break;
        }
    }
}
