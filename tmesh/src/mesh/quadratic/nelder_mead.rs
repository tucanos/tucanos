use std::cmp::Ordering;

use crate::Vertex;

/// Parameters for the Nelder-Mead optimization algorithm
#[derive(Clone, Copy, Debug)]
pub struct NelderMeadParams<const D: usize> {
    pub initial: Vertex<D>,
    pub step: f64,
    pub max_iters: usize,
    pub tolerance: f64,
}

/// Minimize a function using the Nelder-Mead algorithm
pub fn minimize_nelder_mead<const D: usize, F>(
    mut func: F,
    params: NelderMeadParams<D>,
) -> Vertex<D>
where
    F: FnMut(&Vertex<D>) -> f64,
{
    let NelderMeadParams {
        initial,
        step,
        max_iters,
        tolerance,
    } = params;

    assert!(D > 0, "Nelder-Mead needs at least one dimension");

    let step = step.abs().max(1e-12);
    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;

    let mut simplex = Vec::with_capacity(D + 1);
    simplex.push(initial);
    for axis in 0..D {
        let mut vertex = initial;
        vertex[axis] += step;
        simplex.push(vertex);
    }

    #[allow(clippy::redundant_closure)]
    let mut values = simplex.iter().map(|x| func(x)).collect::<Vec<_>>();

    let order = |values: &[f64]| {
        let mut indices = (0..values.len()).collect::<Vec<_>>();
        indices.sort_by(|&left, &right| {
            values[left]
                .partial_cmp(&values[right])
                .unwrap_or(Ordering::Equal)
        });
        indices
    };

    for _iter in 0..max_iters.max(1) {
        let indices = order(&values);
        let best = indices[0];
        let worst = indices[D];
        let second_worst = indices[D - 1];

        let best_value = values[best];
        let worst_value = values[worst];
        let value_spread = worst_value - best_value;
        let simplex_spread = simplex
            .iter()
            .map(|vertex| (vertex - simplex[best]).norm())
            .fold(0.0, f64::max);

        if value_spread <= tolerance || simplex_spread <= tolerance {
            return simplex[best];
        }

        let centroid = indices[..D]
            .iter()
            .fold(Vertex::zeros(), |acc, &index| acc + simplex[index])
            / D as f64;

        let reflected = centroid + alpha * (centroid - simplex[worst]);
        let reflected_value = func(&reflected);

        if reflected_value < best_value {
            let expanded = centroid + gamma * (reflected - centroid);
            let expanded_value = func(&expanded);
            if expanded_value < reflected_value {
                simplex[worst] = expanded;
                values[worst] = expanded_value;
            } else {
                simplex[worst] = reflected;
                values[worst] = reflected_value;
            }
            continue;
        }

        if reflected_value < values[second_worst] {
            simplex[worst] = reflected;
            values[worst] = reflected_value;
            continue;
        }

        let contracted = if reflected_value < worst_value {
            centroid + rho * (reflected - centroid)
        } else {
            centroid + rho * (simplex[worst] - centroid)
        };
        let contracted_value = func(&contracted);

        if contracted_value < worst_value {
            simplex[worst] = contracted;
            values[worst] = contracted_value;
            continue;
        }

        let best_vertex = simplex[best];
        for &index in &indices[1..] {
            simplex[index] = best_vertex + sigma * (simplex[index] - best_vertex);
            values[index] = func(&simplex[index]);
        }
    }

    let indices = order(&values);
    simplex[indices[0]]
}
