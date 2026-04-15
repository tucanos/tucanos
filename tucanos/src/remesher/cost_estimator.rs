use crate::metric::{ImpliedMetric, Metric};
use core::f64;
use rayon::iter::ParallelIterator;
use std::marker::PhantomData;
use tmesh::mesh::{GSimplex, Mesh, Simplex};

/// Defines the interface for element-wise computational load estimation.
pub trait ElementCostEstimator<const D: usize, C: Simplex, T: Metric<D>>:
    Send + Sync + Clone + Copy
{
    fn new() -> Self;

    /// Evaluates and returns the estimated computational weight of each mesh element.
    fn compute(&self, msh: &impl Mesh<D, C = C>, m: Option<&[T]>) -> Vec<f64>;
}

/// Assigns a strictly uniform weight of 1.0 to all elements, ignoring local metric-based refinement needs.
#[derive(Clone)]
pub struct NoCostEstimator<const D: usize, C: Simplex, T: Metric<D>> {
    _c: PhantomData<C>,
    _m: PhantomData<T>,
}

impl<const D: usize, C: Simplex, T: Metric<D>> Copy for NoCostEstimator<D, C, T> {}

impl<const D: usize, C: Simplex, T: Metric<D>> ElementCostEstimator<D, C, T>
    for NoCostEstimator<D, C, T>
{
    fn new() -> Self {
        Self {
            _c: PhantomData,
            _m: PhantomData,
        }
    }

    fn compute(&self, msh: &impl Mesh<D, C = C>, _m: Option<&[T]>) -> Vec<f64> {
        vec![1.0; msh.n_elems() as usize]
    }
}

/// Estimates workload based on constant unit costs for topological operations driven by metric density discrepancies.
#[derive(Clone)]
pub struct UniformCostEstimator<const D: usize, C: Simplex, T: Metric<D>> {
    _c: PhantomData<C>,
    _m: PhantomData<T>,
}

impl<const D: usize, C: Simplex, T: Metric<D>> UniformCostEstimator<D, C, T> {
    /// Estimates the operational work required to transition from the initial to the target metric density.
    fn work_eval(
        initial_density: f64,
        target_density: f64,
        intersected_density: f64,
        vol: f64,
    ) -> f64 {
        const SPLIT_UNIT_COST: f64 = 1.0;
        const COLLAPSE_UNIT_COST: f64 = 1.0;
        const SMOOTH_UNIT_COST: f64 = 1.0;

        let insert_prop = intersected_density - initial_density;
        let collapse_prop = intersected_density - target_density;

        vol * (SPLIT_UNIT_COST * insert_prop
            + COLLAPSE_UNIT_COST * collapse_prop
            + SMOOTH_UNIT_COST * target_density)
    }
}

impl<const D: usize, C: Simplex, T: Metric<D>> Copy for UniformCostEstimator<D, C, T> {}

impl<const D: usize, C: Simplex, T: Metric<D>> ElementCostEstimator<D, C, T>
    for UniformCostEstimator<D, C, T>
where
    C::GEOM<D>: ImpliedMetric<T>,
{
    fn new() -> Self {
        Self {
            _c: PhantomData,
            _m: PhantomData,
        }
    }

    fn compute(&self, msh: &impl Mesh<D, C = C>, m: Option<&[T]>) -> Vec<f64> {
        let m_slice = m.expect("UniformCostEstimator requires metrics");
        msh.par_elems()
            .map(|e| {
                let ge = msh.gelem(&e);
                let implied_metric = ge.implied_metric();
                let vol = ge.vol();
                let n_verts = C::GEOM::<D>::N_VERTS;
                let weight = 1.0 / (n_verts as f64);
                let mean_target_metric =
                    T::interpolate(e.into_iter().map(|i| (weight, &m_slice[i as usize])));
                let intersected_metric = implied_metric.intersect(&mean_target_metric);

                let d_initial_metric = implied_metric.density();
                let d_target_metric = mean_target_metric.density();
                let d_intersected = intersected_metric.density();

                UniformCostEstimator::<D, C, T>::work_eval(
                    d_initial_metric,
                    d_target_metric,
                    d_intersected,
                    vol,
                )
            })
            .collect()
    }
}

/// Advanced cost estimator integrating a probabilistic model of operation cascades and empirical execution time ratios.
#[derive(Clone)]
pub struct CustomCostEstimator<const D: usize, C: Simplex, T: Metric<D>> {
    _c: PhantomData<C>,
    _m: PhantomData<T>,
}

impl<const D: usize, C: Simplex, T: Metric<D>> CustomCostEstimator<D, C, T> {
    /// Predicts computational cost by resolving the interdependent probabilities of edge splits and collapses.
    fn work_eval(
        initial_density: f64,
        target_density: f64,
        intersected_density: f64,
        vol: f64,
    ) -> f64 {
        // Empirical relative durations of fundamental topological operations
        const SPLIT_UNIT_COST: f64 = 3.65;
        const COLLAPSE_UNIT_COST: f64 = 8.395;
        const VERIF_COST_SWAP: f64 = 1.0;
        const VERIF_COST_SPLIT: f64 = 0.016;
        const VERIF_COST_COLLAPSE: f64 = 0.016;
        const TOTAL_VERIF_COST: f64 = 1.032;

        // Probabilistic operation cascade modeling
        const RHO_COLLAPSE: f64 = 0.0005; // Probability of a collapse triggered by a preceding split
        const RHO_SPLIT: f64 = 0.0005; // Probability of a split triggered by a preceding collapse
        const LAMBDA: f64 = 6.0; // Average number of edges created/removed per operation

        // Coupled costs derived from solving the interdependent recursion equations
        const TOTAL_SPLIT_COST: f64 = (SPLIT_UNIT_COST
            + COLLAPSE_UNIT_COST * RHO_COLLAPSE
            + LAMBDA * TOTAL_VERIF_COST * (1.0 - RHO_COLLAPSE))
            / (1.0 - RHO_SPLIT * RHO_COLLAPSE);

        const TOTAL_COLLAPSE_COST: f64 =
            (TOTAL_SPLIT_COST - SPLIT_UNIT_COST - LAMBDA * TOTAL_VERIF_COST) / RHO_COLLAPSE;

        let insert_prop = intersected_density - initial_density;
        // Heuristic threshold to bypass verification costs if topological changes are negligible
        let insert_bool = if insert_prop < 0.1 { 1.0 } else { 0.0 };

        let collapse_prop = intersected_density - target_density;
        let collapse_bool = if collapse_prop < 0.1 { 1.0 } else { 0.0 };

        vol * (TOTAL_SPLIT_COST * insert_prop + TOTAL_COLLAPSE_COST * collapse_prop)
            + (VERIF_COST_SWAP
                + VERIF_COST_SPLIT * insert_bool
                + VERIF_COST_COLLAPSE * collapse_bool)
    }
}

impl<const D: usize, C: Simplex, T: Metric<D>> Copy for CustomCostEstimator<D, C, T> {}

impl<const D: usize, C: Simplex, T: Metric<D>> ElementCostEstimator<D, C, T>
    for CustomCostEstimator<D, C, T>
where
    C::GEOM<D>: ImpliedMetric<T>,
{
    fn new() -> Self {
        Self {
            _c: PhantomData,
            _m: PhantomData,
        }
    }

    fn compute(&self, msh: &impl Mesh<D, C = C>, m: Option<&[T]>) -> Vec<f64> {
        let m_slice = m.expect("CustomCostEstimator requires metrics");
        msh.par_elems()
            .map(|e| {
                let ge = msh.gelem(&e);
                let implied_metric = ge.implied_metric();
                let vol = ge.vol();
                let n_verts = C::GEOM::<D>::N_VERTS;
                let weight = 1.0 / (n_verts as f64);
                let mean_target_metric =
                    T::interpolate(e.into_iter().map(|i| (weight, &m_slice[i as usize])));
                let intersected_metric = implied_metric.intersect(&mean_target_metric);

                let d_initial_metric = implied_metric.density();
                let d_target_metric = mean_target_metric.density();
                let d_intersected = intersected_metric.density();

                CustomCostEstimator::<D, C, T>::work_eval(
                    d_initial_metric,
                    d_target_metric,
                    d_intersected,
                    vol,
                )
            })
            .collect()
    }
}
