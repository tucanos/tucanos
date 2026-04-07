use core::f64;
use rayon::iter::ParallelIterator;
use std::marker::PhantomData;
use tmesh::{
    mesh::{Mesh, Simplex, GSimplex},
};
use crate::{
    metric::{ ImpliedMetric, Metric},
};
// const ADD_PERCENTAGE: f64 = 40.0;
pub trait ElementCostEstimator<const D: usize, M: Mesh<D>, T: Metric<D>>: Send + Sync 
where
<<M as Mesh<D>>::C as Simplex>::GEOM<D>: ImpliedMetric<T>{
    fn new(m: &[T]) -> Self;
    fn compute(&self, msh: &M, m: &[T]) -> Vec<f64>;
}

pub struct NoCostEstimator<const D: usize, M: Mesh<D>, T: Metric<D>>
where 
<<M as Mesh<D>>::C as Simplex>::GEOM<D>: ImpliedMetric<T> {
    _e: PhantomData<M>,
    _m: PhantomData<T>,
}

impl<const D: usize, M, T> ElementCostEstimator<D, M, T> for NoCostEstimator<D, M, T>
where 
    M: Mesh<D>,
    T: Metric<D>,
    <<M as Mesh<D>>::C as Simplex>::GEOM<D>: ImpliedMetric<T>,
    {
    fn new(_m: &[T]) -> Self {
        Self {
            _e: PhantomData,
            _m: PhantomData,
        }
    }

    fn compute(&self, msh: &M, _m: &[T]) -> Vec<f64> {
        vec![1.0; msh.n_elems() as usize]
    }
}

pub struct TotoCostEstimator<
    const D: usize,
    M: Mesh<D>,
    T: Metric<D>
> where
    <<M as Mesh<D>>::C as Simplex>::GEOM<D>: ImpliedMetric<T>
{
    _e: PhantomData<M::C>,
    _m: PhantomData<T>,
}


const SPLIT_UNIT_COST: f64 = 3.65; // Un split dure en moyenne 3.65 * plus longtemps qu'une vérif
const COLLAPSE_UNIT_COST: f64 = 8.395; // Un collpase dure en moyenne 2.3 * plus longtemps qu'une vérif 2.3*3.65 = 8.395
const VERIF_COST_SWAP: f64 = 1.0; // Coût de vérification de swap défini comme coût référence
const VERIF_COST_SPLIT: f64 = 0.016;
const VERIF_COST_COLLAPSE: f64 = 0.016;
const TOTAL_VERIF_COST: f64 = 1.032; // Cout cumulé des vérifications
const RHO_COLLAPSE: f64 = 0.0005; // Proportion de collapse engendré par un split
const RHO_SPLIT: f64 = 0.0005; // Proportion de split engendré par un collapse
const LAMBDA: f64 = 6.0; // Nombre d'arrêtes crées (resp supprimées) par un split (resp collapse) en moyenne

const TOTAL_SPLIT_COST: f64 = (SPLIT_UNIT_COST
    + COLLAPSE_UNIT_COST * RHO_COLLAPSE
    + LAMBDA * TOTAL_VERIF_COST * (1.0 - RHO_COLLAPSE))
    / (1.0 - RHO_SPLIT * RHO_COLLAPSE); // alpha dans la formule

const TOTAL_COLLAPSE_COST: f64 =
    (TOTAL_SPLIT_COST - SPLIT_UNIT_COST - LAMBDA * TOTAL_VERIF_COST) / RHO_COLLAPSE;

fn work_eval(initial_density: f64, actual_density: f64, intersected_density: f64, vol: f64) -> f64 {
    let insert_prop = intersected_density - initial_density;
    let insert_bool = if insert_prop < 0.1 { 1.0 } else { 0.0 };

    let collapse_prop = intersected_density - actual_density;
    let collapse_bool = if collapse_prop < 0.1 { 1.0 } else { 0.0 };

    vol * (TOTAL_SPLIT_COST * insert_prop // Un split provoque 0.10 collapse et 6 nouvelles arrêtes
        + TOTAL_COLLAPSE_COST * collapse_prop)// Un collapse provoque 0.78 split et supprime 6 arrêtes 
        + (VERIF_COST_SWAP + VERIF_COST_SPLIT * insert_bool + VERIF_COST_COLLAPSE * collapse_bool) // Si collapse ou split, on ne fait pas la vérification correspondante 
}

#[allow(clippy::new_without_default)]
impl<
    const D: usize,
    M: Mesh<D>,
    T: Metric<D>
> TotoCostEstimator<D, M, T>
where
   <<M as Mesh<D>>::C as Simplex>::GEOM<D>: ImpliedMetric<T>,
{
    #[must_use]
    pub const fn new(_m: &[T]) -> Self {
        Self {
            _e: PhantomData,
            _m: PhantomData,
        }
    }
}

impl<
    const D: usize,
    M: Mesh<D>,
    T: Metric<D>
        // + Into<<E::Geom<D, IsoMetric<D>> as ImpliedMetric<D, IsoMetric<D>>>::ImpliedMetricType>,
> ElementCostEstimator<D, M, T> for TotoCostEstimator<D, M, T>
where
<<M as Mesh<D>>::C as Simplex>::GEOM<D>: ImpliedMetric<T>
{
    fn new(_m: &[T]) -> Self {
        Self {
            _e: PhantomData,
            _m: PhantomData,
        }
    }

    fn compute(&self, msh: &M, m: &[T]) -> Vec<f64> {
        msh.par_elems()
            .map(|e| {
                let ge = msh.gelem(&e);
                let implied_metric = ge.implied_metric();
                let vol = ge.vol();
                let n_verts = <<M as Mesh<D>>::C as Simplex>::GEOM::<D>::N_VERTS;
                let weight = 1.0 / (n_verts as f64);
                let mean_target_metric =
                    T::interpolate(e.into_iter().map(|i| (weight, &m[i as usize])));
                let intersected_metric = implied_metric.intersect(&mean_target_metric);
                let d_initial_metric = implied_metric.density();
                let d_target_metric = mean_target_metric.density();
                let d_intersected = intersected_metric.density();
                work_eval(d_initial_metric, d_target_metric, d_intersected, vol)
            })
            .collect()
        
    }
}
