use std::marker::PhantomData;

use crate::metric::Metric;
use log::debug;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use tmesh::mesh::{GSimplex, Mesh, Simplex};

mod complexity;
mod curvature;
mod gradation;
pub mod implied;
mod scaling;
mod smoothing;

pub struct MetricField<'a, const D: usize, C: Simplex, M: Mesh<D, C>, T: Metric<D>> {
    msh: &'a M,
    metric: Vec<T>,
    vols: Vec<f64>,
    _c: PhantomData<C>,
}

impl<'a, const D: usize, C: Simplex, M: Mesh<D, C>, T: Metric<D>> MetricField<'a, D, C, M, T> {
    pub fn new(msh: &'a M, metric: Vec<T>) -> Self {
        let mut vols = vec![0.0; msh.n_verts()];
        for e in msh.elems() {
            let v = msh.gelem(&e).vol() / C::N_VERTS as f64;
            e.into_iter().for_each(|i| vols[i] += v);
        }

        Self {
            msh,
            metric,
            vols,
            _c: PhantomData,
        }
    }

    #[must_use]
    pub fn metric(&self) -> &[T] {
        &self.metric
    }

    /// Convert a metric field defined at the element centers (P0) to a field defined at the vertices (P1)
    /// using the interpolation method appropriate for the metric type.
    /// vertex-to-element connectivity and volumes are required
    pub fn from_elem_metric(msh: &'a M, metric: &[T]) -> Self {
        debug!("Convert metric element data to vertex data");

        let n_elems = msh.n_elems();
        let n_verts = msh.n_verts();
        assert_eq!(metric.len(), n_elems);

        let mut res = vec![T::default(); n_verts];

        let v2e = msh.vertex_to_elems();

        let vol = msh.gelems().map(|ge| ge.vol()).collect::<Vec<_>>();

        res.par_iter_mut().enumerate().for_each(|(i_vert, m_vert)| {
            let elems = v2e.row(i_vert);
            let mut vert_vol = 0.0;
            for &i_elem in elems {
                vert_vol += vol[i_elem];
            }

            let n_elems = elems.len();
            let mut weights = Vec::with_capacity(n_elems);
            let mut metrics = Vec::with_capacity(n_elems);
            weights.extend(elems.iter().map(|&i| vol[i] / vert_vol));
            metrics.extend(elems.iter().map(|&i| metric[i]));
            let wm = weights.iter().copied().zip(metrics.iter());
            *m_vert = T::interpolate(wm);
        });

        Self::new(msh, res)
    }

    /// Convert a metric field defined at the element centers (P0) to a field defined at the vertices (P1)
    /// using the interpolation method appropriate for the metric type.
    /// vertex-to-element connectivity and volumes are required
    #[must_use]
    pub fn to_elem_data(&self) -> Vec<T> {
        debug!("Convert metric vertex data to element data");

        let n_elems = self.msh.n_elems();

        let mut res = vec![T::default(); n_elems];

        let f = 1. / C::N_VERTS as f64;

        res.par_iter_mut()
            .zip(self.msh.par_elems())
            .for_each(|(m_elem, e)| {
                let mut weights = Vec::with_capacity(C::N_VERTS);
                let mut metrics = Vec::with_capacity(C::N_VERTS);
                weights.resize(C::N_VERTS, f);
                metrics.extend(e.into_iter().map(|i| self.metric[i]));
                let wm = weights.iter().copied().zip(metrics.iter());
                *m_elem = T::interpolate(wm);
            });
        res
    }
}
