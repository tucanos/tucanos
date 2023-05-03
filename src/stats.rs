use crate::{geometry::Geometry, metric::Metric, remesher::Remesher, topo_elems::Elem, Idx};
use core::fmt;
use serde::Serialize;

/// Simple statistics (histogram + mean) to be used on edge lengths and element qualities
#[derive(Serialize)]
pub struct Stats {
    /// Histogram bins (length = n+1)
    pub bins: Vec<f64>,
    /// Histogram values (length = n)
    pub vals: Vec<f64>,
    pub mean: f64,
}

impl Stats {
    /// Compute the stats
    /// the bins use the minimum / maximum of the iterator as first and last values, and values in between
    pub fn new<I: Iterator<Item = f64>>(f: I, values: &[f64]) -> Self {
        let mut mini = f64::INFINITY;
        let mut maxi = f64::NEG_INFINITY;
        let mut count = 0;

        let n = values.len();

        let mut bins = vec![0.0; n + 2];
        bins[0] = mini;
        bins[1..=n].copy_from_slice(&values[..(n + 1 - 1)]);
        bins[n + 1] = maxi;
        let mut vals = vec![0.0; n + 1];
        let mut mean = 0.;
        for val in f {
            mini = mini.min(val);
            maxi = maxi.max(val);
            count += 1;
            mean += val;
            if val < bins[1] {
                vals[0] += 1.0;
            } else if val > bins[n] {
                vals[n] += 1.;
            } else {
                for i in 1..n {
                    if val > bins[i] && val <= bins[i + 1] {
                        vals[i] += 1.0;
                        break;
                    }
                }
            }
        }

        loop {
            let mut stop = true;
            for (i, val) in vals.iter().enumerate() {
                if *val < 0.5 {
                    bins.remove(i + 1);
                    vals.remove(i);
                    stop = false;
                    break;
                }
            }
            if stop {
                break;
            }
        }

        for val in &mut vals {
            *val /= f64::from(count);
        }

        let n = bins.len();
        bins[0] = mini;
        bins[n - 1] = maxi;
        mean /= f64::from(count);

        Self { bins, vals, mean }
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mean = {:.2}", self.mean)?;
        let n = self.bins.len() - 1;
        for i in 0..n {
            write!(
                f,
                ", {:.2} < {:.1}% < {:.2}",
                self.bins[i],
                100.0 * self.vals[i],
                self.bins[i + 1]
            )?;
        }
        Ok(())
    }
}

/// Statistics on the remesher state
#[derive(Serialize)]
pub struct RemesherStats {
    /// The # of vertices in the mesh
    n_verts: Idx,
    /// The # of elements in the mesh
    n_elems: Idx,
    /// The # of edges in the mesh
    n_edges: Idx,
    /// Edge length stats
    stats_l: Stats,
    /// Element quality stats
    stats_q: Stats,
}

impl RemesherStats {
    pub fn new<const D: usize, E: Elem, M: Metric<D>, G: Geometry<D>>(
        r: &Remesher<D, E, M, G>,
    ) -> Self {
        Self {
            n_verts: r.n_verts(),
            n_elems: r.n_elems(),
            n_edges: r.n_edges(),
            stats_l: Stats::new(r.lengths_iter(), &[f64::sqrt(0.5), f64::sqrt(2.0)]),
            stats_q: Stats::new(r.qualities_iter(), &[0.4, 0.6, 0.8]),
        }
    }
}

/// Statistics for each remeshing step that include `RemesherStats` and additional step-dependent info
#[derive(Serialize)]
pub enum StepStats {
    Init(InitStats),
    Split(SplitStats),
    Swap(SwapStats),
    Collapse(CollapseStats),
    Smooth(SmoothStats),
}

#[derive(Serialize)]
pub struct InitStats {
    r_stats: RemesherStats,
}

impl InitStats {
    pub fn new<const D: usize, E: Elem, M: Metric<D>, G: Geometry<D>>(
        r: &Remesher<D, E, M, G>,
    ) -> Self {
        Self {
            r_stats: RemesherStats::new(r),
        }
    }
}

#[derive(Serialize)]
pub struct SplitStats {
    n_splits: Idx,
    n_fails: Idx,
    r_stats: RemesherStats,
}

impl SplitStats {
    pub fn new<const D: usize, E: Elem, M: Metric<D>, G: Geometry<D>>(
        n_splits: Idx,
        n_fails: Idx,
        r: &Remesher<D, E, M, G>,
    ) -> Self {
        Self {
            n_splits,
            n_fails,
            r_stats: RemesherStats::new(r),
        }
    }
}

#[derive(Serialize)]
pub struct SwapStats {
    n_swaps: Idx,
    n_fails: Idx,
    r_stats: RemesherStats,
}

impl SwapStats {
    pub fn new<const D: usize, E: Elem, M: Metric<D>, G: Geometry<D>>(
        n_swaps: Idx,
        n_fails: Idx,
        r: &Remesher<D, E, M, G>,
    ) -> Self {
        Self {
            n_swaps,
            n_fails,
            r_stats: RemesherStats::new(r),
        }
    }
}

#[derive(Serialize)]
pub struct CollapseStats {
    n_collapses: Idx,
    n_fails: Idx,
    r_stats: RemesherStats,
}

impl CollapseStats {
    pub fn new<const D: usize, E: Elem, M: Metric<D>, G: Geometry<D>>(
        n_collapses: Idx,
        n_fails: Idx,
        r: &Remesher<D, E, M, G>,
    ) -> Self {
        Self {
            n_collapses,
            n_fails,
            r_stats: RemesherStats::new(r),
        }
    }
}

#[derive(Serialize)]
pub struct SmoothStats {
    n_fails: Idx,
    r_stats: RemesherStats,
}

impl SmoothStats {
    pub fn new<const D: usize, E: Elem, M: Metric<D>, G: Geometry<D>>(
        n_fails: Idx,
        r: &Remesher<D, E, M, G>,
    ) -> Self {
        Self {
            n_fails,
            r_stats: RemesherStats::new(r),
        }
    }
}
