//! Mesh partitioners
use super::{GSimplex, Mesh, hilbert::hilbert_indices};
use crate::{Result, graph::CSRGraph};

#[cfg(feature = "kahip")]
mod partition_kahip;
#[cfg(feature = "metis")]
mod partition_metis;

#[cfg(feature = "kahip")]
pub use partition_kahip::*;
#[cfg(feature = "metis")]
pub use partition_metis::*;

/// Mesh partitioners
pub trait Partitioner: Sized + Send + Sync {
    /// Create a new mesh partitionner to partition `msh` into `n_parts`
    /// Element weights can optionally be provided
    fn new<const D: usize, M: Mesh<D>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self>;

    /// Compute the element partition
    fn compute(&self) -> Result<Vec<usize>>;
    /// Get the number of partitions
    fn n_parts(&self) -> usize;
    /// Get the element-to-element graph
    fn graph(&self) -> &CSRGraph;
    /// Get the element weights
    fn weights(&self) -> impl Iterator<Item = f64> {
        (0..self.graph().n()).map(|_| 1.0)
    }
    /// Get the total weight of all the partitions
    fn partition_weights(&self, parts: &[usize]) -> Vec<f64> {
        let mut res = vec![0.0; self.n_parts()];
        for (&i_part, w) in parts.iter().zip(self.weights()) {
            res[i_part] += w;
        }
        res
    }
    /// Compute the imbalance between the partitions
    /// defined as (max(part_weights) - min(part_weights)) / mean(part_weights)
    fn partition_imbalance(&self, parts: &[usize]) -> f64 {
        let weights = self.partition_weights(parts);
        let (min, max, avg) = weights.iter().fold((f64::MAX, f64::MIN, 0.0), |a, &b| {
            (a.0.min(b), a.1.max(b), a.2 + b)
        });
        let avg = avg / weights.len() as f64;
        (max - min) / avg
    }
    /// Compute the quality of the partitioning defined as the ratio
    /// of the number of faces between elements on different partitions to the total
    /// number of internal faces
    fn partition_quality(&self, parts: &[usize]) -> f64 {
        let mut count = 0;
        let mut split = 0;
        for (i, row) in self.graph().rows().enumerate() {
            for &j in row {
                if j != i {
                    count += 1;
                    if parts[i] != parts[j] {
                        split += 1;
                    }
                }
            }
        }
        f64::from(split) / f64::from(count)
    }
}

fn init_or_check_weights(weights: Option<Vec<f64>>, n: usize) -> Vec<f64> {
    weights.map_or_else(
        || vec![1.0; n],
        |weights| {
            assert_eq!(
                weights.len(),
                n,
                "weights length must match number of elements"
            );
            weights
        },
    )
}

/// Simple geometric partitionner based on the Hilbert indices of the element centers
pub struct HilbertPartitioner {
    n_parts: usize,
    graph: CSRGraph,
    ids: Vec<usize>,
    weights: Vec<f64>,
}

impl Partitioner for HilbertPartitioner {
    fn new<const D: usize, M: Mesh<D>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self> {
        let faces = msh.all_faces();
        let graph = msh.element_pairs(&faces);

        let centers = msh.gelems().map(|ge| ge.center());
        let ids = hilbert_indices(centers);
        let weights = init_or_check_weights(weights, msh.n_elems());
        Ok(Self {
            n_parts,
            graph,
            ids,
            weights,
        })
    }

    fn compute(&self) -> Result<Vec<usize>> {
        let target_weight = self.weights.iter().copied().sum::<f64>() / self.n_parts as f64;
        let mut res = vec![0; self.weights.len()];
        let mut part = 0;
        let mut weight = 0.0;
        for &j in &self.ids {
            if weight > target_weight {
                part = self.n_parts.min(part + 1);
                weight = 0.0;
            }
            res[j] = part;
            weight += self.weights[j];
        }
        Ok(res)
    }

    fn n_parts(&self) -> usize {
        self.n_parts
    }

    fn graph(&self) -> &CSRGraph {
        &self.graph
    }

    fn weights(&self) -> impl Iterator<Item = f64> {
        self.weights.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use crate::mesh::{
        Mesh, Mesh3d, box_mesh,
        partition::{HilbertPartitioner, Partitioner},
    };

    #[test]
    fn test_hilbert() {
        let msh: Mesh3d = box_mesh(1.0, 10, 1.0, 15, 1.0, 20);
        let msh = msh.random_shuffle();

        let partitioner = HilbertPartitioner::new(&msh, 4, None).unwrap();
        let parts = partitioner.compute().unwrap();

        assert!(partitioner.partition_quality(&parts) < 0.06);
        assert!(partitioner.partition_imbalance(&parts) < 0.002);
    }
}
