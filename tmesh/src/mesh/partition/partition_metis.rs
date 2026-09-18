use super::{Mesh, Partitioner, init_or_check_weights};
use crate::{Result, graph::CSRGraph};
use std::marker::PhantomData;

/// Metis partitioning method
pub enum MetisMethod {
    /// Recursive algorithm in Metis
    Recursive,
    /// KWay algorithm in Metis
    KWay,
}

/// Metis partitioning method selector
pub trait MetisPartMethod: Send + Sync {
    /// Metis partitioning method
    fn method() -> MetisMethod;
}

/// Recursive algorithm in Metis
pub struct MetisRecursive;

impl MetisPartMethod for MetisRecursive {
    fn method() -> MetisMethod {
        MetisMethod::Recursive
    }
}

/// KWay algorithm in Metis
pub struct MetisKWay;

impl MetisPartMethod for MetisKWay {
    fn method() -> MetisMethod {
        MetisMethod::KWay
    }
}

/// Metis partitioner
pub struct MetisPartitioner<T: MetisPartMethod> {
    n_parts: usize,
    graph: CSRGraph,
    weights: Vec<f64>,
    t: PhantomData<T>,
}

impl<T: MetisPartMethod> Partitioner for MetisPartitioner<T> {
    fn new<const D: usize, M: Mesh<D>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self> {
        let faces = msh.all_faces();
        let graph = msh.element_pairs(&faces);

        let weights = init_or_check_weights(weights, msh.n_elems());

        Ok(Self {
            n_parts,
            graph,
            weights,
            t: PhantomData::<T>,
        })
    }

    fn compute(&self) -> Result<Vec<usize>> {
        if self.n_parts == 1 {
            return Ok(vec![0; self.graph.n()]);
        }

        let mut xadj = Vec::<metis::Idx>::with_capacity(self.graph.n() + 1);
        let mut adjncy = Vec::<metis::Idx>::with_capacity(self.graph.n_edges());

        xadj.push(0);
        for row in self.graph.rows() {
            for &j in row {
                adjncy.push(j.try_into().unwrap());
            }
            xadj.push(adjncy.len().try_into().unwrap());
        }

        let metis_graph =
            metis::Graph::new(1, self.n_parts.try_into().unwrap(), &mut xadj, &mut adjncy);

        let mut partition = vec![0; self.graph.n()];

        let _ = match T::method() {
            MetisMethod::Recursive => metis_graph.part_recursive(&mut partition)?,
            MetisMethod::KWay => metis_graph.part_kway(&mut partition)?,
        };

        let partition = partition.iter().map(|&x| x.try_into().unwrap()).collect();
        Ok(partition)
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
        partition::{MetisPartitioner, MetisRecursive, Partitioner},
    };

    #[test]
    fn test_metis_recursive() {
        let msh: Mesh3d = box_mesh(1.0, 10, 1.0, 15, 1.0, 20);
        let msh = msh.random_shuffle();

        let partitioner = MetisPartitioner::<MetisRecursive>::new(&msh, 4, None).unwrap();
        let parts = partitioner.compute().unwrap();

        assert!(partitioner.partition_quality(&parts) < 0.041);
        assert!(partitioner.partition_imbalance(&parts) < 0.001);
    }
}
