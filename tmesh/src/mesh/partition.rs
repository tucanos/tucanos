//! Mesh partitioners
use super::{GSimplex, Mesh, hilbert::hilbert_indices};
use crate::{Error, Result, graph::CSRGraph};
#[cfg(feature = "coupe")]
use coupe::{Partition, nalgebra::SVector};
#[cfg(feature = "metis")]
use std::marker::PhantomData;

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

/// Check the partitioner inputs
fn check_inputs(n_elems: usize, n_parts: usize, weights: Option<&[f64]>) -> Result<()> {
    if n_parts == 0 {
        return Err(Error::from("The number of partitions must be > 0"));
    }
    if let Some(weights) = weights {
        if weights.len() != n_elems {
            return Err(Error::from(&format!(
                "Invalid number of weights: {} (expected {n_elems})",
                weights.len()
            )));
        }
        if weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
            return Err(Error::from("The weights must be finite and >= 0"));
        }
    }
    Ok(())
}

/// Partition elements ordered by `ids` into `n_parts` contiguous chunks of similar
/// weights: element `j` is assigned to part `floor(n_parts * w / w_tot)` where `w` is
/// the cumulative weight up to the middle of element `j`
fn ordered_partition(ids: &[usize], weights: &[f64], n_parts: usize) -> Vec<usize> {
    let total = weights.iter().sum::<f64>();
    let mut res = vec![0; weights.len()];
    if total <= 0.0 {
        return res;
    }
    let mut cumul = 0.0;
    for &j in ids {
        let w = cumul + 0.5 * weights[j];
        res[j] = ((n_parts as f64 * w / total) as usize).min(n_parts - 1);
        cumul += weights[j];
    }
    res
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
        check_inputs(msh.n_elems(), n_parts, weights.as_deref())?;
        let faces = msh.all_faces();
        let graph = msh.element_pairs(&faces);

        let centers = msh.gelems().map(|ge| ge.center());
        let ids = hilbert_indices(centers);
        let weights = weights.unwrap_or_else(|| vec![1.0; msh.n_elems()]);
        Ok(Self {
            n_parts,
            graph,
            ids,
            weights,
        })
    }

    fn compute(&self) -> Result<Vec<usize>> {
        Ok(ordered_partition(&self.ids, &self.weights, self.n_parts))
    }

    fn weights(&self) -> impl Iterator<Item = f64> {
        self.weights.iter().copied()
    }

    fn n_parts(&self) -> usize {
        self.n_parts
    }

    fn graph(&self) -> &CSRGraph {
        &self.graph
    }
}

/// Simple partioner based on the RCM ordering of the element-to-element
/// connectivity
pub struct RCMPartitioner {
    n_parts: usize,
    graph: CSRGraph,
    ids: Vec<usize>,
    weights: Vec<f64>,
}

impl Partitioner for RCMPartitioner {
    fn new<const D: usize, M: Mesh<D>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self> {
        check_inputs(msh.n_elems(), n_parts, weights.as_deref())?;
        let faces = msh.all_faces();
        let graph = msh.element_pairs(&faces);

        let weights = weights.unwrap_or_else(|| vec![1.0; msh.n_elems()]);
        let ids = graph.reverse_cuthill_mckee();
        Ok(Self {
            n_parts,
            graph,
            ids,
            weights,
        })
    }
    fn compute(&self) -> Result<Vec<usize>> {
        Ok(ordered_partition(&self.ids, &self.weights, self.n_parts))
    }

    fn weights(&self) -> impl Iterator<Item = f64> {
        self.weights.iter().copied()
    }

    fn n_parts(&self) -> usize {
        self.n_parts
    }

    fn graph(&self) -> &CSRGraph {
        &self.graph
    }
}

#[cfg(feature = "coupe")]
/// KMeans partitionner based on `coupe` (2d)
pub struct KMeansPartitioner2d {
    n_parts: usize,
    graph: CSRGraph,
    centers: Vec<SVector<f64, 2>>,
    weights: Vec<f64>,
}

#[cfg(feature = "coupe")]
impl Partitioner for KMeansPartitioner2d {
    fn new<const D: usize, M: Mesh<D>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self> {
        check_inputs(msh.n_elems(), n_parts, weights.as_deref())?;
        match D {
            2 => {
                let faces = msh.all_faces();
                let graph = msh.element_pairs(&faces);

                let centers = msh
                    .gelems()
                    .map(|ge| SVector::from_row_slice(ge.center().as_slice()))
                    .collect();
                let weights = weights.unwrap_or_else(|| vec![1.0; msh.n_elems()]);
                Ok(Self {
                    n_parts,
                    graph,
                    centers,
                    weights,
                })
            }
            _ => Err(Error::from("Partitioner only available for D=2")),
        }
    }
    fn compute(&self) -> Result<Vec<usize>> {
        let mut partition = vec![0; self.centers.len()];

        coupe::HilbertCurve {
            part_count: self.n_parts(),
            ..Default::default()
        }
        .partition(&mut partition, (self.centers.as_slice(), &self.weights))?;

        coupe::KMeans {
            delta_threshold: 0.0,
            ..Default::default()
        }
        .partition(&mut partition, (self.centers.as_slice(), &self.weights))?;

        Ok(partition)
    }

    fn weights(&self) -> impl Iterator<Item = f64> {
        self.weights.iter().copied()
    }

    fn n_parts(&self) -> usize {
        self.n_parts
    }

    fn graph(&self) -> &CSRGraph {
        &self.graph
    }
}

#[cfg(feature = "coupe")]
/// KMeans partitionner based on `coupe` (3d)
pub struct KMeansPartitioner3d {
    n_parts: usize,
    graph: CSRGraph,
    centers: Vec<SVector<f64, 3>>,
    weights: Vec<f64>,
}

#[cfg(feature = "coupe")]
impl Partitioner for KMeansPartitioner3d {
    fn new<const D: usize, M: Mesh<D>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self> {
        check_inputs(msh.n_elems(), n_parts, weights.as_deref())?;
        match D {
            3 => {
                let faces = msh.all_faces();
                let graph = msh.element_pairs(&faces);

                let centers = msh
                    .gelems()
                    .map(|ge| SVector::from_row_slice(ge.center().as_slice()))
                    .collect();
                let weights = weights.unwrap_or_else(|| vec![1.0; msh.n_elems()]);
                Ok(Self {
                    n_parts,
                    graph,
                    centers,
                    weights,
                })
            }
            _ => Err(Error::from("Partitioner only available for D=3")),
        }
    }
    fn compute(&self) -> Result<Vec<usize>> {
        let mut partition = vec![0; self.centers.len()];

        coupe::HilbertCurve {
            part_count: self.n_parts(),
            ..Default::default()
        }
        .partition(&mut partition, (self.centers.as_slice(), &self.weights))?;

        coupe::KMeans {
            delta_threshold: 0.0,
            ..Default::default()
        }
        .partition(&mut partition, (self.centers.as_slice(), &self.weights))?;
        Ok(partition)
    }

    fn weights(&self) -> impl Iterator<Item = f64> {
        self.weights.iter().copied()
    }

    fn n_parts(&self) -> usize {
        self.n_parts
    }

    fn graph(&self) -> &CSRGraph {
        &self.graph
    }
}

#[cfg(feature = "metis")]
/// Metis partitioning method
pub enum MetisMethod {
    /// Recursive algorithm in Metis
    Recursive,
    /// KWay algorithm in Metis
    KWay,
}

#[cfg(feature = "metis")]
/// Metis partitioning method
pub trait MetisPartMethod: Send + Sync {
    /// Metis partitioning method
    fn method() -> MetisMethod;
}

#[cfg(feature = "metis")]
/// Recursive algorithm in Metis
pub struct MetisRecursive;

#[cfg(feature = "metis")]
impl MetisPartMethod for MetisRecursive {
    fn method() -> MetisMethod {
        MetisMethod::Recursive
    }
}

#[cfg(feature = "metis")]
/// KWay algorithm in Metis
pub struct MetisKWay;

#[cfg(feature = "metis")]
impl MetisPartMethod for MetisKWay {
    fn method() -> MetisMethod {
        MetisMethod::KWay
    }
}

/// Metis preconditionner
#[cfg(feature = "metis")]
pub struct MetisPartitioner<T: MetisPartMethod> {
    n_parts: usize,
    graph: CSRGraph,
    weights: Vec<f64>,
    t: PhantomData<T>,
}

#[cfg(feature = "metis")]
impl<T: MetisPartMethod> Partitioner for MetisPartitioner<T> {
    fn new<const D: usize, M: Mesh<D>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self> {
        check_inputs(msh.n_elems(), n_parts, weights.as_deref())?;
        let faces = msh.all_faces();
        let graph = msh.element_pairs(&faces);

        let weights = weights.unwrap_or_else(|| vec![1.0; msh.n_elems()]);

        Ok(Self {
            n_parts,
            graph,
            weights,
            t: PhantomData::<T>,
        })
    }

    fn weights(&self) -> impl Iterator<Item = f64> {
        self.weights.iter().copied()
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

        // integer weights for Metis (only if the weights are not uniform)
        let w_max = self.weights.iter().copied().fold(0.0, f64::max);
        let w_tot = self.weights.iter().sum::<f64>();
        let mut vwgt = if self.weights.iter().any(|&w| w < w_max) {
            let scale = f64::min(1e6 / w_max, f64::from(i32::MAX / 2) / w_tot);
            self.weights
                .iter()
                .map(|&w| (w * scale).round().max(1.0) as metis::Idx)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let mut metis_graph =
            metis::Graph::new(1, self.n_parts.try_into().unwrap(), &mut xadj, &mut adjncy);
        if !vwgt.is_empty() {
            metis_graph = metis_graph.set_vwgt(&mut vwgt);
        }

        let mut partition = vec![0; self.graph.n()];

        let _ = match T::method() {
            MetisMethod::Recursive => metis_graph.part_recursive(&mut partition)?,
            MetisMethod::KWay => metis_graph.part_kway(&mut partition)?,
        };
        // metis_graph.part_recursive(&mut partition)?;
        // metis_graph.part_kway(&mut partition).unwrap()?;

        // convert to usize
        let partition = partition.iter().map(|&x| x.try_into().unwrap()).collect();
        Ok(partition)
    }

    fn n_parts(&self) -> usize {
        self.n_parts
    }

    fn graph(&self) -> &CSRGraph {
        &self.graph
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "coupe")]
    use crate::mesh::partition::{KMeansPartitioner2d, KMeansPartitioner3d};
    #[cfg(feature = "metis")]
    use crate::mesh::partition::{MetisPartitioner, MetisRecursive};
    use crate::mesh::{
        Mesh, Mesh2d, Mesh3d, box_mesh,
        partition::{HilbertPartitioner, Partitioner, RCMPartitioner},
        rectangle_mesh,
    };

    fn part_sizes(parts: &[usize], n_parts: usize) -> Vec<usize> {
        let mut sizes = vec![0; n_parts];
        for &i in parts {
            sizes[i] += 1;
        }
        sizes
    }

    #[test]
    fn test_no_empty_parts() {
        // 8 triangles
        let msh = rectangle_mesh::<Mesh2d>(1.0, 3, 1.0, 3);
        for n_parts in 1..=8 {
            let p = HilbertPartitioner::new(&msh, n_parts, None).unwrap();
            let sizes = part_sizes(&p.compute().unwrap(), n_parts);
            assert!(sizes.iter().all(|&n| n > 0), "Hilbert {n_parts}: {sizes:?}");
            assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);

            let p = RCMPartitioner::new(&msh, n_parts, None).unwrap();
            let sizes = part_sizes(&p.compute().unwrap(), n_parts);
            assert!(sizes.iter().all(|&n| n > 0), "RCM {n_parts}: {sizes:?}");
            assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
        }
    }

    #[test]
    fn test_partition_weights() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 3, 1.0, 3);
        let w = (0..8)
            .map(|i| if i < 4 { 10.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let p = HilbertPartitioner::new(&msh, 2, Some(w.clone())).unwrap();
        let parts = p.compute().unwrap();
        let mut expected = vec![0.0; 2];
        parts.iter().zip(&w).for_each(|(&i, &w)| expected[i] += w);
        assert_eq!(p.partition_weights(&parts), expected);
    }

    #[test]
    fn test_invalid_inputs() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 3, 1.0, 3);
        assert!(HilbertPartitioner::new(&msh, 0, None).is_err());
        assert!(RCMPartitioner::new(&msh, 0, None).is_err());
        assert!(HilbertPartitioner::new(&msh, 2, Some(vec![1.0; 3])).is_err());
    }

    #[test]
    fn test_hilbert() {
        let msh: Mesh3d = box_mesh(1.0, 10, 1.0, 15, 1.0, 20);
        let msh = msh.random_shuffle();

        let partitioner = HilbertPartitioner::new(&msh, 4, None).unwrap();
        let parts = partitioner.compute().unwrap();

        assert!(partitioner.partition_quality(&parts) < 0.06);
        assert!(partitioner.partition_imbalance(&parts) < 0.002);
    }

    #[test]
    fn test_rcm() {
        let msh: Mesh3d = box_mesh(1.0, 10, 1.0, 15, 1.0, 20);
        let msh = msh.random_shuffle();

        let partitioner = RCMPartitioner::new(&msh, 4, None).unwrap();
        let parts = partitioner.compute().unwrap();

        assert!(partitioner.partition_quality(&parts) < 0.06);
        assert!(partitioner.partition_imbalance(&parts) < 0.002);
    }

    #[test]
    #[cfg(feature = "coupe")]
    fn test_coupe_kmeans2d() {
        let msh: Mesh2d = rectangle_mesh(1.0, 5, 1.0, 6);
        let msh = msh.random_shuffle();

        let partitioner = KMeansPartitioner2d::new(&msh, 4, None).unwrap();
        let parts = partitioner.compute().unwrap();

        assert!(partitioner.partition_quality(&parts) < 0.2);
        assert!(partitioner.partition_imbalance(&parts) < 0.41);
    }

    #[test]
    #[cfg(feature = "coupe")]
    #[cfg_attr(debug_assertions, ignore = "Kmeans is slow")]
    fn test_coupe_kmeans() {
        let msh: Mesh3d = box_mesh(1.0, 6, 1.0, 5, 1.0, 5);
        let msh = msh.random_shuffle();

        let partitioner = KMeansPartitioner3d::new(&msh, 4, None).unwrap();
        let parts = partitioner.compute().unwrap();

        assert!(partitioner.partition_quality(&parts) < 0.11);
        assert!(partitioner.partition_imbalance(&parts) < 0.04);
    }

    #[cfg(feature = "metis")]
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
