//! Mesh partitioners
use super::{Cell, Face, Mesh, Simplex, cell_center, hilbert::hilbert_indices};
use crate::{Error, Result, Vert2d, Vert3d, argmax, graph::CSRGraph};
use coupe::Partition;

use std::collections::{HashSet, VecDeque};
#[cfg(feature = "metis")]
use std::marker::PhantomData;

#[derive(Copy, Clone)]
pub enum PartitionType {
    Hilbert(usize),
    HilbertBall(usize),
    BFS(usize),
    BFSWR(usize),
    RCM(usize),
    KMeans(usize),
    #[cfg(feature = "metis")]
    MetisRecursive(usize),
    #[cfg(feature = "metis")]
    MetisKWay(usize),
    None,
}
/// Mesh partitioners
pub trait Partitioner: Sized + Send + Sync {
    /// Create a new mesh partitionner to partition `msh` into `n_parts`
    /// Element weights can optionally be provided
    fn new<const D: usize, const C: usize, const F: usize, M: Mesh<D, C, F>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self>
    where
        Cell<C>: Simplex<C>,
        Face<F>: Simplex<F>;
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
    /// todo
    fn partition_correction(&self, part: &mut Vec<usize>) {
        let n_elems = self.graph().n();
        let weights = self.weights().collect::<Vec<_>>();
        for i_part in 0..self.n_parts() {
            let elem_ids = (0..n_elems)
                .filter(|&i| part[i] == i_part)
                .collect::<Vec<_>>();
            let sgraph = self.graph().subgraph(elem_ids.iter().copied());
            let cc = sgraph.connected_components().unwrap();
            let n_cc = cc.iter().copied().max().unwrap_or(0) + 1;
            println!("{i_part} -> {n_cc}");
            if n_cc > 1 {
                let mut cc_weights = vec![0.0; n_cc];
                let mut n_faces = vec![vec![0; self.n_parts()]; n_cc];
                for (i, &j) in elem_ids.iter().enumerate() {
                    let i_cc = cc[i];
                    cc_weights[cc[i]] += weights[j];
                    for &i_neighbor in self.graph().row(j) {
                        let i_part_neighbor = part[i_neighbor];
                        if i_part_neighbor != i_part {
                            n_faces[i_cc][i_part_neighbor] += 1;
                        }
                    }
                }
                let i_max_cc = argmax(&cc_weights).unwrap();
                for (i_cc, n_faces) in n_faces.iter().enumerate() {
                    if i_cc != i_max_cc {
                        //Option 1 : Maximize the quality of partitions
                        let i_new_part = argmax(n_faces).unwrap();
                        //Todo
                        //Option 2 : Maximize the load balancing
                        //Option 3 : Maximize Both
                        for (i, &j) in elem_ids.iter().enumerate() {
                            if cc[i] == i_cc {
                                part[j] = i_new_part;
                            }
                        }
                    }
                }
            }
        }
    }
}
pub struct BFSWRPartitionner {
    n_parts: usize,
    graph: CSRGraph,
    ids: Vec<usize>,
    weights: Vec<f64>,
}

impl Partitioner for BFSWRPartitionner {
    fn new<const D: usize, const C: usize, const F: usize, M: Mesh<D, C, F>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self>
    where
        Cell<C>: Simplex<C>,
        Face<F>: Simplex<F>,
    {
        let faces = msh.all_faces();
        let graph = msh.element_pairs(&faces);

        let centers = msh.gelems().map(|ge| cell_center(&ge));
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
        let target_weight = self.weights.iter().copied().sum::<f64>() / self.n_parts() as f64;
        let mut res = vec![0; self.weights.len()];
        let n_elems = self.weights.len();
        let mut assigned_elements: HashSet<usize> = HashSet::new();
        let mut current_partition_idx = 0;
        let mut current_work_partition = 0.0;
        let mut queue: VecDeque<usize> = VecDeque::new();

        let mut next_unassigned_elem_root = 0;

        while assigned_elements.len() < n_elems && current_partition_idx < self.n_parts() {
            while next_unassigned_elem_root < n_elems
                && assigned_elements.contains(&(self.ids[next_unassigned_elem_root]))
            {
                next_unassigned_elem_root += 1;
            }

            if next_unassigned_elem_root == n_elems {
                break;
            }

            let start_elem_id = next_unassigned_elem_root;
            queue.push_back(self.ids[start_elem_id]);
            assigned_elements.insert(self.ids[start_elem_id]);

            while let Some(current_elem_id) = queue.pop_front() {
                let elem_work = self.weights[current_elem_id];

                if current_work_partition + elem_work > target_weight
                    && current_partition_idx + 1 < self.n_parts()
                {
                    current_partition_idx += 1;
                    current_work_partition = 0.0;
                    for &elem_id_in_queue in &queue {
                        assigned_elements.remove(&elem_id_in_queue);
                    }
                    queue.clear();
                }

                res[current_elem_id] = current_partition_idx;
                current_work_partition += elem_work;

                for &neighbor_elem_id in self.graph.row(current_elem_id) {
                    if assigned_elements.insert(neighbor_elem_id) {
                        queue.push_back(neighbor_elem_id);
                    }
                }
            }
            if queue.is_empty() {
                next_unassigned_elem_root = 0;
            }
        }

        Ok(res)
    }
    fn n_parts(&self) -> usize {
        self.n_parts
    }

    fn graph(&self) -> &CSRGraph {
        &self.graph
    }
}

pub struct BFSPartitionner {
    n_parts: usize,
    graph: CSRGraph,
    ids: Vec<usize>,
    weights: Vec<f64>,
}

impl Partitioner for BFSPartitionner {
    fn new<const D: usize, const C: usize, const F: usize, M: Mesh<D, C, F>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self>
    where
        Cell<C>: Simplex<C>,
        Face<F>: Simplex<F>,
    {
        let faces = msh.all_faces();
        let graph = msh.element_pairs(&faces);

        let centers = msh.gelems().map(|ge| cell_center(&ge));
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
        let target_weight = self.weights.iter().copied().sum::<f64>() / self.n_parts() as f64;
        let mut res = vec![0; self.weights.len()];
        let n_elems = self.weights.len();
        let mut assigned_elements: HashSet<usize> = HashSet::new();
        let mut current_partition_idx = 0;
        let mut current_work_partition = 0.0;
        let mut queue: VecDeque<usize> = VecDeque::new();

        let mut next_unassigned_elem_root = 0;

        while assigned_elements.len() < n_elems && current_partition_idx < self.n_parts() {
            while next_unassigned_elem_root < n_elems
                && assigned_elements.contains(&(next_unassigned_elem_root))
            {
                next_unassigned_elem_root += 1;
            }

            if next_unassigned_elem_root == n_elems {
                break;
            }

            let start_elem_id = next_unassigned_elem_root;
            queue.push_back(self.ids[start_elem_id]);
            assigned_elements.insert(self.ids[start_elem_id]);

            while let Some(current_elem_id) = queue.pop_front() {
                let elem_work = self.weights[current_elem_id];

                if current_work_partition + elem_work > target_weight
                    && current_partition_idx + 1 < self.n_parts()
                {
                    current_partition_idx += 1;
                    current_work_partition = 0.0;
                }

                res[current_elem_id] = current_partition_idx;
                current_work_partition += elem_work;

                for &neighbor_elem_id in self.graph.row(current_elem_id) {
                    if assigned_elements.insert(neighbor_elem_id) {
                        queue.push_back(neighbor_elem_id);
                    }
                }
            }
            if current_partition_idx + 1 < self.n_parts() && assigned_elements.len() < n_elems {
                current_partition_idx += 1;
                current_work_partition = 0.0;
            }
        }
        Ok(res)
    }

    fn n_parts(&self) -> usize {
        self.n_parts
    }

    fn graph(&self) -> &CSRGraph {
        &self.graph
    }
}
pub struct HilbertBallPartitioner {
    n_parts: usize,
    graph: CSRGraph,
    ids: Vec<usize>,
    weights: Vec<f64>,
    v2e: CSRGraph,
}
impl Partitioner for HilbertBallPartitioner {
    fn new<const D: usize, const C: usize, const F: usize, M: Mesh<D, C, F>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self>
    where
        Cell<C>: Simplex<C>,
        Face<F>: Simplex<F>,
    {
        let faces = msh.all_faces();
        let graph = msh.element_pairs(&faces);

        let ids = hilbert_indices(msh.verts());
        let weights = weights.unwrap_or_else(|| vec![1.0; msh.n_elems()]);
        Ok(Self {
            n_parts,
            graph,
            ids,
            weights,
            v2e: msh.vertex_to_elems(),
        })
    }

    fn compute(&self) -> Result<Vec<usize>> {
        let target_weight = self.weights.iter().copied().sum::<f64>() / self.n_parts as f64;

        let mut partition = vec![usize::MAX; self.weights.len()];
        let mut current_partition_idx = 0;
        let mut current_work_partition = 0.0;

        //To parallelize
        for &i_vert in &self.ids {
            let element_in_ball = self.v2e.row(i_vert);
            for &i_elem in element_in_ball {
                if partition[i_elem] == usize::MAX {
                    let elem_work = self.weights[i_elem];
                    if (current_work_partition + elem_work) > target_weight {
                        current_work_partition = elem_work; // change
                        if current_partition_idx < self.n_parts - 1 {
                            current_partition_idx += 1;
                        }
                    } else {
                        current_work_partition += elem_work;
                    }
                    partition[i_elem] = current_partition_idx;
                }
            }
        }
        Ok(partition)
    }

    fn n_parts(&self) -> usize {
        self.n_parts
    }

    fn graph(&self) -> &CSRGraph {
        &self.graph
    }
}
/// Simple geometric partitionner based on the Hilbert indices of the element centers
pub struct HilbertPartitioner {
    n_parts: usize,
    graph: CSRGraph,
    ids: Vec<usize>,
    weights: Vec<f64>,
}

impl Partitioner for HilbertPartitioner {
    fn new<const D: usize, const C: usize, const F: usize, M: Mesh<D, C, F>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self>
    where
        Cell<C>: Simplex<C>,
        Face<F>: Simplex<F>,
    {
        let faces = msh.all_faces();
        let graph = msh.element_pairs(&faces);

        let centers = msh.gelems().map(|ge| cell_center(&ge));
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
    fn new<const D: usize, const C: usize, const F: usize, M: Mesh<D, C, F>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self>
    where
        Cell<C>: Simplex<C>,
        Face<F>: Simplex<F>,
    {
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
}

/// KMeans partitionner based on `coupe` (2d)
pub struct KMeansPartitioner2d {
    n_parts: usize,
    graph: CSRGraph,
    centers: Vec<SVector<f64, 2>>,
    weights: Vec<f64>,
}
impl Partitioner for KMeansPartitioner2d {
    fn new<const D: usize, const C: usize, const F: usize, M: Mesh<D, C, F>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self>
    where
        Cell<C>: Simplex<C>,
        Face<F>: Simplex<F>,
    {
        match D {
            2 => {
                let faces = msh.all_faces();
                let graph = msh.element_pairs(&faces);

                let centers = msh
                    .gelems()
                    .map(|ge| SVector::from_row_slice(cell_center(&ge).as_slice()))
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

    fn n_parts(&self) -> usize {
        self.n_parts
    }

    fn graph(&self) -> &CSRGraph {
        &self.graph
    }
}

/// KMeans partitionner based on `coupe` (3d)
pub struct KMeansPartitioner3d {
    n_parts: usize,
    graph: CSRGraph,
    centers: Vec<SVector<f64, 3>>,
    weights: Vec<f64>,
}

impl Partitioner for KMeansPartitioner3d {
    fn new<const D: usize, const C: usize, const F: usize, M: Mesh<D, C, F>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self>
    where
        Cell<C>: Simplex<C>,
        Face<F>: Simplex<F>,
    {
        match D {
            3 => {
                let faces = msh.all_faces();
                let graph = msh.element_pairs(&faces);

                let centers = msh
                    .gelems()
                    .map(|ge| SVector::from_row_slice(cell_center(&ge).as_slice()))
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
pub trait MetisPartMethod {
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
    #[allow(dead_code)]
    weights: Vec<f64>,
    t: PhantomData<T>,
}

#[cfg(feature = "metis")]
impl<T: MetisPartMethod + std::marker::Send + std::marker::Sync> Partitioner
    for MetisPartitioner<T>
{
    fn new<const D: usize, const C: usize, const F: usize, M: Mesh<D, C, F>>(
        msh: &M,
        n_parts: usize,
        weights: Option<Vec<f64>>,
    ) -> Result<Self>
    where
        Cell<C>: Simplex<C>,
        Face<F>: Simplex<F>,
    {
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
    fn compute(&self) -> Result<Vec<usize>> {
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
    #[cfg(feature = "metis")]
    use crate::mesh::partition::{MetisPartitioner, MetisRecursive};
    use crate::mesh::{
        Mesh, Mesh2d, Mesh3d, box_mesh,
        partition::{
            BFSPartitionner, HilbertBallPartitioner, HilbertPartitioner, KMeansPartitioner2d,
            KMeansPartitioner3d, Partitioner, RCMPartitioner,
        },
        rectangle_mesh,
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
    #[test]
    fn test_hilbertb() {
        let msh: Mesh3d = box_mesh(1.0, 10, 1.0, 15, 1.0, 20);
        let msh = msh.random_shuffle();

        let partitioner = HilbertBallPartitioner::new(&msh, 4, None).unwrap();
        let mut parts = partitioner.compute().unwrap();
        partitioner.partition_correction(&mut parts);
        assert!(partitioner.partition_quality(&parts) < 0.1);
        assert!(partitioner.partition_imbalance(&parts) < 0.008);
    }
    #[test]
    fn test_bfs() {
        let msh: Mesh3d = box_mesh(1.0, 10, 1.0, 15, 1.0, 20);
        let msh = msh.random_shuffle();

        let partitioner = BFSPartitionner::new(&msh, 4, None).unwrap();
        let mut parts = partitioner.compute().unwrap();
        partitioner.partition_correction(&mut parts);
        assert!(partitioner.partition_quality(&parts) < 0.1);
        assert!(partitioner.partition_imbalance(&parts) < 0.1);
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
    fn test_coupe_kmeans2d() {
        let msh: Mesh2d = rectangle_mesh(1.0, 5, 1.0, 6);
        let msh = msh.random_shuffle();

        let partitioner = KMeansPartitioner2d::new(&msh, 4, None).unwrap();
        let parts = partitioner.compute().unwrap();

        assert!(partitioner.partition_quality(&parts) < 0.2);
        assert!(partitioner.partition_imbalance(&parts) < 0.41);
    }

    #[test]
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
