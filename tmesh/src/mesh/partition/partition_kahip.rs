use super::{Mesh, Partitioner, init_or_check_weights};
use crate::{Result, graph::CSRGraph};
use kahip::{
    KaHIPGraph, KaMinParGraph, KahipMode, KahipParams, KaminparOutputLevel, KaminparParams,
};
use std::marker::PhantomData;

fn scale_vertex_weights_to_kahip(weights: &[f64]) -> Vec<kahip::Idx> {
    let max_weight = weights.iter().copied().fold(0.0_f64, f64::max);
    if max_weight == 0.0 {
        return vec![1; weights.len()];
    }

    let target_max = kahip::Idx::MAX as f64;
    let scale = if max_weight > target_max {
        target_max / max_weight
    } else {
        1.0
    };

    weights
        .iter()
        .map(|&w| {
            assert!(w.is_finite(), "vertex weights must be finite");
            assert!(w >= 0.0, "vertex weights must be non-negative");
            let v = (w * scale).round().max(1.0).min(target_max);
            v as kahip::Idx
        })
        .collect()
}

fn scale_vertex_weights_to_kaminpar(weights: &[f64]) -> Vec<kahip::KaminparNodeWeight> {
    let max_weight = weights.iter().copied().fold(0.0_f64, f64::max);
    if max_weight == 0.0 {
        return vec![1; weights.len()];
    }

    let target_max = kahip::KaminparNodeWeight::MAX as f64;
    let scale = if max_weight > target_max {
        target_max / max_weight
    } else {
        1.0
    };

    weights
        .iter()
        .map(|&w| {
            assert!(w.is_finite(), "vertex weights must be finite");
            assert!(w >= 0.0, "vertex weights must be non-negative");
            let v = (w * scale).round().max(1.0).min(target_max);
            v as kahip::KaminparNodeWeight
        })
        .collect()
}

/// KaHIP partitioning method
pub enum KahipMethod {
    /// Eco mode in KaHIP
    Eco,
    /// Fast mode in KaHIP
    Fast,
    /// Strong mode in KaHIP
    Strong,
}

/// KaHIP partitioning method selector
pub trait KahipPartMethod: Send + Sync {
    /// KaHIP partitioning method
    fn method() -> KahipMethod;
}

/// KaHIP Eco mode selector
pub struct KahipEco;

impl KahipPartMethod for KahipEco {
    fn method() -> KahipMethod {
        KahipMethod::Eco
    }
}

/// KaHIP Fast mode selector
pub struct KahipFast;

impl KahipPartMethod for KahipFast {
    fn method() -> KahipMethod {
        KahipMethod::Fast
    }
}

/// KaHIP Strong mode selector
pub struct KahipStrong;

impl KahipPartMethod for KahipStrong {
    fn method() -> KahipMethod {
        KahipMethod::Strong
    }
}

/// KaHIP partitioner
pub struct KaHIPPartitioner<T: KahipPartMethod> {
    n_parts: usize,
    graph: CSRGraph,
    weights: Vec<f64>,
    t: PhantomData<T>,
}

impl<T: KahipPartMethod> Partitioner for KaHIPPartitioner<T> {
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

        let mut xadj = Vec::<kahip::Idx>::with_capacity(self.graph.n() + 1);
        let mut adjncy = Vec::<kahip::Idx>::with_capacity(self.graph.n_edges());
        let mut vwgt = scale_vertex_weights_to_kahip(&self.weights);

        xadj.push(0);
        for row in self.graph.rows() {
            for &j in row {
                adjncy.push(j.try_into().unwrap());
            }
            xadj.push(adjncy.len().try_into().unwrap());
        }

        let mode = match T::method() {
            KahipMethod::Eco => KahipMode::Eco,
            KahipMethod::Fast => KahipMode::Fast,
            KahipMethod::Strong => KahipMode::Strong,
        };

        let mut kahip_graph = KaHIPGraph::new(&mut xadj, &mut adjncy).set_vwgt(&mut vwgt);
        let (partition, _) = kahip_graph.partition(
            self.n_parts.try_into().unwrap(),
            KahipParams {
                mode,
                ..KahipParams::default()
            },
        );

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

/// KaMinPar partitioning method
pub enum KaMinParMethod {
    /// Default preset in KaMinPar
    Default,
    /// Strong preset in KaMinPar
    Strong,
    /// TeraPart preset in KaMinPar
    TeraPart,
    /// LargeK preset in KaMinPar
    LargeK,
    /// VCycle preset in KaMinPar
    VCycle,
}

/// KaMinPar partitioning method selector
pub trait KaMinParPartMethod: Send + Sync {
    /// KaMinPar partitioning method
    fn method() -> KaMinParMethod;
}

/// KaMinPar default preset selector
pub struct KaMinParDefault;

impl KaMinParPartMethod for KaMinParDefault {
    fn method() -> KaMinParMethod {
        KaMinParMethod::Default
    }
}

/// KaMinPar strong preset selector
pub struct KaMinParStrong;

impl KaMinParPartMethod for KaMinParStrong {
    fn method() -> KaMinParMethod {
        KaMinParMethod::Strong
    }
}

/// KaMinPar TeraPart preset selector
pub struct KaMinParTeraPart;

impl KaMinParPartMethod for KaMinParTeraPart {
    fn method() -> KaMinParMethod {
        KaMinParMethod::TeraPart
    }
}

/// KaMinPar LargeK preset selector
pub struct KaMinParLargeK;

impl KaMinParPartMethod for KaMinParLargeK {
    fn method() -> KaMinParMethod {
        KaMinParMethod::LargeK
    }
}

/// KaMinPar VCycle preset selector
pub struct KaMinParVCycle;

impl KaMinParPartMethod for KaMinParVCycle {
    fn method() -> KaMinParMethod {
        KaMinParMethod::VCycle
    }
}

/// KaMinPar partitioner
pub struct KMinParPartitioner<T: KaMinParPartMethod> {
    n_parts: usize,
    graph: CSRGraph,
    weights: Vec<f64>,
    t: PhantomData<T>,
}

impl<T: KaMinParPartMethod> Partitioner for KMinParPartitioner<T> {
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

        let mut xadj = Vec::<kahip::KaminparEdgeId>::with_capacity(self.graph.n() + 1);
        let mut adjncy = Vec::<kahip::KaminparNodeId>::with_capacity(self.graph.n_edges());
        let mut vwgt = scale_vertex_weights_to_kaminpar(&self.weights);

        xadj.push(0);
        for row in self.graph.rows() {
            for &j in row {
                adjncy.push(j.try_into().unwrap());
            }
            xadj.push(adjncy.len().try_into().unwrap());
        }

        let preset = match T::method() {
            KaMinParMethod::Default => kahip::KaminparPreset::Default,
            KaMinParMethod::Strong => kahip::KaminparPreset::Strong,
            KaMinParMethod::TeraPart => kahip::KaminparPreset::TeraPart,
            KaMinParMethod::LargeK => kahip::KaminparPreset::LargeK,
            KaMinParMethod::VCycle => kahip::KaminparPreset::VCycle,
        };

        let mut kminpar_graph = KaMinParGraph::new(&mut xadj, &mut adjncy).set_vwgt(&mut vwgt);
        let num_threads = rayon::current_num_threads().try_into().unwrap();
        let (partition, _) = kminpar_graph.partition_with_epsilon(
            self.n_parts.try_into().unwrap(),
            KaminparParams {
                preset,
                num_threads,
                output_level: KaminparOutputLevel::Quiet,
                ..KaminparParams::default()
            },
        );

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
