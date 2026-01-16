use rustc_hash::FxHashMap;
use std::{
    cmp::Ordering,
    collections::{BTreeSet, hash_map::Entry},
    fmt::Debug,
    hash::Hash,
};

struct Value<K, M, O> {
    ord: O,
    meta: M,
    key: K,
}

pub trait AnyOrd: Ord {}

impl<T: Ord> AnyOrd for T {}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct OrdF64(f64);

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for OrdF64 {}

impl From<f64> for OrdF64 {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

impl From<OrdF64> for f64 {
    fn from(value: OrdF64) -> Self {
        value.0
    }
}

impl<K, M, O> Ord for Value<K, M, O>
where
    O: AnyOrd,
    K: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.ord
            .cmp(&other.ord)
            .then_with(|| self.key.cmp(&other.key))
    }
}

impl<K, M, O> PartialOrd for Value<K, M, O>
where
    O: AnyOrd,
    K: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K, M, O> PartialEq for Value<K, M, O>
where
    O: AnyOrd,
    K: Ord,
{
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<K, M, O> Eq for Value<K, M, O>
where
    O: AnyOrd,
    K: Ord,
{
}

impl<K, M, O> From<(O, K)> for Value<K, M, O>
where
    M: Default,
{
    fn from(value: (O, K)) -> Self {
        Self {
            ord: value.0,
            meta: M::default(),
            key: value.1,
        }
    }
}

pub struct OrderedHashMap<K, M, O> {
    tree: BTreeSet<Value<K, M, O>>,
    map: FxHashMap<K, O>,
}

impl<K, M, O> Default for OrderedHashMap<K, M, O> {
    fn default() -> Self {
        Self {
            tree: BTreeSet::default(),
            map: FxHashMap::default(),
        }
    }
}

impl<K, M, O> OrderedHashMap<K, M, O>
where
    O: AnyOrd + Copy + Debug,
    K: Eq + Hash + Copy + Ord,
    M: Default,
{
    pub fn insert<IO>(&mut self, key: K, ord: IO, meta: M)
    where
        O: From<IO>,
    {
        let ord = ord.into();
        self.tree.insert(Value { ord, meta, key });
        self.map.insert(key, ord);
    }

    pub fn pop_last<FO>(&mut self) -> Option<(K, M, FO)>
    where
        FO: From<O>,
    {
        match self.tree.pop_last() {
            Some(v) => {
                let ov = self.map.remove(&v.key);
                debug_assert_eq!(Some(&v.ord), ov.as_ref());
                Some((v.key, v.meta, v.ord.into()))
            }
            None => None,
        }
    }

    pub fn update<IO>(&mut self, key: K, value: Option<(M, IO)>)
    where
        IO: Into<O>,
    {
        match (self.map.entry(key), value) {
            (Entry::Occupied(mut ve), Some((meta, ord))) => {
                // replace
                let ord = ord.into();
                self.tree.remove(&(*ve.get(), key).into());
                ve.insert(ord);
                self.tree.insert(Value { ord, meta, key });
            }
            (Entry::Occupied(ve), None) => {
                // remove
                let present = self.tree.remove(&(*ve.get(), key).into());
                debug_assert!(present);
                ve.remove();
            }
            (Entry::Vacant(ve), Some((meta, ord))) => {
                // insert
                let ord = ord.into();
                ve.insert(ord);
                self.tree.insert(Value { ord, meta, key });
            }
            (Entry::Vacant(_), None) => {}
        }
        debug_assert_eq!(self.tree.len(), self.map.len());
    }
}
