use rustc_hash::FxHashMap;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, hash_map::Entry},
    fmt::Debug,
    hash::Hash,
};

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

pub struct OrderedHashMap<OrdKey, HashKey, Value> {
    tree_map: BTreeMap<(OrdKey, HashKey), Value>,
    hash_map: FxHashMap<HashKey, OrdKey>,
}

impl<O, K, M> Default for OrderedHashMap<O, K, M> {
    fn default() -> Self {
        Self {
            tree_map: BTreeMap::default(),
            hash_map: FxHashMap::default(),
        }
    }
}

impl<OrdKey, HashKey, Value> OrderedHashMap<OrdKey, HashKey, Value>
where
    OrdKey: Ord + Copy + Debug,
    HashKey: Eq + Hash + Copy + Ord,
{
    pub fn insert<OK>(&mut self, hash_key: HashKey, ord_key: OK, value: Value)
    where
        OrdKey: From<OK>,
    {
        self.update(hash_key, Some((ord_key, value)));
    }

    pub fn pop_last<OK>(&mut self) -> Option<(HashKey, Value, OK)>
    where
        OK: From<OrdKey>,
    {
        match self.tree_map.pop_last() {
            Some(((ord_key, hash_key), value)) => {
                let ov = self.hash_map.remove(&hash_key);
                debug_assert_eq!(Some(&ord_key), ov.as_ref());
                Some((hash_key, value, ord_key.into()))
            }
            None => None,
        }
    }

    pub fn update<OK>(&mut self, hash_key: HashKey, value: Option<(OK, Value)>)
    where
        OK: Into<OrdKey>,
    {
        match (self.hash_map.entry(hash_key), value) {
            (Entry::Occupied(mut entry), Some((ord_key, value))) => {
                // replace
                let ord_key = ord_key.into();
                self.tree_map.remove(&(*entry.get(), hash_key));
                entry.insert(ord_key);
                self.tree_map.insert((ord_key, hash_key), value);
            }
            (Entry::Occupied(entry), None) => {
                // remove
                let present = self.tree_map.remove(&(*entry.get(), hash_key));
                debug_assert!(present.is_some());
                entry.remove();
            }
            (Entry::Vacant(entry), Some((ord_key, value))) => {
                // insert
                let ord_key = ord_key.into();
                entry.insert(ord_key);
                self.tree_map.insert((ord_key, hash_key), value);
            }
            (Entry::Vacant(_), None) => {}
        }
        debug_assert_eq!(self.tree_map.len(), self.hash_map.len());
    }
}
