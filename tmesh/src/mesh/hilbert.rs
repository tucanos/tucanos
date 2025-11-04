use crate::{Vertex, mesh::Idx};
use lindel::Lineariseable;

/// Get the bounding box
#[must_use]
fn bounding_box<const D: usize, I: Iterator<Item = Vertex<D>>>(
    mut verts: I,
) -> (Vertex<D>, Vertex<D>) {
    let first = verts.next().unwrap();
    let mut mini = first;
    let mut maxi = first;
    for p in verts {
        for j in 0..D {
            mini[j] = f64::min(mini[j], p[j]);
            maxi[j] = f64::max(maxi[j], p[j]);
        }
    }
    (mini, maxi)
}

/// Get the Hilbert indices
#[must_use]
pub fn hilbert_indices<T: Idx, const D: usize, I: ExactSizeIterator<Item = Vertex<D>> + Clone>(
    verts: I,
) -> Vec<T> {
    let n = verts.len();

    // bounding box
    let (mini, maxi) = bounding_box(verts.clone());

    // Hilbert index
    let order = 16;
    let scale = usize::pow(2, order) as f64 - 1.0;
    let hilbert = |x: Vertex<D>| {
        let mut tmp = [0; 3];
        for j in 0..D {
            tmp[j] = (scale * (x[j] - mini[j]) / (maxi[j] - mini[j])).round() as u16;
        }
        tmp.hilbert_index() as usize
    };

    let hilbert_ids = verts.map(hilbert).collect::<Vec<_>>();

    let mut indices = (0..n).map(|x| x.into()).collect::<Vec<T>>();
    indices.sort_by_key(|&i| hilbert_ids[i.into()]);
    indices
}
